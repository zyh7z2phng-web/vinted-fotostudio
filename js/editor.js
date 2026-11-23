// Nachbessern: Maske per Pinsel radieren/zurückholen, Pinch-Zoom, Zwei-Finger-Verschieben.

const HARDNESS = 0.55;
const MAX_UNDO = 25;

export class Editor {
  constructor(root) {
    this.root = root;
    this.canvas = root.querySelector('#edCanvas');
    this.ctx = this.canvas.getContext('2d');
    this.sizeInput = root.querySelector('#edSize');
    this.modeButtons = [...root.querySelectorAll('[data-mode]')];
    this.mode = 'erase';
    this.pointers = new Map();
    this.view = { s: 1, tx: 0, ty: 0 };
    this.cursor = null;
    this.frame = 0;

    this.modeButtons.forEach((b) => b.addEventListener('click', () => this.setMode(b.dataset.mode)));
    root.querySelector('#edUndo').addEventListener('click', () => this.undo());
    root.querySelector('#edFit').addEventListener('click', () => { this.fit(); this.render(); });
    root.querySelector('#edCancel').addEventListener('click', () => this.close(null));
    root.querySelector('#edDone').addEventListener('click', () => this.close(this.alpha.slice()));
    this.sizeInput.addEventListener('input', () => this.render());

    const c = this.canvas;
    c.addEventListener('pointerdown', (e) => this.onDown(e));
    c.addEventListener('pointermove', (e) => this.onMove(e));
    c.addEventListener('pointerup', (e) => this.onUp(e));
    c.addEventListener('pointercancel', (e) => this.onUp(e, true));
    c.addEventListener('pointerleave', () => { if (!this.pointers.size) { this.cursor = null; this.render(); } });
    c.addEventListener('wheel', (e) => this.onWheel(e), { passive: false });
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    // iOS Safari: eigene Gesten (Seiten-Zoom) im Editor unterdrücken
    ['gesturestart', 'gesturechange'].forEach((t) => c.addEventListener(t, (e) => e.preventDefault()));
    this.onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); this.undo(); }
      if (e.key === 'Escape') this.close(null);
    };
    this.ro = new ResizeObserver(() => this.resize());
  }

  /** @returns Promise<Uint8Array|null> neue Maske oder null bei Abbruch */
  open(imageData, alpha, w, h) {
    this.w = w; this.h = h;
    this.alpha = alpha;
    this.cov = new Uint8Array(w * h);
    this.undoStack = [];
    this.stroke = null; this.gesture = null; this.pointers.clear();

    this.imgCanvas = makeCanvas(w, h);
    this.imgCanvas.getContext('2d').putImageData(imageData, 0, 0);
    this.prodData = new ImageData(new Uint8ClampedArray(imageData.data), w, h);
    this.prodCanvas = makeCanvas(w, h);
    this.prodCtx = this.prodCanvas.getContext('2d');
    this.updateProduct(0, 0, w, h);

    this.root.hidden = false;
    document.body.style.overflow = 'hidden';
    this.setMode('erase');
    this.ro.observe(this.canvas);
    window.addEventListener('keydown', this.onKey);
    this.resize(true);
    return new Promise((res) => { this.resolve = res; });
  }

  close(result) {
    this.root.hidden = true;
    document.body.style.overflow = '';
    this.ro.disconnect();
    window.removeEventListener('keydown', this.onKey);
    for (const c of [this.imgCanvas, this.prodCanvas]) if (c) { c.width = c.height = 1; }
    this.imgCanvas = this.prodCanvas = this.prodData = null;
    this.undoStack = [];
    const r = this.resolve; this.resolve = null;
    if (r) r(result);
  }

  setMode(m) {
    this.mode = m;
    this.modeButtons.forEach((b) => b.setAttribute('aria-checked', String(b.dataset.mode === m)));
  }

  resize(refit) {
    const dpr = Math.min(3, window.devicePixelRatio || 1);
    const r = this.canvas.getBoundingClientRect();
    this.cw = r.width; this.ch = r.height; this.dpr = dpr;
    this.canvas.width = Math.max(1, Math.round(r.width * dpr));
    this.canvas.height = Math.max(1, Math.round(r.height * dpr));
    if (refit || !this.fitted) this.fit();
    this.render();
  }

  fit() {
    const s = Math.min(this.cw / this.w, this.ch / this.h) * 0.94;
    this.view = { s, tx: (this.cw - this.w * s) / 2, ty: (this.ch - this.h * s) / 2 };
    this.fitScale = s;
    this.fitted = this.cw > 0;
  }

  /* ---------------------------------------------------------- Zeichnen */

  render() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => { this.frame = 0; this.draw(); });
  }

  draw() {
    if (!this.prodCanvas) return;
    const { ctx, dpr } = this;
    const { s, tx, ty } = this.view;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#2a2b2d';
    ctx.fillRect(0, 0, this.cw, this.ch);
    // Schachbrett = "wird entfernt"
    if (!this.pattern) this.pattern = ctx.createPattern(checker(), 'repeat');
    ctx.save();
    ctx.translate(tx, ty);
    ctx.fillStyle = this.pattern;
    ctx.fillRect(0, 0, this.w * s, this.h * s);
    ctx.scale(s, s);
    ctx.imageSmoothingEnabled = s < 3;
    ctx.globalAlpha = 0.3; // entfernte Bereiche blass sichtbar, damit man sie zurückholen kann
    ctx.drawImage(this.imgCanvas, 0, 0);
    ctx.globalAlpha = 1;
    ctx.drawImage(this.prodCanvas, 0, 0);
    ctx.restore();
    if (this.cursor && !this.gesture) {
      const r = this.brushScreenRadius();
      ctx.beginPath();
      ctx.arc(this.cursor.x, this.cursor.y, r, 0, Math.PI * 2);
      ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(0,0,0,0.6)'; ctx.stroke();
      ctx.lineWidth = 1; ctx.strokeStyle = this.mode === 'erase' ? '#ff8a80' : '#7ee0c8'; ctx.stroke();
    }
  }

  updateProduct(x0, y0, x1, y1) {
    x0 = Math.max(0, x0 | 0); y0 = Math.max(0, y0 | 0);
    x1 = Math.min(this.w, Math.ceil(x1)); y1 = Math.min(this.h, Math.ceil(y1));
    if (x1 <= x0 || y1 <= y0) return;
    const d = this.prodData.data, a = this.alpha, w = this.w;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) d[(y * w + x) * 4 + 3] = a[y * w + x];
    this.prodCtx.putImageData(this.prodData, 0, 0, x0, y0, x1 - x0, y1 - y0);
  }

  brushScreenRadius() { return +this.sizeInput.value / 2; }

  toImage(p) { return { x: (p.x - this.view.tx) / this.view.s, y: (p.y - this.view.ty) / this.view.s }; }

  local(e) { const r = this.canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; }

  /* ---------------------------------------------------------- Pinsel */

  startStroke(p) {
    this.undoStack.push(this.alpha.slice());
    if (this.undoStack.length > MAX_UNDO) this.undoStack.shift();
    this.stroke = { base: this.undoStack[this.undoStack.length - 1], last: null, t: performance.now(), n: 0, box: null };
    this.strokeTo(p);
  }

  strokeTo(p) {
    const st = this.stroke;
    const ip = this.toImage(p);
    const r = Math.max(1, this.brushScreenRadius() / this.view.s);
    if (!st.last) this.stamp(ip.x, ip.y, r);
    else {
      const dx = ip.x - st.last.x, dy = ip.y - st.last.y;
      const dist = Math.hypot(dx, dy), step = Math.max(0.5, r * 0.25);
      const n = Math.ceil(dist / step);
      for (let i = 1; i <= n; i++) this.stamp(st.last.x + (dx * i) / n, st.last.y + (dy * i) / n, r);
    }
    st.last = ip; st.n++;
    this.render();
  }

  stamp(cx, cy, r) {
    const { w, h, cov, alpha } = this;
    const base = this.stroke.base, erase = this.mode === 'erase';
    const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(w - 1, Math.ceil(cx + r));
    const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(h - 1, Math.ceil(cy + r));
    if (x1 < x0 || y1 < y0) return;
    for (let y = y0; y <= y1; y++) {
      for (let x = x0; x <= x1; x++) {
        const d = Math.hypot(x + 0.5 - cx, y + 0.5 - cy) / r;
        if (d >= 1) continue;
        let c = d <= HARDNESS ? 1 : 1 - (d - HARDNESS) / (1 - HARDNESS);
        c = c * c * (3 - 2 * c);
        const i = y * w + x, cv = Math.round(c * 255);
        if (cv <= cov[i]) continue;
        cov[i] = cv;
        const b = base[i], k = cv / 255;
        alpha[i] = erase ? Math.round(b * (1 - k)) : Math.round(b + (255 - b) * k);
      }
    }
    const st = this.stroke;
    st.box = st.box ? [Math.min(st.box[0], x0), Math.min(st.box[1], y0), Math.max(st.box[2], x1 + 1), Math.max(st.box[3], y1 + 1)] : [x0, y0, x1 + 1, y1 + 1];
    this.updateProduct(x0, y0, x1 + 1, y1 + 1);
  }

  endStroke() {
    const st = this.stroke; if (!st) return;
    if (st.box) { const [x0, y0, x1, y1] = st.box; for (let y = y0; y < y1; y++) this.cov.fill(0, y * this.w + x0, y * this.w + x1); }
    this.stroke = null;
  }

  cancelStroke() {
    const st = this.stroke; if (!st) return;
    this.alpha.set(st.base);
    this.undoStack.pop();
    if (st.box) this.updateProduct(...st.box);
    this.endStroke();
  }

  undo() {
    if (this.stroke || !this.undoStack.length) return;
    this.alpha.set(this.undoStack.pop());
    this.updateProduct(0, 0, this.w, this.h);
    this.render();
  }

  /* ---------------------------------------------------------- Eingabe */

  onDown(e) {
    e.preventDefault();
    try { this.canvas.setPointerCapture(e.pointerId); } catch {}
    const p = this.local(e);
    this.pointers.set(e.pointerId, p);
    if (this.pointers.size === 1) {
      this.ignore = false;
      if (e.pointerType === 'mouse' && (e.button === 1 || e.button === 2 || e.altKey || e.shiftKey)) {
        this.pan = { start: p, tx: this.view.tx, ty: this.view.ty };
      } else {
        this.startStroke(p);
      }
    } else if (this.pointers.size === 2) {
      // zweiter Finger: Malen abbrechen (falls gerade erst begonnen) und Geste starten
      if (this.stroke) {
        if (performance.now() - this.stroke.t < 400 || this.stroke.n < 6) this.cancelStroke();
        else this.endStroke();
      }
      this.startGesture();
    }
  }

  onMove(e) {
    const p = this.local(e);
    if (e.pointerType !== 'touch') this.cursor = p;
    if (!this.pointers.has(e.pointerId)) { this.render(); return; }
    this.pointers.set(e.pointerId, p);
    if (e.pointerType === 'touch') this.cursor = p;
    if (this.gesture) this.updateGesture();
    else if (this.pan) { this.view.tx = this.pan.tx + p.x - this.pan.start.x; this.view.ty = this.pan.ty + p.y - this.pan.start.y; this.render(); }
    else if (this.stroke && !this.ignore) {
      const evs = e.getCoalescedEvents ? e.getCoalescedEvents() : [];
      if (evs.length > 1) for (const ce of evs) this.strokeTo(this.local(ce));
      else this.strokeTo(p);
    } else this.render();
  }

  onUp(e, cancelled) {
    this.pointers.delete(e.pointerId);
    if (this.gesture) {
      if (this.pointers.size < 2) { this.gesture = null; this.ignore = true; }
    } else if (this.stroke && this.pointers.size === 0) {
      if (cancelled) this.cancelStroke(); else this.endStroke();
    }
    if (this.pointers.size === 0) { this.pan = null; this.ignore = false; if (e.pointerType === 'touch') this.cursor = null; }
    this.render();
  }

  startGesture() {
    const [a, b] = [...this.pointers.values()];
    const m = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    this.gesture = { d0: Math.hypot(a.x - b.x, a.y - b.y) || 1, m0: m, view: { ...this.view } };
  }

  updateGesture() {
    const pts = [...this.pointers.values()];
    if (pts.length < 2) return;
    const [a, b] = pts;
    const g = this.gesture;
    const d = Math.hypot(a.x - b.x, a.y - b.y) || 1;
    const m = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    const s = clamp(g.view.s * (d / g.d0), this.fitScale * 0.5, this.fitScale * 16);
    const ix = (g.m0.x - g.view.tx) / g.view.s, iy = (g.m0.y - g.view.ty) / g.view.s;
    this.view = { s, tx: m.x - ix * s, ty: m.y - iy * s };
    this.render();
  }

  onWheel(e) {
    e.preventDefault();
    const p = this.local(e);
    if (e.ctrlKey || e.deltaMode !== 0 || Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
      const f = Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015));
      const s = clamp(this.view.s * f, this.fitScale * 0.5, this.fitScale * 16);
      const ix = (p.x - this.view.tx) / this.view.s, iy = (p.y - this.view.ty) / this.view.s;
      this.view = { s, tx: p.x - ix * s, ty: p.y - iy * s };
    } else { this.view.tx -= e.deltaX; }
    this.cursor = p;
    this.render();
  }
}

function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

function checker() {
  const c = makeCanvas(16, 16), x = c.getContext('2d');
  x.fillStyle = '#ffffff'; x.fillRect(0, 0, 16, 16);
  x.fillStyle = '#e4e4e4'; x.fillRect(0, 0, 8, 8); x.fillRect(8, 8, 8, 8);
  return c;
}
