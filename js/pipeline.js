// Vinted Fotostudio – Bildverarbeitungs-Pipeline (reines JavaScript, ohne DOM).
// Läuft im Web Worker und zum Testen auch in Node.
//
// Ablauf pro Foto:
//   1. prepareInput      – Arbeitsbild (≤1500 px) -> 1024x1024 Tensor für ISNet
//   2. postprocessMask   – Modellausgabe -> Alpha in Arbeitsauflösung
//   3. cleanupAlpha      – kleine Inseln / Schleier entfernen
//   4. refineHi          – Alpha auf Exportauflösung hochrechnen, mit (schnellem)
//                          Guided Filter an echte Bildkanten anlegen, Randfarben
//                          dekontaminieren (Blur-Fusion-Vordergrundschätzung)
//   5. compose           – zuschneiden, zentrieren, Schatten, weißer Hintergrund

export const MODEL_SIZE = 1024;
export const WORK_MAX = 1500;

/* ---------------------------------------------------------------- Grundlagen */

/** Box-Filter (Mittelwert) mit Radius r, Ränder werden über die echte Fensterbreite normiert. */
export function boxBlur(src, w, h, r) {
  r = Math.round(r);
  const n = w * h;
  if (r < 1) return Float32Array.from(src);
  const tmp = new Float32Array(n);
  // horizontal
  for (let y = 0; y < h; y++) {
    const o = y * w;
    let sum = 0;
    const first = Math.min(w - 1, r);
    for (let i = 0; i <= first; i++) sum += src[o + i];
    for (let x = 0; x < w; x++) {
      const l = x - r, rr = x + r;
      tmp[o + x] = sum / (Math.min(w - 1, rr) - Math.max(0, l) + 1);
      if (rr + 1 < w) sum += src[o + rr + 1];
      if (l >= 0) sum -= src[o + l];
    }
  }
  // vertikal (zeilenweise Akkumulatoren, cachefreundlich)
  const out = new Float32Array(n);
  const acc = new Float64Array(w);
  const firstRow = Math.min(h - 1, r);
  for (let y = 0; y <= firstRow; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) acc[x] += tmp[o + x];
  }
  for (let y = 0; y < h; y++) {
    const cnt = Math.min(h - 1, y + r) - Math.max(0, y - r) + 1;
    const inv = 1 / cnt, o = y * w;
    for (let x = 0; x < w; x++) out[o + x] = acc[x] * inv;
    const add = y + r + 1, sub = y - r;
    if (add < h) { const oa = add * w; for (let x = 0; x < w; x++) acc[x] += tmp[oa + x]; }
    if (sub >= 0) { const os = sub * w; for (let x = 0; x < w; x++) acc[x] -= tmp[os + x]; }
  }
  return out;
}

function boxesForGauss(sigma, n = 3) {
  const wIdeal = Math.sqrt((12 * sigma * sigma) / n + 1);
  let wl = Math.floor(wIdeal);
  if (wl % 2 === 0) wl--;
  const wu = wl + 2;
  const mIdeal = (12 * sigma * sigma - n * wl * wl - 4 * n * wl - 3 * n) / (-4 * wl - 4);
  const m = Math.round(mIdeal);
  const sizes = [];
  for (let i = 0; i < n; i++) sizes.push(i < m ? wl : wu);
  return sizes;
}

/** Gauß-Näherung über drei Box-Filter. */
export function gaussBlur(src, w, h, sigma) {
  if (sigma < 0.5) return Float32Array.from(src);
  let a = src;
  for (const s of boxesForGauss(sigma)) a = boxBlur(a, w, h, (s - 1) / 2);
  return a;
}

/**
 * Separables Resampling eines einzelnen Kanals (Dreiecksfilter, beim Verkleinern
 * mit verbreitertem Träger = Antialiasing, wie Pillow BILINEAR).
 */
export function resample(src, sw, sh, dw, dh) {
  if (sw === dw && sh === dh) return Float32Array.from(src);
  const tmp = new Float32Array(dw * sh);
  const hx = weights(sw, dw);
  for (let y = 0; y < sh; y++) {
    const so = y * sw, d = y * dw;
    for (let x = 0; x < dw; x++) {
      const { start, w: ws } = hx[x];
      let s = 0;
      for (let k = 0; k < ws.length; k++) s += src[so + start + k] * ws[k];
      tmp[d + x] = s;
    }
  }
  const out = new Float32Array(dw * dh);
  const hy = weights(sh, dh);
  for (let y = 0; y < dh; y++) {
    const { start, w: ws } = hy[y];
    const d = y * dw;
    for (let k = 0; k < ws.length; k++) {
      const wk = ws[k], o = (start + k) * dw;
      for (let x = 0; x < dw; x++) out[d + x] += tmp[o + x] * wk;
    }
  }
  return out;
}

const weightCache = new Map();
function weights(sn, dn) {
  const key = sn + 'x' + dn;
  if (weightCache.has(key)) return weightCache.get(key);
  const scale = dn / sn;
  const support = scale < 1 ? 1 / scale : 1;
  const res = [];
  for (let i = 0; i < dn; i++) {
    const center = (i + 0.5) / scale - 0.5;
    const start = Math.max(0, Math.ceil(center - support));
    const end = Math.min(sn - 1, Math.floor(center + support));
    const w = [];
    let sum = 0;
    for (let j = start; j <= end; j++) {
      const v = Math.max(0, 1 - Math.abs(j - center) / support);
      w.push(v); sum += v;
    }
    if (sum === 0) { // Randfall
      const j = Math.min(sn - 1, Math.max(0, Math.round(center)));
      res.push({ start: j, w: new Float32Array([1]) });
      continue;
    }
    res.push({ start, w: Float32Array.from(w, (v) => v / sum) });
  }
  if (weightCache.size > 64) weightCache.clear();
  weightCache.set(key, res);
  return res;
}

/** RGBA (Uint8, interleaved) -> planare Float-Kanäle 0..1 */
export function toPlanes(rgba, n) {
  const r = new Float32Array(n), g = new Float32Array(n), b = new Float32Array(n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    r[i] = rgba[j] / 255; g[i] = rgba[j + 1] / 255; b[i] = rgba[j + 2] / 255;
  }
  return [r, g, b];
}

/* ------------------------------------------------------------- Modell-I/O */

export const NORM = {
  isnet: { mean: [0.5, 0.5, 0.5], std: [1, 1, 1] },
  imagenet: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225] },
};

/** Arbeitsbild -> normalisierter NCHW-Tensor (ISNet: x/255 - 0.5, BiRefNet: ImageNet). */
export function prepareInput(rgba, w, h, norm = NORM.isnet) {
  const n = w * h;
  const planes = toPlanes(rgba, n);
  const N = MODEL_SIZE * MODEL_SIZE;
  const t = new Float32Array(3 * N);
  for (let c = 0; c < 3; c++) {
    const p = resample(planes[c], w, h, MODEL_SIZE, MODEL_SIZE);
    const o = c * N;
    const m = norm.mean[c], sd = norm.std[c];
    for (let i = 0; i < N; i++) t[o + i] = (p[i] - m) / sd;
  }
  return t;
}

/** Modellausgabe (1024²) -> Alpha in Arbeitsauflösung. mode: 'minmax' (ISNet) | 'sigmoid' (BiRefNet-Logits) */
export function postprocessMask(out, w, h, mode = 'minmax') {
  if (mode === 'sigmoid') {
    const m = new Float32Array(out.length);
    for (let i = 0; i < out.length; i++) m[i] = 1 / (1 + Math.exp(-out[i]));
    out = m;
  }
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < out.length; i++) { const v = out[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
  const range = mx - mn || 1;
  const m = new Float32Array(out.length);
  for (let i = 0; i < out.length; i++) m[i] = (out[i] - mn) / range;
  const a = resample(m, MODEL_SIZE, MODEL_SIZE, w, h);
  for (let i = 0; i < a.length; i++) a[i] = a[i] < 0 ? 0 : a[i] > 1 ? 1 : a[i];
  return a;
}

/* --------------------------------------------------------------- Aufräumen */

/**
 * Entfernt kleine, nicht zusammenhängende Flecken und den Restschleier im
 * Hintergrund. Große Teile (z.B. zwei Schuhe) bleiben erhalten.
 */
export function cleanupAlpha(alpha, w, h) {
  const n = w * h;
  const labels = new Int32Array(n).fill(-1);
  const areas = [];
  const queue = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    if (labels[i] !== -1 || alpha[i] < 0.5) continue;
    const id = areas.length;
    let head = 0, tail = 0, area = 0;
    queue[tail++] = i; labels[i] = id;
    while (head < tail) {
      const p = queue[head++]; area++;
      const x = p % w, y = (p - x) / w;
      if (x > 0) { const q = p - 1; if (labels[q] === -1 && alpha[q] >= 0.5) { labels[q] = id; queue[tail++] = q; } }
      if (x < w - 1) { const q = p + 1; if (labels[q] === -1 && alpha[q] >= 0.5) { labels[q] = id; queue[tail++] = q; } }
      if (y > 0) { const q = p - w; if (labels[q] === -1 && alpha[q] >= 0.5) { labels[q] = id; queue[tail++] = q; } }
      if (y < h - 1) { const q = p + w; if (labels[q] === -1 && alpha[q] >= 0.5) { labels[q] = id; queue[tail++] = q; } }
    }
    areas.push(area);
  }
  if (!areas.length) return { alpha, empty: true };
  const largest = Math.max(...areas);
  const minArea = Math.max(largest * 0.02, n * 0.0005);
  const keep = new Float32Array(n);
  for (let i = 0; i < n; i++) if (labels[i] >= 0 && areas[labels[i]] >= minArea) keep[i] = 1;
  // weiche Zone um behaltene Teile (für halbtransparente Kanten/Fransen)
  const r = Math.max(2, Math.round(Math.max(w, h) * 0.006));
  const zone = boxBlur(keep, w, h, r);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const z = zone[i] > 0.001 ? 1 : 0;
    let a = alpha[i] * z;
    if (a < 0.02) a = 0; // Restschleier
    out[i] = a;
  }
  return { alpha: out, empty: false };
}

/** Bounding Box (inklusive) der Pixel mit alpha > thr, oder null. */
export function bboxOf(alpha, w, h, thr = 0.1) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) {
    const o = y * w;
    for (let x = 0; x < w; x++) {
      if (alpha[o + x] > thr) {
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
    }
  }
  return x1 < 0 ? null : { x0, y0, x1, y1 };
}

/**
 * Produkt-Rechteck (Arbeits-Pixel) plus 3 % Polster für die Verfeinerung.
 * scale: Wertebereich des Alphas (1 für Float, 255 für Uint8).
 */
export function paddedRect(alpha, w, h, scale = 1) {
  const bb = bboxOf(alpha, w, h, 0.1 * scale);
  if (!bb) return null;
  const pad = Math.round(Math.max(w, h) * 0.03);
  const x0 = Math.max(0, bb.x0 - pad), y0 = Math.max(0, bb.y0 - pad);
  const x1 = Math.min(w - 1, bb.x1 + pad), y1 = Math.min(h - 1, bb.y1 + pad);
  return {
    x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1, // gepolstert (Arbeits-Pixel)
    bx: bb.x0, by: bb.y0, bw: bb.x1 - bb.x0 + 1, bh: bb.y1 - bb.y0 + 1, // Produkt
    ww: w, wh: h,
  };
}

/** Ausschnitt (Rechteck in Pixeln) aus einem Kanal. */
export function cropPlane(src, w, x0, y0, cw, ch) {
  const out = new Float32Array(cw * ch);
  for (let y = 0; y < ch; y++) out.set(src.subarray((y0 + y) * w + x0, (y0 + y) * w + x0 + cw), y * cw);
  return out;
}

/* ------------------------------------------------------ Kantenverfeinerung */

/**
 * Schneller Farb-Guided-Filter (He et al.), Koeffizienten auf 1/s Auflösung.
 * I: [r,g,b] planar 0..1, p: Eingabe-Alpha.
 */
export function fastGuidedFilter(I, p, w, h, r, eps, s = 2) {
  const sw = Math.max(1, Math.round(w / s)), sh = Math.max(1, Math.round(h / s));
  const rs = Math.max(1, Math.round(r / s));
  const Ir = resample(I[0], w, h, sw, sh), Ig = resample(I[1], w, h, sw, sh), Ib = resample(I[2], w, h, sw, sh);
  const P = resample(p, w, h, sw, sh);
  const n = sw * sh;
  const B = (x) => boxBlur(x, sw, sh, rs);
  const mul = (a, b) => { const o = new Float32Array(n); for (let i = 0; i < n; i++) o[i] = a[i] * b[i]; return o; };
  const mR = B(Ir), mG = B(Ig), mB = B(Ib), mP = B(P);
  const mRP = B(mul(Ir, P)), mGP = B(mul(Ig, P)), mBP = B(mul(Ib, P));
  const vRR = B(mul(Ir, Ir)), vRG = B(mul(Ir, Ig)), vRB = B(mul(Ir, Ib));
  const vGG = B(mul(Ig, Ig)), vGB = B(mul(Ig, Ib)), vBB = B(mul(Ib, Ib));
  const aR = new Float32Array(n), aG = new Float32Array(n), aB = new Float32Array(n), bb = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const mr = mR[i], mg = mG[i], mb = mB[i], mp = mP[i];
    const cr = mRP[i] - mr * mp, cg = mGP[i] - mg * mp, cb = mBP[i] - mb * mp;
    const rr = vRR[i] - mr * mr + eps, rg = vRG[i] - mr * mg, rb = vRB[i] - mr * mb;
    const gg = vGG[i] - mg * mg + eps, gb = vGB[i] - mg * mb, b2 = vBB[i] - mb * mb + eps;
    // Inverse der symmetrischen 3x3-Matrix
    const i00 = gg * b2 - gb * gb, i01 = gb * rb - rg * b2, i02 = rg * gb - gg * rb;
    const i11 = rr * b2 - rb * rb, i12 = rb * rg - rr * gb, i22 = rr * gg - rg * rg;
    const det = rr * i00 + rg * i01 + rb * i02;
    const inv = det !== 0 ? 1 / det : 0;
    const ar = (i00 * cr + i01 * cg + i02 * cb) * inv;
    const ag = (i01 * cr + i11 * cg + i12 * cb) * inv;
    const ab = (i02 * cr + i12 * cg + i22 * cb) * inv;
    aR[i] = ar; aG[i] = ag; aB[i] = ab;
    bb[i] = mp - ar * mr - ag * mg - ab * mb;
  }
  const maR = resample(B(aR), sw, sh, w, h), maG = resample(B(aG), sw, sh, w, h);
  const maB = resample(B(aB), sw, sh, w, h), mbb = resample(B(bb), sw, sh, w, h);
  const q = new Float32Array(w * h);
  for (let i = 0; i < q.length; i++) q[i] = maR[i] * I[0][i] + maG[i] * I[1][i] + maB[i] * I[2][i] + mbb[i];
  return q;
}

/**
 * Vordergrundfarben schätzen (Forte & Pitié, "Approximate Fast Foreground
 * Colour Estimation", 2021). Entfernt die Hintergrundfarbe aus halbtransparenten
 * Randpixeln -> kein heller/dunkler Saum auf Weiß.
 * Die zwei Blur-Stufen laufen auf halber Auflösung, die Korrektur in voller.
 */
export function estimateForeground(I, a, w, h) {
  const s = 2;
  const sw = Math.max(1, Math.round(w / s)), sh = Math.max(1, Math.round(h / s));
  const n = sw * sh;
  const A = resample(a, w, h, sw, sh);
  const Is = I.map((c) => resample(c, w, h, sw, sh));
  const k1 = Math.max(3, Math.round(Math.max(sw, sh) * 0.06));
  const k2 = Math.max(2, Math.round(Math.max(sw, sh) * 0.004));
  let F = Is.map((c) => Float32Array.from(c));
  let Bk = Is.map((c) => Float32Array.from(c));
  let blurF, blurB;
  for (const k of [k1, k2]) {
    const r = Math.max(1, Math.round(k / 2));
    const bA = boxBlur(A, sw, sh, r);
    blurF = []; blurB = [];
    const nextF = [];
    for (let c = 0; c < 3; c++) {
      const fa = new Float32Array(n), b1a = new Float32Array(n);
      for (let i = 0; i < n; i++) { fa[i] = F[c][i] * A[i]; b1a[i] = Bk[c][i] * (1 - A[i]); }
      const bFA = boxBlur(fa, sw, sh, r), bB1A = boxBlur(b1a, sw, sh, r);
      const bf = new Float32Array(n), bbk = new Float32Array(n), nf = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        bf[i] = bFA[i] / (bA[i] + 1e-5);
        bbk[i] = bB1A[i] / (1 - bA[i] + 1e-5);
        const al = A[i];
        let f = bf[i] + al * (Is[c][i] - al * bf[i] - (1 - al) * bbk[i]);
        nf[i] = f < 0 ? 0 : f > 1 ? 1 : f;
      }
      blurF.push(bf); blurB.push(bbk); nextF.push(nf);
    }
    F = nextF; Bk = blurB;
  }
  // Korrektur in voller Auflösung mit den geglätteten Schätzungen
  const out = [];
  for (let c = 0; c < 3; c++) {
    const bf = resample(blurF[c], sw, sh, w, h), bbk = resample(blurB[c], sw, sh, w, h);
    const o = new Float32Array(w * h);
    for (let i = 0; i < o.length; i++) {
      const al = a[i];
      if (al >= 0.995) { o[i] = I[c][i]; continue; }
      let f = bf[i] + al * (I[c][i] - al * bf[i] - (1 - al) * bbk[i]);
      o[i] = f < 0 ? 0 : f > 1 ? 1 : f;
    }
    out.push(o);
  }
  return out;
}

/**
 * Alpha für den hochaufgelösten Ausschnitt verfeinern und Farben dekontaminieren.
 * @param rgba  Uint8ClampedArray des Ausschnitts (Originalpixel)
 * @param alphaUp grob hochskaliertes Alpha (Float32, gleiche Größe)
 * @returns {rgba: Uint8ClampedArray (F + Alpha), bbox, gains}
 */
export const REFINE = { eps: 1e-3, maxGain: 0.1, maxLoss: 0.5, mix: 0.5, choke: 0.6 };

export function refineHi(rgba, w, h, alphaUp, opts = REFINE) {
  const n = w * h;
  const I = toPlanes(rgba, n);
  const size = Math.max(w, h);
  // Guided Filter nur im Unsicherheitsband um die Kante anwenden
  const r = Math.max(2, Math.round(size * 0.004));
  const q = opts.mix > 0 ? fastGuidedFilter(I, alphaUp, w, h, r, opts.eps, size > 900 ? 2 : 1) : alphaUp;
  const hard = new Float32Array(n);
  for (let i = 0; i < n; i++) hard[i] = alphaUp[i] > 0.5 ? 1 : 0;
  const band = boxBlur(hard, w, h, Math.max(2, Math.round(r * 1.5)));
  const a = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = alphaUp[i];
    let v = p;
    if (band[i] > 0.0005 && band[i] < 0.9995) {
      // Der Filter darf Kanten an echte Bildkanten anlegen und Hintergrund
      // wegnehmen, aber kaum neue Fläche hinzufügen (sonst wandert z.B. Kies
      // oder Teppichstruktur in den Rand).
      let d = (q[i] - p) * opts.mix;
      if (d > opts.maxGain) d = opts.maxGain;
      if (d < -opts.maxLoss) d = -opts.maxLoss;
      v = p + d;
    }
    // leichte Kontrastkurve: entfernt Restschleier, Kanten bleiben weich
    v = (v - 0.04) / 0.92;
    a[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  chokeInPlace(a, w, h, opts.choke ?? 0.6);
  const F = estimateForeground(I, a, w, h);
  const out = new Uint8ClampedArray(n * 4);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    out[j] = F[0][i] * 255 + 0.5; out[j + 1] = F[1][i] * 255 + 0.5; out[j + 2] = F[2][i] * 255 + 0.5;
    out[j + 3] = a[i] * 255 + 0.5;
  }
  return { rgba: out, bbox: bboxOf(a, w, h, 0.06), gains: autoGains(F, a, n) };
}

/**
 * Zieht die Kante um ca. `amount` Pixel nach innen (Mischung mit 3x3-Minimum).
 * Der äußerste, oft noch hintergrundfarbene Pixelring wird dadurch
 * halbtransparent und von der Farbdekontamination erfasst.
 */
function chokeInPlace(a, w, h, amount) {
  if (amount <= 0) return;
  const src = Float32Array.from(a);
  for (let y = 0; y < h; y++) {
    const y0 = y > 0 ? y - 1 : y, y1 = y < h - 1 ? y + 1 : y;
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const v = src[i];
      if (v === 0) continue;
      const x0 = x > 0 ? x - 1 : x, x1 = x < w - 1 ? x + 1 : x;
      let m = v;
      for (let yy = y0; yy <= y1; yy++) {
        const o = yy * w;
        for (let xx = x0; xx <= x1; xx++) if (src[o + xx] < m) m = src[o + xx];
      }
      a[i] = v + (m - v) * amount;
    }
  }
}

/** Vorschlag für leichte Helligkeits-/Weißabgleich-Korrektur (nur Produktpixel). */
function autoGains(F, a, n) {
  const lum = [];
  let wr = 0, wg = 0, wb = 0, wc = 0;
  for (let i = 0; i < n; i += 2) {
    if (a[i] < 0.95) continue;
    const r = F[0][i], g = F[1][i], b = F[2][i];
    lum.push(0.2126 * r + 0.7152 * g + 0.0722 * b);
  }
  if (lum.length < 100) return [1, 1, 1];
  lum.sort((x, y) => x - y);
  const l90 = lum[Math.floor(lum.length * 0.9)], l99 = lum[Math.floor(lum.length * 0.99)];
  for (let i = 0; i < n; i += 2) {
    if (a[i] < 0.95) continue;
    const r = F[0][i], g = F[1][i], b = F[2][i];
    const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    if (l >= l90 && Math.max(r, g, b) - Math.min(r, g, b) < 0.12) { wr += r; wg += g; wb += b; wc++; }
  }
  let gr = 1, gg = 1, gb = 1;
  if (wc > lum.length * 0.01) {
    const m = (wr + wg + wb) / 3;
    const clamp = (v) => Math.min(1.08, Math.max(0.92, v));
    gr = clamp(m / (wr || 1)); gg = clamp(m / (wg || 1)); gb = clamp(m / (wb || 1));
  }
  const expo = l99 < 0.93 ? Math.min(1.2, 0.95 / Math.max(l99, 0.01)) : 1;
  return [gr * expo, gg * expo, gb * expo];
}

/* -------------------------------------------------------------- Komposition */

export const DEFAULTS = { format: 'portrait', margin: 0.08, shadow: 0.35, autoColor: false };

export function outputSize(format) {
  return format === 'square' ? [1200, 1200] : [1200, 1600];
}

/**
 * Setzt das freigestellte Produkt auf Weiß mit zweistufigem Schatten.
 * @param hi {w,h,rgba,bbox,gains}
 * @returns {w,h,data: Uint8ClampedArray RGBA}
 */
export function compose(hi, settings) {
  const st = { ...DEFAULTS, ...settings };
  const [W, H] = outputSize(st.format);
  const out = new Uint8ClampedArray(W * H * 4).fill(255);
  if (!hi.bbox) return { w: W, h: H, data: out };
  const { x0, y0, x1, y1 } = hi.bbox;
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  const m = Math.min(0.3, Math.max(0, st.margin));
  const s = Math.min((W * (1 - 2 * m)) / bw, (H * (1 - 2 * m)) / bh);
  const tw = Math.max(1, Math.round(bw * s)), th = Math.max(1, Math.round(bh * s));
  const ox = Math.round((W - tw) / 2), oy = Math.round((H - th) / 2);

  // vormultiplizierte Kanäle des Ausschnitts
  const n = bw * bh;
  const pr = new Float32Array(n), pg = new Float32Array(n), pb = new Float32Array(n), pa = new Float32Array(n);
  const gains = st.autoColor ? hi.gains : [1, 1, 1];
  for (let y = 0; y < bh; y++) {
    for (let x = 0; x < bw; x++) {
      const j = ((y0 + y) * hi.w + x0 + x) * 4, i = y * bw + x;
      const a = hi.rgba[j + 3] / 255;
      pr[i] = Math.min(1, (hi.rgba[j] / 255) * gains[0]) * a;
      pg[i] = Math.min(1, (hi.rgba[j + 1] / 255) * gains[1]) * a;
      pb[i] = Math.min(1, (hi.rgba[j + 2] / 255) * gains[2]) * a;
      pa[i] = a;
    }
  }
  const R = resample(pr, bw, bh, tw, th), G = resample(pg, bw, bh, tw, th);
  const Bc = resample(pb, bw, bh, tw, th), A = resample(pa, bw, bh, tw, th);

  // Schatten auf halber Auflösung berechnen
  const k = Math.min(1, Math.max(0, st.shadow));
  let shadow = null;
  if (k > 0.001) shadow = buildShadow(pa, bw, bh, W, H, tw, th, ox, oy, k);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const o = (y * W + x) * 4;
      let bg = 1;
      if (shadow) bg = 1 - shadow[y * W + x];
      const lx = x - ox, ly = y - oy;
      let r = bg, g = bg, b = bg;
      if (lx >= 0 && ly >= 0 && lx < tw && ly < th) {
        const i = ly * tw + lx;
        const a = Math.min(1, Math.max(0, A[i]));
        r = R[i] + bg * (1 - a); g = G[i] + bg * (1 - a); b = Bc[i] + bg * (1 - a);
      }
      out[o] = r * 255 + 0.5; out[o + 1] = g * 255 + 0.5; out[o + 2] = b * 255 + 0.5;
    }
  }
  return { w: W, h: H, data: out };
}

/**
 * Zwei Schattenebenen aus der Silhouette:
 *  - Kontaktschatten: eng, leicht nach unten versetzt
 *  - weicher Schatten: groß, weich, weiter nach unten versetzt
 * Gibt die Abdunklung (0..1) pro Ausgabepixel zurück.
 */
export function buildShadow(alphaCrop, bw, bh, W, H, tw, th, ox, oy, k) {
  const f = 4; // Rechenauflösung 1/4
  const sW = Math.ceil(W / f), sH = Math.ceil(H / f);
  const stw = Math.max(1, Math.round(tw / f)), sth = Math.max(1, Math.round(th / f));
  const small = resample(alphaCrop, bw, bh, stw, sth);
  const plane = new Float32Array(sW * sH);
  const sox = Math.round(ox / f), soy = Math.round(oy / f);
  for (let y = 0; y < sth; y++) {
    const ty = y + soy; if (ty < 0 || ty >= sH) continue;
    for (let x = 0; x < stw; x++) {
      const tx = x + sox; if (tx < 0 || tx >= sW) continue;
      plane[ty * sW + tx] = small[y * stw + x];
    }
  }
  const S = Math.min(W, H) / f;
  const contact = gaussBlur(plane, sW, sH, S * 0.006);
  const soft = gaussBlur(plane, sW, sH, S * 0.035);
  const dyC = S * 0.005, dyS = S * 0.018;
  const oc = 0.5 * k, os = 0.3 * k;
  const d = new Float32Array(sW * sH);
  for (let y = 0; y < sH; y++) {
    const yc = Math.min(sH - 1, Math.max(0, Math.round(y - dyC)));
    const ys = Math.min(sH - 1, Math.max(0, Math.round(y - dyS)));
    for (let x = 0; x < sW; x++) {
      const c = contact[yc * sW + x] * oc, s = soft[ys * sW + x] * os;
      d[y * sW + x] = 1 - (1 - c) * (1 - s);
    }
  }
  return resample(d, sW, sH, W, H);
}
