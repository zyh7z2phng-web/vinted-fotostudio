import { Editor } from './editor.js';

const WORK_MAX = 1500;           // interne Arbeitsauflösung (lange Seite)
const SOURCE_MAX = 3000;         // Quellbild für die Verfeinerung (lange Seite)
const PRODUCT_MAX = [1200, 1600]; // größtmögliche Produktfläche im Export
const $ = (s) => document.querySelector(s);
const isTouch = matchMedia('(pointer: coarse)').matches;

/* ------------------------------------------------------------ Einstellungen */

const settings = { format: 'portrait', margin: 0.08, shadow: 0.35, autoColor: false };
try { Object.assign(settings, JSON.parse(localStorage.getItem('fs.settings') || '{}')); } catch {}
function saveSettings() { try { localStorage.setItem('fs.settings', JSON.stringify(settings)); } catch {} }

function setFill(input) {
  const min = +input.min || 0, max = +input.max || 100;
  input.style.setProperty('--fill', ((input.value - min) / (max - min)) * 100 + '%');
}

function syncSettingsUI() {
  document.querySelectorAll('[data-format]').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.format === settings.format)));
  $('#margin').value = Math.round(settings.margin * 100);
  $('#marginOut').textContent = Math.round(settings.margin * 100) + ' %';
  $('#shadow').value = Math.round(settings.shadow * 100);
  $('#shadowOut').textContent = Math.round(settings.shadow * 100) + ' %';
  setFill($('#margin')); setFill($('#shadow'));
  $('#autoColor').checked = !!settings.autoColor;
  document.body.classList.toggle('square', settings.format === 'square');
}

let recomposeTimer = 0;
function settingsChanged(immediate) {
  syncSettingsUI();
  saveSettings();
  clearTimeout(recomposeTimer);
  recomposeTimer = setTimeout(recomposeAll, immediate ? 0 : 250);
}

document.querySelectorAll('[data-format]').forEach((b) => b.addEventListener('click', () => { settings.format = b.dataset.format; settingsChanged(true); }));
$('#margin').addEventListener('input', (e) => { settings.margin = e.target.value / 100; settingsChanged(); });
$('#shadow').addEventListener('input', (e) => { settings.shadow = e.target.value / 100; settingsChanged(); });
$('#autoColor').addEventListener('change', (e) => { settings.autoColor = e.target.checked; settingsChanged(true); });
syncSettingsUI();

/* Sheet „Anpassen“: von unten, Fotos bleiben darüber sichtbar */
function openSheet() {
  $('#sheet').hidden = false; $('#sheetBackdrop').hidden = false;
  $('#sheet').style.transform = '';
  $('#sheetDone').focus({ preventScroll: true });
}
function closeSheet() {
  $('#sheet').hidden = true; $('#sheetBackdrop').hidden = true;
  $('#openSettings').focus({ preventScroll: true });
}
$('#openSettings').addEventListener('click', openSheet);
$('#sheetDone').addEventListener('click', closeSheet);
$('#sheetBackdrop').addEventListener('click', closeSheet);
// Nach unten ziehen schließt das Sheet
(() => {
  const head = $('.sheet-head'), sheet = $('#sheet');
  let startY = null, dy = 0;
  head.addEventListener('pointerdown', (e) => { if (e.target.closest('button')) return; startY = e.clientY; dy = 0; head.setPointerCapture?.(e.pointerId); });
  head.addEventListener('pointermove', (e) => { if (startY === null) return; dy = Math.max(0, e.clientY - startY); sheet.style.transform = `translateY(${dy}px)`; });
  const end = () => { if (startY === null) return; startY = null; if (dy > 80) closeSheet(); sheet.style.transform = ''; };
  head.addEventListener('pointerup', end); head.addEventListener('pointercancel', end);
})();

/* ------------------------------------------------------------------ Worker */

/** Startet einen Modul-Worker mit einfachem Anfrage/Antwort-Protokoll. */
function spawn(url, onEvent) {
  const w = new Worker(new URL(url, import.meta.url), { type: 'module' });
  const pending = new Map();
  let rid = 0, dead = false;
  const failAll = (msg) => {
    dead = true;
    for (const p of pending.values()) p.reject(new Error(msg));
    pending.clear();
  };
  w.onmessage = (e) => {
    const m = e.data;
    if (m.type) { if (onEvent) onEvent(m); return; }
    const p = pending.get(m.rid);
    if (!p) return;
    pending.delete(m.rid);
    m.ok ? p.resolve(m.result) : p.reject(new Error(m.error));
  };
  w.onerror = (e) => { console.error(e); failAll(e.message || 'Hintergrundprozess abgestürzt'); };
  return {
    get dead() { return dead; },
    call(cmd, args, transfer) {
      if (dead) return Promise.reject(new Error('Hintergrundprozess beendet'));
      return new Promise((resolve, reject) => {
        const id = ++rid;
        pending.set(id, { resolve, reject });
        w.postMessage({ rid: id, cmd, args }, transfer || []);
      });
    },
    terminate() { w.terminate(); failAll('Hintergrundprozess beendet'); },
  };
}

// Bildverarbeitung + Ablage der Masken (lebt so lange wie die Seite)
const proc = spawn('./worker.js');
const call = (cmd, args, transfer) => proc.call(cmd, args, transfer).catch((e) => {
  if (proc.dead) setStatus('Fehler: ' + e.message, true);
  throw e;
});

/* KI-Freistellung in einem eigenen Worker. Der WebAssembly-Speicher der ONNX
   Runtime wächst bei der ersten Berechnung um ~700 MB und wird nie wieder
   freigegeben. Zusammen mit der übrigen Verarbeitung hat das auf dem Handy
   (v.a. iPhone/Safari) den Tab nach dem ersten Foto abstürzen lassen.
   Sparmodus für Handys: Der KI-Worker wird nach jedem Foto beendet und für das
   nächste neu gestartet (das Modell kommt dann aus dem Cache), und die Runtime
   läuft mit einem Thread. Mit mehreren Threads liegt der Speicher als
   SharedArrayBuffer vor und wird nach dem Beenden erst verzögert frei, alter und
   neuer Worker würden sich dann überlappen.
   Testschalter: ?lowmem=1 bzw. ?lowmem=0 */
const params = new URLSearchParams(location.search);
const LOW_MEM = params.has('lowmem') ? params.get('lowmem') === '1'
  : isTouch || /iPhone|iPad|iPod|Android/i.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
const THREADS = LOW_MEM ? 1 : 4;
let forceBackend = params.get('backend');
let infer = null; // { w, ready }

function onInferEvent(m) {
  if (m.type === 'progress') { if (!engine) onModelProgress(m); return; }
  if (m.type === 'backend') noteBackend(m);
}
function noteBackend(info) {
  // WebGPU war unbrauchbar -> beim Neustart gleich WASM nehmen (lädt sonst beide Runtimes)
  if (info.backend === 'wasm') forceBackend = 'wasm';
  if (!engine || engine.backend !== info.backend || engine.threads !== info.threads) showReady(info);
}
function startInfer() {
  const w = spawn('./infer.js', onInferEvent);
  const ready = w.call('init', { force: forceBackend, threads: THREADS }).then((info) => { noteBackend(info); return info; });
  const inst = { w, ready, failed: false };
  ready.catch(() => { inst.failed = true; w.terminate(); }); // nächster Versuch startet neu
  return inst;
}
async function segment(args, transfer) {
  if (!infer || infer.w.dead || infer.failed) infer = startInfer();
  const cur = infer;
  try {
    await cur.ready;
    return await cur.w.call('segment', { ...args, force: forceBackend, threads: THREADS }, transfer);
  } finally {
    if (LOW_MEM && infer === cur) { cur.w.terminate(); infer = null; }
    else if (cur.w.dead && infer === cur) infer = null;
  }
}

function setStatus(t, err) { const s = $('#status'); s.textContent = t; s.classList.toggle('err', !!err); }
function mb(b) { return (b / 1048576).toFixed(0); }
let engine = null;
function onModelProgress({ loaded, total }) {
  setStatus(`KI-Modell wird einmalig geladen, ${mb(loaded)} von ${mb(total)} MB`);
}
function showReady(info) {
  engine = info;
  const how = info.backend === 'webgpu' ? 'auf dem Grafikchip' : `im Prozessor${info.threads > 1 ? ' mit ' + info.threads + ' Threads' : ''}`;
  $('#engineNote').textContent = `Fotos bleiben auf deinem Gerät. Die KI läuft ${how}.`;
  updateStatus();
}
function updateStatus() {
  if (!engine) return;
  const n = photos.filter((p) => p.blob).length;
  setStatus(n ? `${n} ${n === 1 ? 'Foto' : 'Fotos'} fertig` : 'Bereit. Fotos bleiben auf deinem Gerät.');
}

setStatus('KI wird vorbereitet');
infer = startInfer();
const ready = infer.ready.catch((e) => { setStatus('KI konnte nicht starten: ' + e.message, true); throw e; });

/* --------------------------------------------------------- Bilder laden */

function isHeic(file) { return /heic|heif/i.test(file.type) || /\.(heic|heif)$/i.test(file.name); }

let heicLib = null;
function loadScript(src) {
  return new Promise((res, rej) => { const s = document.createElement('script'); s.src = src; s.onload = res; s.onerror = () => rej(new Error('Konnte ' + src + ' nicht laden')); document.head.append(s); });
}

async function decodeBlob(blob) {
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.decoding = 'async';
  img.src = url;
  try { await img.decode(); } catch (e) { URL.revokeObjectURL(url); throw e; }
  // <img> wendet die EXIF-Drehung an; naturalWidth/Height sind bereits gedreht
  return { img, release: () => { img.src = ''; URL.revokeObjectURL(url); } };
}

async function loadImage(file) {
  try { return await decodeBlob(file); }
  catch (e) {
    if (!isHeic(file)) throw new Error('Bildformat wird nicht unterstützt');
    // Safari kann HEIC selbst; andere Browser brauchen den Decoder
    if (!heicLib) heicLib = loadScript('vendor/heic2any.min.js');
    await heicLib;
    const jpg = await window.heic2any({ blob: file, toType: 'image/jpeg', quality: 0.95 });
    return decodeBlob(Array.isArray(jpg) ? jpg[0] : jpg);
  }
}

function makeCanvas(w, h) { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; }
function freeCanvas(c) { c.width = c.height = 1; }

/** Ausschnitt hochwertig verkleinern (stufenweise halbieren), gibt Canvas zurück. */
function drawScaled(src, sx, sy, sw, sh, dw, dh) {
  let cur = src, cx = sx, cy = sy, cw = sw, ch = sh;
  const temps = [];
  while (cw / 2 >= dw && ch / 2 >= dh) {
    const nw = Math.round(cw / 2), nh = Math.round(ch / 2);
    const c = makeCanvas(nw, nh), x = c.getContext('2d');
    x.imageSmoothingQuality = 'high';
    x.drawImage(cur, cx, cy, cw, ch, 0, 0, nw, nh);
    temps.push(c); cur = c; cx = cy = 0; cw = nw; ch = nh;
  }
  const out = makeCanvas(dw, dh), ctx = out.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(cur, cx, cy, cw, ch, 0, 0, dw, dh);
  temps.forEach(freeCanvas);
  return out;
}

function canvasToBlob(c, type, q) { return new Promise((res) => c.toBlob(res, type, q)); }

/**
 * Foto laden und sofort auf ein Quellbild (≤ SOURCE_MAX) verkleinern.
 * Das voll aufgelöste Original (iPhone: 12–48 MP, bis knapp 200 MB dekodiert)
 * wird direkt danach freigegeben – sonst bleibt es während der KI-Berechnung im
 * Speicher und Safari beendet den Tab.
 * Setzt p.ow/p.oh (Quellbild), p.sc und p.ww/p.wh (Arbeitsbild).
 */
async function loadSource(p) {
  const { img, release } = await loadImage(p.file);
  let src;
  try {
    const iw = img.naturalWidth, ih = img.naturalHeight;
    const ss = Math.min(1, SOURCE_MAX / Math.max(iw, ih));
    src = drawScaled(img, 0, 0, iw, ih, Math.max(1, Math.round(iw * ss)), Math.max(1, Math.round(ih * ss)));
  } finally { release(); }
  const ow = src.width, oh = src.height;
  const sc = Math.min(1, WORK_MAX / Math.max(ow, oh));
  p.ow = ow; p.oh = oh; p.sc = sc;
  p.ww = Math.max(1, Math.round(ow * sc)); p.wh = Math.max(1, Math.round(oh * sc));
  return src;
}

/* ------------------------------------------------------------------ Fotos */

const photos = [];
let nextId = 1;
let chain = Promise.resolve();
let batch = { total: 0, done: 0 };

$('#fileInput').addEventListener('change', (e) => {
  const files = [...e.target.files];
  e.target.value = '';
  addFiles(files);
});

function addFiles(files) {
  if (!files.length) return;
  for (const file of files) {
    const p = { id: nextId++, file, ver: 0, state: 'queued' };
    p.el = createCard(p);
    photos.push(p);
    $('#grid').append(p.el);
    batch.total++;
    chain = chain.then(() => processPhoto(p)).catch(() => {});
  }
  updateChrome();
}

function updateProgress() {
  const box = $('#progress');
  if (batch.done >= batch.total) { box.hidden = true; batch = { total: 0, done: 0 }; updateStatus(); return; }
  box.hidden = false;
  setStatus('Fotos werden freigestellt');
  $('#progressBar').style.width = (100 * batch.done / batch.total) + '%';
  $('#progressText').textContent = `${batch.done} von ${batch.total} fertig`;
}

function updateChrome() {
  $('#empty').hidden = photos.length > 0;
  $('#shareAll').disabled = !photos.some((p) => p.blob);
  updateStatus();
  if (!$('#progress').hidden) setStatus('Fotos werden freigestellt');
}

async function processPhoto(p) {
  if (p.deleted) { p.state = 'skipped'; batch.done++; updateProgress(); return; }
  try {
    setCard(p, 'busy', 'Wird geladen');
    updateProgress();
    let src = await loadSource(p), wc;
    try {
      wc = drawScaled(src, 0, 0, p.ow, p.oh, p.ww, p.wh);
      const thumbSc = Math.min(1, 900 / Math.max(p.ww, p.wh));
      const tc = drawScaled(wc, 0, 0, p.ww, p.wh, Math.round(p.ww * thumbSc), Math.round(p.wh * thumbSc));
      p.origUrl = URL.createObjectURL(await canvasToBlob(tc, 'image/jpeg', 0.85));
      freeCanvas(tc);
      p.el.querySelector('.original').src = p.origUrl;
      const data = wc.getContext('2d').getImageData(0, 0, p.ww, p.wh).data;
      freeCanvas(wc); wc = null;

      await ready.catch(() => {}); // schlägt der Start fehl, versucht segment() es erneut
      setCard(p, 'busy', 'Wird freigestellt');
      const seg = await segment({ rgba: data.buffer, w: p.ww, h: p.wh }, [data.buffer]);
      if (!seg.rect) throw new Error('Kein Produkt erkannt');
      await call('put', { id: p.id, alpha: seg.alpha, w: p.ww, h: p.wh }, [seg.alpha]);
      p.rect = seg.rect;
      setCard(p, 'busy', 'Wird verfeinert');
      await refineFromImage(p, src);
    } finally {
      if (wc) freeCanvas(wc);
      freeCanvas(src);
    }
    if (p.deleted) { p.state = 'skipped'; return; }
    await composePhoto(p);
  } catch (e) {
    console.error(e);
    setCard(p, 'error', errorText(e) + ' Tippen zum Wiederholen.');
  } finally {
    batch.done++;
    updateProgress();
    updateChrome();
  }
}

/** Produkt-Ausschnitt in hoher Auflösung aus dem Original holen und verfeinern lassen. */
async function refineFromImage(p, img) {
  const r = p.rect, sc = p.sc;
  const ox = Math.max(0, Math.floor(r.x / sc)), oy = Math.max(0, Math.floor(r.y / sc));
  const ow = Math.min(p.ow - ox, Math.ceil(r.w / sc)), oh = Math.min(p.oh - oy, Math.ceil(r.h / sc));
  const hs = Math.min(1, PRODUCT_MAX[0] / (r.bw / sc), PRODUCT_MAX[1] / (r.bh / sc));
  const hw = Math.max(1, Math.round(ow * hs)), hh = Math.max(1, Math.round(oh * hs));
  const c = drawScaled(img, ox, oy, ow, oh, hw, hh);
  const data = c.getContext('2d').getImageData(0, 0, hw, hh).data;
  freeCanvas(c);
  const res = await call('refine', { id: p.id, rgba: data.buffer, w: hw, h: hh, rect: { x: r.x, y: r.y, w: r.w, h: r.h } }, [data.buffer]);
  if (res.empty) throw new Error('Kein Produkt erkannt');
}

async function composePhoto(p) {
  if (p.deleted || !p.rect) return;
  const v = ++p.ver;
  if (p.blob) setCard(p, 'busy');
  const r = await call('compose', { id: p.id, settings: { ...settings }, quality: 0.92 });
  if (v !== p.ver || p.deleted) return;
  let blob = r.blob;
  if (!blob) { // Fallback ohne OffscreenCanvas
    const c = makeCanvas(r.w, r.h);
    c.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(r.raw), r.w, r.h), 0, 0);
    blob = await canvasToBlob(c, 'image/jpeg', 0.92);
    freeCanvas(c);
  }
  p.blob = blob;
  const old = p.url;
  p.url = URL.createObjectURL(blob);
  const img = p.el.querySelector('.result');
  img.src = p.url;
  if (old) setTimeout(() => URL.revokeObjectURL(old), 1000);
  setCard(p, 'ready');
  if (viewerPhoto === p) renderViewer();
  updateChrome();
}

async function recomposeAll() {
  for (const p of photos) {
    if (!p.blob || p.deleted) continue;
    try { await composePhoto(p); } catch (e) { console.error(e); }
  }
}

/* ------------------------------------------------------------------ Karten */

function errorText(e) {
  const m = String((e && e.message) || e);
  if (/Kein Produkt/.test(m)) return 'Kein Teil erkannt.';
  if (/Bildformat/.test(m)) return 'Bildformat wird nicht unterstützt.';
  if (/Alles wurde entfernt/.test(m)) return 'Alles wurde wegradiert.';
  return 'Das hat nicht geklappt.';
}

function createCard(p) {
  const el = $('#cardTpl').content.firstElementChild.cloneNode(true);
  el.querySelector('.result').alt = '';
  el.querySelector('.original').alt = '';
  el.querySelector('.tile-btn').addEventListener('click', () => {
    if (p.state === 'error') retry(p);
    else if (p.blob) openViewer(p);
  });
  p.el = el;
  setCard(p, 'queued', 'Wartet');
  return el;
}

function setCard(p, state, msg) {
  p.state = state;
  const el = p.el;
  if (!el) return;
  const cls = state === 'busy' && p.blob ? 'updating' : state;
  el.className = 'tile ' + cls;
  if (msg !== undefined) el.querySelector('.msg').textContent = msg;
  const label = { ready: 'Ergebnis ansehen', updating: 'Wird aktualisiert', error: el.querySelector('.msg').textContent, queued: 'Wartet', busy: el.querySelector('.msg').textContent }[cls] || '';
  el.querySelector('.tile-btn').setAttribute('aria-label', `${p.file.name}: ${label}`);
  if (viewerPhoto === p) renderViewer();
}

function retry(p) {
  batch.total++;
  setCard(p, 'queued', 'Wartet');
  updateProgress();
  chain = chain.then(() => processPhoto(p)).catch(() => {});
}

/* Löschen ohne Bestätigungsdialog (manche Browser blockieren confirm()),
   stattdessen mit „Rückgängig“ für einige Sekunden. */
let pendingDelete = null;

function removePhoto(p) {
  finalizeDelete();
  const index = photos.indexOf(p);
  if (index < 0) return;
  const next = p.el.nextElementSibling;
  p.deleted = true;
  photos.splice(index, 1);
  p.el.remove();
  updateChrome();
  const entry = { p, index, next };
  entry.timer = setTimeout(finalizeDelete, 6000);
  pendingDelete = entry;
  showToast('Foto gelöscht', 'Rückgängig', () => undoDelete(entry), 6000);
}

function undoDelete(entry) {
  if (pendingDelete !== entry) return;
  clearTimeout(entry.timer);
  pendingDelete = null;
  const { p } = entry;
  p.deleted = false;
  photos.splice(Math.min(entry.index, photos.length), 0, p);
  const grid = $('#grid');
  if (entry.next && entry.next.parentNode === grid) grid.insertBefore(p.el, entry.next);
  else grid.append(p.el);
  if (p.blob) composePhoto(p).catch(() => {}); // Einstellungen könnten sich geändert haben
  else if (p.state === 'skipped' || p.state === 'error') retry(p);
  updateChrome();
}

function finalizeDelete() {
  const entry = pendingDelete;
  if (!entry) return;
  pendingDelete = null;
  clearTimeout(entry.timer);
  const { p } = entry;
  if (p.url) URL.revokeObjectURL(p.url);
  if (p.origUrl) URL.revokeObjectURL(p.origUrl);
  p.blob = null;
  call('remove', { id: p.id }).catch(() => {});
}

/* Kurzer Hinweis unten am Bildschirm, optional mit Aktion. */
let toastTimer = 0;
function showToast(text, actionLabel, onAction, ms = 4000) {
  const t = $('#toast');
  t.querySelector('.toast-text').textContent = text;
  const btn = t.querySelector('button');
  btn.hidden = !actionLabel;
  btn.textContent = actionLabel || '';
  btn.onclick = () => { hideToast(); if (onAction) onAction(); };
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(hideToast, ms);
}
function hideToast() { clearTimeout(toastTimer); $('#toast').hidden = true; }

/* ------------------------------------------------------------- Nachbessern */

const editor = new Editor($('#editor'));
$('#edSize').addEventListener('input', (e) => setFill(e.target));
setFill($('#edSize'));

async function openEditor(p) {
  if (!p.blob) return;
  setCard(p, 'busy');
  let imageData, alpha;
  try {
    const src = await loadSource(p);
    const wc = drawScaled(src, 0, 0, p.ow, p.oh, p.ww, p.wh);
    freeCanvas(src);
    imageData = wc.getContext('2d').getImageData(0, 0, p.ww, p.wh);
    freeCanvas(wc);
    const a = await call('getAlpha', { id: p.id });
    alpha = new Uint8Array(a.alpha);
  } catch (e) { setCard(p, 'ready'); showToast('Nachbessern konnte nicht geöffnet werden.'); return; }
  setCard(p, 'ready');
  const result = await editor.open(imageData, alpha, p.ww, p.wh);
  if (!result) return;
  setCard(p, 'busy');
  try {
    const { rect } = await call('setAlpha', { id: p.id, alpha: result.buffer }, [result.buffer]);
    if (!rect) throw new Error('Alles wurde entfernt');
    p.rect = rect;
    const src = await loadSource(p);
    try { await refineFromImage(p, src); } finally { freeCanvas(src); }
    await composePhoto(p);
  } catch (e) {
    setCard(p, 'error', errorText(e) + ' Tippen zum Wiederholen.');
  }
}

/* --------------------------------------------------------------- Speichern */

function fileName(p) {
  const i = photos.indexOf(p) + 1;
  const base = p.file.name.replace(/\.[^.]+$/, '').replace(/[^\w\-]+/g, '_').slice(0, 40) || 'foto';
  return `${String(i).padStart(2, '0')}_${base}_studio.jpg`;
}

function toFile(p) { return new File([p.blob], fileName(p), { type: 'image/jpeg' }); }

function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
}

async function share(files) {
  try { await navigator.share({ files }); return true; }
  catch (e) { if (e.name === 'AbortError') return true; console.warn(e); return false; }
}

function canShareFiles(files) {
  try { return isTouch && !!navigator.canShare && navigator.canShare({ files }); } catch { return false; }
}

async function saveOne(p) {
  if (!p.blob) return;
  const f = toFile(p);
  // iPhone: Teilen-Menü -> „Bild sichern“ legt das Foto in der Fotos-App ab
  if (canShareFiles([f]) && await share([f])) return;
  download(f, f.name);
}

function openActions() {
  const ready = photos.filter((p) => p.blob);
  if (!ready.length) return;
  const shareable = canShareFiles(ready.map(toFile));
  const n = ready.length;
  $('#saveAllLabel').textContent = shareable ? `In Fotos sichern (${n})` : `Alle herunterladen (${n})`;
  $('#zipAllLabel').textContent = shareable ? 'Als ZIP sichern' : 'Als ZIP herunterladen';
  $('#actions').hidden = false; $('#actionsBackdrop').hidden = false;
  $('#saveAll').focus({ preventScroll: true });
}
function closeActions() { $('#actions').hidden = true; $('#actionsBackdrop').hidden = true; }
$('#shareAll').addEventListener('click', openActions);
$('#actionsCancel').addEventListener('click', closeActions);
$('#actionsBackdrop').addEventListener('click', closeActions);

$('#saveAll').addEventListener('click', async () => {
  closeActions();
  const ready = photos.filter((p) => p.blob);
  if (!ready.length) return;
  const files = ready.map(toFile);
  // iPhone: Teilen-Menü mit allen Bildern -> „x Bilder sichern“
  if (canShareFiles(files) && await share(files)) return;
  for (const f of files) { download(f, f.name); await new Promise((r) => setTimeout(r, 350)); }
});

let jszip = null;
$('#zipAll').addEventListener('click', async () => {
  closeActions();
  const ready = photos.filter((p) => p.blob);
  if (!ready.length) return;
  showToast('ZIP wird erstellt', null, null, 30000);
  try {
    if (!jszip) jszip = loadScript('vendor/jszip.min.js');
    await jszip;
    const zip = new window.JSZip();
    for (const p of ready) zip.file(fileName(p), p.blob);
    const blob = await zip.generateAsync({ type: 'blob', compression: 'STORE' });
    const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
    const f = new File([blob], `fotostudio_${stamp}.zip`, { type: 'application/zip' });
    hideToast();
    // Hinweis: Nach dem Erzeugen ist die Nutzergeste evtl. abgelaufen; dann Download
    if (!(canShareFiles([f]) && await share([f]))) download(f, f.name);
  } catch (e) { showToast('ZIP konnte nicht erstellt werden.'); }
});

/* --------------------------------------------------------------- Einzelansicht */

let viewerPhoto = null;
let viewMode = 'result';

function viewerList() { return photos.filter((p) => p.blob); }

function openViewer(p) {
  viewerPhoto = p; viewMode = 'result';
  $('#viewer').hidden = false;
  document.body.classList.add('locked');
  renderViewer();
  $('#viewerClose').focus({ preventScroll: true });
}
function closeViewer() {
  const p = viewerPhoto;
  viewerPhoto = null;
  $('#viewer').hidden = true;
  document.body.classList.remove('locked');
  if (p && p.el && p.el.isConnected) p.el.querySelector('.tile-btn').focus({ preventScroll: true });
}
function renderViewer() {
  const p = viewerPhoto;
  if (!p) return;
  const list = viewerList(), i = list.indexOf(p);
  $('#viewerTitle').textContent = i >= 0 ? `${i + 1} von ${list.length}` : '';
  const img = $('#viewerImg');
  const src = viewMode === 'original' ? p.origUrl : p.url;
  if (src && img.getAttribute('src') !== src) img.src = src;
  img.alt = (viewMode === 'original' ? 'Original: ' : 'Ergebnis: ') + p.file.name;
  $('.viewer-frame').classList.toggle('show-original', viewMode === 'original');
  $('#viewerBusy').hidden = !(p.state === 'busy');
  document.querySelectorAll('[data-view]').forEach((b) => b.setAttribute('aria-checked', String(b.dataset.view === viewMode)));
  $('#viewerPrev').disabled = i <= 0;
  $('#viewerNext').disabled = i < 0 || i >= list.length - 1;
  const busy = p.state === 'busy';
  $('#viewerEdit').disabled = busy; $('#viewerSave').disabled = busy;
}
function stepViewer(d) {
  const list = viewerList(), i = list.indexOf(viewerPhoto);
  const n = list[i + d];
  if (n) { viewerPhoto = n; renderViewer(); }
}
$('#viewerClose').addEventListener('click', closeViewer);
$('#viewerPrev').addEventListener('click', () => stepViewer(-1));
$('#viewerNext').addEventListener('click', () => stepViewer(1));
document.querySelectorAll('[data-view]').forEach((b) => b.addEventListener('click', () => { viewMode = b.dataset.view; renderViewer(); }));
$('#viewerEdit').addEventListener('click', () => viewerPhoto && openEditor(viewerPhoto));
$('#viewerSave').addEventListener('click', () => viewerPhoto && saveOne(viewerPhoto));
$('#viewerDelete').addEventListener('click', () => {
  const p = viewerPhoto; if (!p) return;
  const list = viewerList(), i = list.indexOf(p);
  const next = list[i + 1] || list[i - 1];
  removePhoto(p);
  if (next) { viewerPhoto = next; renderViewer(); } else closeViewer();
});
// Wischen zwischen Fotos
(() => {
  const stage = $('.viewer-stage');
  let x0 = null, y0 = 0;
  stage.addEventListener('pointerdown', (e) => { if (e.target.closest('button')) return; x0 = e.clientX; y0 = e.clientY; });
  stage.addEventListener('pointerup', (e) => {
    if (x0 === null) return;
    const dx = e.clientX - x0, dy = e.clientY - y0; x0 = null;
    if (Math.abs(dx) > 50 && Math.abs(dx) > Math.abs(dy) * 1.5) stepViewer(dx < 0 ? 1 : -1);
  });
  stage.addEventListener('pointercancel', () => { x0 = null; });
})();
window.addEventListener('keydown', (e) => {
  if (!$('#editor').hidden) return;
  if (e.key === 'Escape') {
    if (!$('#actions').hidden) closeActions();
    else if (!$('#sheet').hidden) closeSheet();
    else if (viewerPhoto) closeViewer();
  }
  if (viewerPhoto && e.key === 'ArrowRight') stepViewer(1);
  if (viewerPhoto && e.key === 'ArrowLeft') stepViewer(-1);
});

/* ------------------------------------------------------------------- PWA */

if ('serviceWorker' in navigator && location.protocol !== 'file:') {
  navigator.serviceWorker.register('sw.js').then(async () => {
    await navigator.serviceWorker.ready;
    // Der Service Worker setzt COOP/COEP-Header -> Multithreading für WASM.
    // Beim allerersten Besuch einmal neu laden, damit sie greifen.
    if (!self.crossOriginIsolated && navigator.serviceWorker.controller === null) {
      let reloaded = false;
      try { reloaded = sessionStorage.getItem('fs.coi') === '1'; sessionStorage.setItem('fs.coi', '1'); } catch {}
      if (!reloaded && !photos.length) location.reload();
    }
  }).catch((e) => console.warn('Service Worker:', e));
}

// Testhilfe (Automatisierung): Dateien ohne Dateiauswahl hinzufügen
window.__fotostudio = { addFiles, photos, settings, call, segment, editor, lowMem: LOW_MEM };
