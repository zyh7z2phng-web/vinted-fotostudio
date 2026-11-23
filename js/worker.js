// Web Worker: KI-Freistellung + Bildverarbeitung, damit die Oberfläche flüssig bleibt.
import * as pl from './pipeline.js';

const BASE = new URL('../', import.meta.url).href;
const MODELS = {
  // Standard: ISNet (schnell). fp16 für WebGPU, int8 für WASM.
  webgpu: { files: ['models/isnet_fp16.onnx'], size: 88_116_798, lib: 'vendor/ort/ort.webgpu.min.mjs', ep: 'webgpu', norm: pl.NORM.isnet, post: 'minmax' },
  wasm: { files: ['models/isnet_q8.onnx'], size: 44_348_381, lib: 'vendor/ort/ort.wasm.min.mjs', ep: 'wasm', norm: pl.NORM.isnet, post: 'minmax' },
};

let ort = null;
let session = null;
let current = null;   // Schlüssel in MODELS
let backend = null;   // 'webgpu' | 'wasm'
let initPromise = null;
const store = new Map(); // id -> { ww, wh, alpha:Float32Array, hi }

function post(msg, transfer) { self.postMessage(msg, transfer || []); }

async function fetchModel(files, expected, label) {
  const parts = [];
  let loaded = 0;
  for (const file of files) {
    const res = await fetch(BASE + file);
    if (!res.ok) throw new Error('Modell konnte nicht geladen werden (' + res.status + ')');
    if (!res.body || !res.body.getReader) { const b = new Uint8Array(await res.arrayBuffer()); parts.push(b); loaded += b.length; continue; }
    const reader = res.body.getReader();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      parts.push(value);
      loaded += value.length;
      post({ type: 'progress', loaded, total: Math.max(expected, loaded), label });
    }
  }
  const all = new Uint8Array(loaded);
  let o = 0;
  for (const c of parts) { all.set(c, o); o += c.length; }
  return all;
}

async function loadOrt(lib) {
  const mod = await import(BASE + lib);
  const o = mod.default || mod;
  o.env.wasm.wasmPaths = BASE + 'vendor/ort/';
  o.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, self.navigator.hardwareConcurrency || 2) : 1;
  o.env.wasm.proxy = false;
  o.env.logLevel = 'error';
  return o;
}

/** Lädt genau ein Modell (spart Speicher auf dem iPhone). */
async function useModel(key) {
  if (current === key && session) return;
  const m = MODELS[key];
  if (session) { try { await session.release(); } catch {} session = null; current = null; }
  ort = await loadOrt(m.lib);
  const bytes = await fetchModel(m.files, m.size, 'KI-Modell');
  session = await ort.InferenceSession.create(bytes, { executionProviders: [m.ep], graphOptimizationLevel: 'all' });
  current = key;
  backend = m.ep;
}

async function init(force) {
  let tryGpu = false;
  if (force !== 'wasm' && self.navigator && self.navigator.gpu) {
    try { tryGpu = !!(await self.navigator.gpu.requestAdapter()); } catch { tryGpu = false; }
  }
  if (tryGpu) {
    try { await useModel('webgpu'); }
    catch (e) { console.warn('WebGPU nicht nutzbar, wechsle zu WASM:', e); session = null; current = null; }
  }
  if (!session) await useModel('wasm');
  return info();
}

function info() {
  return { backend, model: current, threads: ort.env.wasm.numThreads, isolated: !!self.crossOriginIsolated };
}

function ensureInit(force) {
  if (!initPromise) initPromise = init(force);
  return initPromise;
}

async function runModel(tensorData) {
  const input = new ort.Tensor('float32', tensorData, [1, 3, pl.MODEL_SIZE, pl.MODEL_SIZE]);
  const out = await session.run({ [session.inputNames[0]]: input });
  const t = out[session.outputNames[0]];
  const data = t.getData ? await t.getData() : t.data;
  if (t.dispose) t.dispose();
  return Float32Array.from(data);
}

function looksBroken(arr) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < arr.length; i += 97) { const v = arr[i]; if (!Number.isFinite(v)) return true; if (v < mn) mn = v; if (v > mx) mx = v; }
  return mx - mn < 1e-4;
}

/* --------------------------------------------------------------- Befehle */

async function segment({ id, rgba, w, h }) {
  await ensureInit();
  const t0 = performance.now();
  const px = new Uint8ClampedArray(rgba);
  let m = MODELS[current];
  let out = await runModel(pl.prepareInput(px, w, h, m.norm));
  if (backend === 'webgpu' && looksBroken(out)) {
    // Sicherheitsnetz: manche GPU/Browser-Kombinationen liefern Unsinn -> WASM
    console.warn('WebGPU-Ergebnis ungültig, wechsle zu WASM');
    await useModel('wasm');
    post({ type: 'backend', ...info() });
    m = MODELS[current];
    out = await runModel(pl.prepareInput(px, w, h, m.norm));
  }
  const tModel = performance.now() - t0;
  let alpha = pl.postprocessMask(out, w, h, m.post);
  const cleaned = pl.cleanupAlpha(alpha, w, h);
  alpha = cleaned.alpha;
  store.set(id, { ww: w, wh: h, alpha, hi: null });
  return { rect: paddedRect(alpha, w, h), empty: cleaned.empty, ms: Math.round(performance.now() - t0), msModel: Math.round(tModel), model: current };
}

function paddedRect(alpha, w, h) {
  const bb = pl.bboxOf(alpha, w, h, 0.1);
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

async function refine({ id, rgba, w, h, rect }) {
  const s = store.get(id);
  if (!s) throw new Error('Foto nicht gefunden');
  const t0 = performance.now();
  const crop = pl.cropPlane(s.alpha, s.ww, rect.x, rect.y, rect.w, rect.h);
  const up = pl.resample(crop, rect.w, rect.h, w, h);
  const hi = pl.refineHi(new Uint8ClampedArray(rgba), w, h, up);
  hi.w = w; hi.h = h;
  s.hi = hi;
  return { ms: Math.round(performance.now() - t0), empty: !hi.bbox };
}

async function encodeJpeg(img, quality) {
  if (typeof OffscreenCanvas !== 'undefined') {
    try {
      const c = new OffscreenCanvas(img.w, img.h);
      const ctx = c.getContext('2d');
      ctx.putImageData(new ImageData(img.data, img.w, img.h), 0, 0);
      const blob = await c.convertToBlob({ type: 'image/jpeg', quality });
      c.width = c.height = 1;
      return { blob };
    } catch (e) { /* Fallback unten */ }
  }
  return { raw: img.data.buffer, w: img.w, h: img.h };
}

async function compose({ id, settings, quality = 0.92 }) {
  const s = store.get(id);
  if (!s || !s.hi) throw new Error('Foto noch nicht bereit');
  const t0 = performance.now();
  const img = pl.compose(s.hi, settings);
  const enc = await encodeJpeg(img, quality);
  return { ...enc, ms: Math.round(performance.now() - t0) };
}

function getAlpha({ id }) {
  const s = store.get(id);
  const a = new Uint8Array(s.ww * s.wh);
  for (let i = 0; i < a.length; i++) a[i] = s.alpha[i] * 255 + 0.5;
  return { alpha: a.buffer, w: s.ww, h: s.wh };
}

function setAlpha({ id, alpha }) {
  const s = store.get(id);
  const a8 = new Uint8Array(alpha);
  const f = new Float32Array(a8.length);
  for (let i = 0; i < f.length; i++) f[i] = a8[i] / 255;
  s.alpha = f;
  s.hi = null;
  return { rect: paddedRect(f, s.ww, s.wh) };
}

const handlers = {
  init: ({ force }) => ensureInit(force),
  segment, refine, compose, getAlpha, setAlpha,
  remove: ({ id }) => { store.delete(id); return true; },
};

self.onmessage = async (e) => {
  const { rid, cmd, args } = e.data;
  try {
    const result = await handlers[cmd](args || {});
    const transfer = [];
    if (result && result.raw) transfer.push(result.raw);
    if (result && result.alpha instanceof ArrayBuffer) transfer.push(result.alpha);
    post({ rid, ok: true, result }, transfer);
  } catch (err) {
    console.error(err);
    post({ rid, ok: false, error: String((err && err.message) || err) });
  }
};
