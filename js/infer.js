// Web Worker nur für die KI-Freistellung (ONNX Runtime).
// Läuft getrennt von der übrigen Bildverarbeitung, damit die Oberfläche ihn
// beenden kann: Der WebAssembly-Speicher der Runtime (bei ISNet ~700 MB nach der
// ersten Berechnung) wird sonst nie wieder freigegeben. Auf dem iPhone führte
// das zusammen mit der restlichen Verarbeitung zum Absturz des Tabs.
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
let maxThreads = 4;  // 1 auf dem iPhone: weniger Speicher, wird beim Beenden sofort frei
function toAlpha8(f) {
  const a = new Uint8Array(f.length);
  for (let i = 0; i < a.length; i++) a[i] = f[i] * 255 + 0.5;
  return a;
}

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
  o.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(maxThreads, self.navigator.hardwareConcurrency || 2)) : 1;
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

function ensureInit(force, threads) {
  if (threads) maxThreads = threads;
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

async function segment({ rgba, w, h, force, threads }) {
  await ensureInit(force, threads);
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
  // Arbeits-Alpha als 8 Bit weitergeben (spart pro Foto ~5 MB gegenüber Float32)
  const alpha8 = toAlpha8(alpha);
  return { alpha: alpha8.buffer, rect: pl.paddedRect(alpha, w, h), empty: cleaned.empty, ms: Math.round(performance.now() - t0), msModel: Math.round(tModel), model: current };
}

const handlers = {
  init: ({ force, threads }) => ensureInit(force, threads),
  segment,
};

self.onmessage = async (e) => {
  const { rid, cmd, args } = e.data;
  try {
    const result = await handlers[cmd](args || {});
    const transfer = [];
    if (result && result.alpha instanceof ArrayBuffer) transfer.push(result.alpha);
    post({ rid, ok: true, result }, transfer);
  } catch (err) {
    console.error(err);
    post({ rid, ok: false, error: String((err && err.message) || err) });
  }
};
