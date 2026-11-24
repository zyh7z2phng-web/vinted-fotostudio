// Web Worker: Bildverarbeitung (Kanten, Farben, Komposition) und Ablage der
// Masken, damit die Oberfläche flüssig bleibt. Die KI selbst läuft in infer.js.
import * as pl from './pipeline.js';

const store = new Map(); // id -> { ww, wh, alpha:Uint8Array (0..255), hi }

function post(msg, transfer) { self.postMessage(msg, transfer || []); }

/** Ergebnis der KI übernehmen. */
function put({ id, alpha, w, h }) {
  store.set(id, { ww: w, wh: h, alpha: new Uint8Array(alpha), hi: null });
  return true;
}

async function refine({ id, rgba, w, h, rect }) {
  const s = store.get(id);
  if (!s) throw new Error('Foto nicht gefunden');
  const t0 = performance.now();
  s.hi = null; // altes Ergebnis vor der Neuberechnung freigeben
  const crop = pl.cropPlane(s.alpha, s.ww, rect.x, rect.y, rect.w, rect.h);
  for (let i = 0; i < crop.length; i++) crop[i] /= 255;
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
  if (!s) throw new Error('Foto nicht gefunden');
  return { alpha: s.alpha.slice().buffer, w: s.ww, h: s.wh };
}

function setAlpha({ id, alpha }) {
  const s = store.get(id);
  if (!s) throw new Error('Foto nicht gefunden');
  const a8 = new Uint8Array(alpha);
  s.alpha = a8;
  s.hi = null;
  return { rect: pl.paddedRect(a8, s.ww, s.wh, 255) };
}

const handlers = {
  put, refine, compose, getAlpha, setAlpha,
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
