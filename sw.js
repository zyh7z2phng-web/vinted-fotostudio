// Service Worker: Offline-Betrieb + Cross-Origin-Isolation (für WASM-Threads).
const VERSION = 'v5';
const SHELL = `shell-${VERSION}`;
const BIG = 'big-v2'; // Modelle + ONNX Runtime (groß, versioniert über Dateinamen)

const SHELL_FILES = [
  './', 'index.html', 'css/app.css', 'js/app.js', 'js/editor.js', 'js/worker.js', 'js/infer.js', 'js/pipeline.js',
  'manifest.webmanifest', 'icons/icons.svg', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/apple-touch-icon.png',
  'vendor/jszip.min.js',
];

const SHELL_PATHS = new Set(SHELL_FILES.map((f) => new URL(f, self.location).pathname));

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(SHELL_FILES)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys()) if (k !== SHELL && k !== BIG) await caches.delete(k);
    await self.clients.claim();
  })());
});

function isolate(res) {
  if (!res || res.status === 0 || res.type === 'opaque') return res;
  const h = new Headers(res.headers);
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Embedder-Policy', 'require-corp');
  h.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(res.body, { status: res.status, statusText: res.statusText, headers: h });
}

self.addEventListener('fetch', (e) => {
  const req = e.request;
  const url = new URL(req.url);
  if (req.method !== 'GET' || url.origin !== location.origin) return;
  const big = /\/(models|vendor)\//.test(url.pathname) && !url.pathname.endsWith('jszip.min.js');

  e.respondWith((async () => {
    if (big) {
      // Cache zuerst – Modelle werden nur einmal geladen
      const cache = await caches.open(BIG);
      const hit = await cache.match(req, { ignoreSearch: true });
      if (hit) return isolate(hit);
      const res = await fetch(req);
      if (res.ok) await cache.put(req, res.clone());
      return isolate(res);
    }
    // App-Dateien: Netz zuerst (für Updates), offline aus dem Cache
    const cache = await caches.open(SHELL);
    try {
      const res = await fetch(req);
      if (res.ok && SHELL_PATHS.has(url.pathname)) cache.put(req, res.clone());
      return isolate(res);
    } catch (err) {
      const hit = await cache.match(req, { ignoreSearch: true }) ||
        (req.mode === 'navigate' ? await cache.match('index.html') : null);
      if (hit) return isolate(hit);
      throw err;
    }
  })());
});
