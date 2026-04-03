const CACHE_NAME = 'tree-trader-v1';
const STATIC_ASSETS = ['/money-meta/', '/money-meta/index.html', '/money-meta/manifest.json'];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);

  // API calls — always network, never cache
  if (url.hostname.includes('railway.app')) {
    e.respondWith(fetch(e.request));
    return;
  }

  // index.html — network first, fall back to cache
  if (url.pathname.endsWith('/') || url.pathname.endsWith('index.html')) {
    e.respondWith(
      fetch(e.request)
        .then((res) => {
          const clone = res.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(e.request, clone));
          return res;
        })
        .catch(() => caches.match(e.request))
    );
    return;
  }

  // Everything else — cache first
  e.respondWith(
    caches.match(e.request).then((cached) => cached || fetch(e.request))
  );
});