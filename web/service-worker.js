const CACHE = "prototype-browser-v3";
const APP_ASSETS = [
  "./",
  "./index.html",
  "./app.js",
  "./runtime.js",
  "./tokenizer.js",
  "./memory.js",
  "./reasoning.js",
  "./learning.js"
];
const TRAINING_FILES = [
  "ai.txt","basics.txt","computers_programming.txt","context.txt","conversation.txt",
  "conversation_instructions.txt","conversation_skills.txt","earth.txt","english_basics.txt",
  "factual_reasoning.txt","humans.txt","language.txt","logic_reasoning.txt","mathematics.txt",
  "matter_energy.txt","memory_learning.txt","nova_protocol.txt","plants.txt","self_model.txt",
  "space.txt","tools_agents.txt","weather_climate.txt"
];

self.addEventListener("install", event => {
  event.waitUntil(caches.open(CACHE).then(cache => cache.addAll(APP_ASSETS)));
  self.skipWaiting();
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", event => {
  if (event.request.method !== "GET") return;
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  const isAppAsset = url.pathname.includes("/web/");
  const isTraining = url.pathname.includes("/data/training/");
  if (!isAppAsset && !isTraining) return;

  // Prefer current published files; use cache only when offline.
  event.respondWith((async () => {
    const cache = await caches.open(CACHE);
    try {
      const response = await fetch(event.request, { cache: "no-cache" });
      if (response.ok) await cache.put(event.request, response.clone());
      return response;
    } catch {
      const cached = await cache.match(event.request);
      if (cached) return cached;
      return new Response("Prototype asset unavailable offline.", {
        status: 503,
        headers: { "content-type": "text/plain; charset=utf-8" }
      });
    }
  })());
});
