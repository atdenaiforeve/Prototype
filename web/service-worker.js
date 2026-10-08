const CACHE = "prototype-browser-v2";
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
  const url = new URL(event.request.url);
  const isTraining = url.pathname.includes("/data/training/");
  if (isTraining || url.pathname.includes("/web/")) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        if (cached) return cached;
        return fetch(event.request).then(response => {
          if (response.ok && (isTraining || url.pathname.includes("/web/"))) {
            const copy = response.clone();
            caches.open(CACHE).then(cache => cache.put(event.request, copy));
          }
          return response;
        });
      })
    );
  }
});
