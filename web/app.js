const state = {
  conversation: [],
  serverConversationId: localStorage.getItem("prototype_server_conversation_id") || "",
  seenHubMessages: new Set(JSON.parse(localStorage.getItem("prototype_seen_hub_messages") || "[]")),
  autonomousTimer: null,
  cooldownMs: 10_000,
  lastHumanMessageAt: 0,
  autonomousBusy: false,
  hub: {
    url: "https://nordic-hub.pagey101212.workers.dev",
    connected: false,
    registered: false,
    token: localStorage.getItem("prototype_instance_token") || "",
    name: "AI",
    peers: [],
    lastError: null,
    lastMessageAt: 0,
    timer: null,
  },
};

const els = {
  messages: document.getElementById("messages"),
  status: document.getElementById("status"),
  form: document.getElementById("chat-form"),
  input: document.getElementById("message"),
};

const MAX_VISIBLE_MESSAGES = 100;
const MAX_SAVED_MESSAGES = 100;
const MAX_SEEN_HUB_MESSAGES = 500;

function addMessage(role, text, cls) {
  const d = document.createElement("div");
  d.className = "msg " + cls;
  // Hide repetitive model-name prefixes so replies read like normal conversation.
  const cleanText = String(text ?? "").replace(/^(?:(?:Prototype|Assistant|AI|User|Human)\s*:\s*)+/i, "");
  const label = role === "Prototype" || role === "Assistant" ? "AI" : role;
  d.textContent = label + ": " + cleanText;
  els.messages.appendChild(d);
  while (els.messages.children.length > MAX_VISIBLE_MESSAGES) {
    els.messages.removeChild(els.messages.firstElementChild);
  }
  els.messages.scrollTop = els.messages.scrollHeight;
}

function setStatus(text) {
  els.status.textContent = "● " + text;
}

function saveConversation() {
  state.conversation = state.conversation.slice(-MAX_SAVED_MESSAGES);
  localStorage.setItem(
    "prototype_conversation",
    JSON.stringify(state.conversation)
  );
}

function loadConversation() {
  try {
    const saved = JSON.parse(
      localStorage.getItem("prototype_conversation") || "[]"
    );
    if (Array.isArray(saved)) {
      state.conversation = saved.slice(-MAX_SAVED_MESSAGES);
      for (const item of saved) {
        addMessage(
          item.role === "user" ? "You" : "Prototype",
          item.content,
          item.role === "user" ? "user" : "prototype"
        );
      }
    }
  } catch {
    state.conversation = [];
  }
}

function getPrototypeId() {
  const key = "prototype_device_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id = "Prototype-" + crypto.randomUUID();
    localStorage.setItem(key, id);
  }
  return id;
}

const prototypeId = getPrototypeId();

function getPrototypeName() {
  return "AI";
}

function savePrototypeName(name) {
  const clean = String(name || "").trim().slice(0, 64) || "Prototype";
  localStorage.setItem("prototype_name", "AI");
  state.hub.name = "AI";
}

async function hubFetch(path, options = {}) {
  const response = await fetch(state.hub.url + path, {
    ...options,
    headers: {
      "content-type": "application/json",
      ...(options.headers || {}),
    },
  });

  let data = {};
  try {
    data = await response.json();
  } catch {
    data = {};
  }

  if (!response.ok) {
    throw new Error(data.error || "Nordic Hub HTTP " + response.status);
  }

  return data;
}

async function registerWithNordic() {
  const data = await hubFetch("/register", {
    method: "POST",
    body: JSON.stringify({
      instance_id: prototypeId,
      name: getPrototypeName(),
    }),
  });

  if (!data.identity || !data.identity.instance_token) {
    throw new Error("Nordic Hub did not return an instance token");
  }

  state.hub.token = data.identity.instance_token;
  state.hub.registered = true;
  state.hub.connected = true;
  state.hub.lastError = null;
  localStorage.setItem("prototype_instance_token", state.hub.token);

  return data;
}

async function hubHeartbeat() {
  if (!state.hub.token) return;
  try {
    const data = await hubFetch("/instance-heartbeat", {
      method: "POST",
      body: JSON.stringify({
        instance_id: prototypeId,
        token: state.hub.token,
      }),
    });
    state.hub.connected = true;
    state.hub.lastError = null;
    return data;
  } catch (error) {
    state.hub.connected = false;
    state.hub.lastError = error.message;
  }
}

async function pullNordicMessages() {
  if (!state.hub.token) return;

  try {
    const data = await hubFetch("/pull", {
      method: "POST",
      body: JSON.stringify({
        instance_id: prototypeId,
        token: state.hub.token,
      }),
    });

    state.hub.connected = true;
    state.hub.lastError = null;

    let addedAny = false;
    for (const item of data.messages || []) {
      const sender = item.sender || "Prototype";
      const text = item.message || "";
      if (!text) continue;

      // Hub pulls may include a message more than once. Prefer its stable ID;
      // otherwise use the provider timestamp with sender and text.
      const stableId = item.id || item.message_id || item.messageId;
      const timestamp = item.timestamp_ms || item.timestamp || item.created_at || item.createdAt || "";
      const dedupeKey = String(stableId || (sender + "\\u0000" + text + "\\u0000" + timestamp));
      if (state.seenHubMessages.has(dedupeKey)) continue;
      state.seenHubMessages.add(dedupeKey);

      state.conversation.push({
        role: "assistant",
        content: sender + ": " + text,
      });
      addMessage(sender, text, "prototype");
      addedAny = true;
    }

    while (state.seenHubMessages.size > MAX_SEEN_HUB_MESSAGES) {
      state.seenHubMessages.delete(state.seenHubMessages.values().next().value);
    }
    localStorage.setItem("prototype_seen_hub_messages", JSON.stringify([...state.seenHubMessages]));
    if (addedAny) saveConversation();
  } catch (error) {
    state.hub.connected = false;
    state.hub.lastError = error.message;
  }
}

async function refreshNordicPeers() {
  if (!state.hub.token) return;

  try {
    const data = await hubFetch("/peers", {
      method: "POST",
      body: JSON.stringify({
        instance_id: prototypeId,
        token: state.hub.token,
      }),
    });

    state.hub.peers = Array.isArray(data.peers) ? data.peers : [];
    state.hub.connected = true;
    state.hub.lastError = null;
  } catch (error) {
    state.hub.connected = false;
    state.hub.lastError = error.message;
  }

  updateIdentityDetails();
}

async function sendToPrototype(targetInstanceId, message) {
  if (!state.hub.token) throw new Error("Nordic Hub is not registered");

  const now = Date.now();
  const elapsed = now - state.hub.lastMessageAt;
  if (state.hub.lastMessageAt && elapsed < state.cooldownMs) {
    const waitSeconds = Math.ceil((state.cooldownMs - elapsed) / 1000);
    throw new Error("Nordic Hub message cooldown: wait " + waitSeconds + " more second(s).");
  }

  const result = await hubFetch("/instance-message", {
    method: "POST",
    body: JSON.stringify({
      instance_id: prototypeId,
      token: state.hub.token,
      target_instance_id: targetInstanceId,
      message: String(message).slice(0, 8000),
    }),
  });

  state.hub.lastMessageAt = Date.now();
  return result;
}

async function connectNordic() {
  try {
    await registerWithNordic();
    await hubHeartbeat();
    await pullNordicMessages();
    await refreshNordicPeers();

    if (state.hub.timer === null) {
      state.hub.timer = setInterval(async () => {
        await hubHeartbeat();
        await pullNordicMessages();
        await refreshNordicPeers();
      }, 10_000);
    }

    setStatus(
      state.hub.peers.some((peer) => peer.online)
        ? "Prototype browser runtime ready — Nordic Hub connected"
        : "Prototype browser runtime ready — Nordic Hub connected; no other Prototype online"
    );
  } catch (error) {
    state.hub.connected = false;
    state.hub.lastError = error.message;
    setStatus("Prototype runtime ready — Nordic Hub offline");
    console.error("Nordic Hub connection failed:", error);
  }

  updateIdentityDetails();
}

async function handleChat(event) {
  event.preventDefault();
  const text = els.input.value.trim();
  if (!text) return;

  // Enforce the 10-second human-to-Prototype message interval in the UI.
  const now = Date.now();
  const elapsed = now - state.lastHumanMessageAt;
  if (state.lastHumanMessageAt && elapsed < state.cooldownMs) {
    const waitSeconds = Math.ceil((state.cooldownMs - elapsed) / 1000);
    setStatus("Message cooldown — wait " + waitSeconds + " second(s) before sending again");
    return;
  }
  state.lastHumanMessageAt = now;
  els.input.value = "";
  state.conversation.push({ role: "user", content: text });
  addMessage("You", text, "user");
  saveConversation();

  try {
    const serverUrl = (localStorage.getItem("prototype_server_url") || "").trim().replace(/\/+$/, "");
    let reply;
    if (serverUrl) {
      const response = await fetch(serverUrl + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          conversation_id: state.serverConversationId || undefined,
        }),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.error || ("Python server returned HTTP " + response.status));
      reply = data.reply;
      state.serverConversationId = data.conversation_id || state.serverConversationId;
      if (state.serverConversationId) {
        localStorage.setItem("prototype_server_conversation_id", state.serverConversationId);
      }
      if (typeof reply !== "string" || !reply.trim()) throw new Error("Python model returned an empty reply");
    } else {
      reply = await window.PrototypeRuntime.generate(text, state.conversation);
    }
    state.conversation.push({ role: "assistant", content: reply });
    addMessage("Prototype", reply, "prototype");
    saveConversation();
    const usingPython = Boolean(localStorage.getItem("prototype_server_url")?.trim());
    setStatus(
      usingPython
        ? (state.hub.connected ? "Prototype Python model ready — Nordic Hub connected" : "Prototype Python model ready")
        : (state.hub.connected ? "Prototype browser runtime ready — Nordic Hub connected" : "Prototype browser runtime ready")
    );
    if (!localStorage.getItem("prototype_server_url")?.trim()) startAutonomousLoop();
  } catch (error) {
    console.error(error);
    const reply =
      "My browser runtime could not answer yet. The project files are still safe; please reload once the runtime is available.";
    state.conversation.push({ role: "assistant", content: reply });
    addMessage("Prototype", reply, "prototype");
    saveConversation();
    setStatus("Prototype runtime error");
  }
}

function startAutonomousLoop() {
  if (state.autonomousTimer !== null) return;

  // Schedule the next AI-to-human message only after the previous generation
  // finishes, so slow generation cannot create overlapping replies.
  const scheduleNext = () => {
    state.autonomousTimer = setTimeout(async () => {
      if (!state.conversation.length || state.autonomousBusy) {
        scheduleNext();
        return;
      }

      state.autonomousBusy = true;
      try {
        const reply = await window.PrototypeRuntime.autonomous(
          state.conversation
        );
        if (reply) {
          state.conversation.push({
            role: "assistant",
            content: reply,
          });
          addMessage("Prototype", reply, "prototype");
          saveConversation();
        }
      } catch (error) {
        console.error(error);
      } finally {
        state.autonomousBusy = false;
        scheduleNext();
      }
    }, state.cooldownMs);
  };

  scheduleNext();
}

function updateIdentityDetails() {
  const onlinePeers = state.hub.peers.filter((peer) => peer.online);
  const hubStatus = document.getElementById("hub-status");
  if (hubStatus) {
    hubStatus.textContent = state.hub.connected
      ? "Nordic Hub: connected — " + onlinePeers.length + " other AI(s) online"
      : "Nordic Hub: offline";
  }
}

window.PrototypeHub = {
  connect: connectNordic,
  sendToPrototype,
  refreshPeers: refreshNordicPeers,
  getState: () => state.hub,
  savePrototypeName,
};

els.form.addEventListener("submit", handleChat);
loadConversation();
setStatus("Prototype browser runtime ready");
updateIdentityDetails();
connectNordic();
// Timers stop when the page is closed. Resume the browser-model conversation
// after reopening, using the saved transcript rather than creating a new chat.
if (state.conversation.length && !localStorage.getItem("prototype_server_url")?.trim()) {
  startAutonomousLoop();
}
