const state = {
  conversation: [],
  autonomousTimer: null,
  cooldownMs: 10_000,
  hub: {
    url: "https://nordic-hub.pagey101212.workers.dev",
    connected: false,
    registered: false,
    token: localStorage.getItem("prototype_instance_token") || "",
    name: localStorage.getItem("prototype_name") || "Prototype",
    peers: [],
    lastError: null,
    timer: null,
  },
};

const els = {
  messages: document.getElementById("messages"),
  status: document.getElementById("status"),
  details: document.getElementById("details"),
  form: document.getElementById("chat-form"),
  input: document.getElementById("message"),
};

function addMessage(role, text, cls) {
  const d = document.createElement("div");
  d.className = "msg " + cls;
  d.textContent = role + ": " + text;
  els.messages.appendChild(d);
  els.messages.scrollTop = els.messages.scrollHeight;
}

function setStatus(text) {
  els.status.textContent = "● " + text;
}

function saveConversation() {
  localStorage.setItem(
    "prototype_conversation",
    JSON.stringify(state.conversation.slice(-50))
  );
}

function loadConversation() {
  try {
    const saved = JSON.parse(
      localStorage.getItem("prototype_conversation") || "[]"
    );
    if (Array.isArray(saved)) {
      state.conversation = saved;
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
  return (localStorage.getItem("prototype_name") || "Prototype").trim() || "Prototype";
}

function savePrototypeName(name) {
  const clean = String(name || "").trim().slice(0, 64) || "Prototype";
  localStorage.setItem("prototype_name", clean);
  state.hub.name = clean;
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

    for (const item of data.messages || []) {
      const sender = item.sender || "Prototype";
      const text = item.message || "";
      if (!text) continue;

      state.conversation.push({
        role: "assistant",
        content: sender + ": " + text,
      });
      addMessage(sender, text, "prototype");
    }

    if ((data.messages || []).length) saveConversation();
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

  return hubFetch("/instance-message", {
    method: "POST",
    body: JSON.stringify({
      instance_id: prototypeId,
      token: state.hub.token,
      target_instance_id: targetInstanceId,
      message: String(message).slice(0, 8000),
    }),
  });
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

  els.input.value = "";
  state.conversation.push({ role: "user", content: text });
  addMessage("You", text, "user");
  saveConversation();

  try {
    const reply = await window.PrototypeRuntime.generate(
      text,
      state.conversation
    );
    state.conversation.push({ role: "assistant", content: reply });
    addMessage("Prototype", reply, "prototype");
    saveConversation();
    setStatus(
      state.hub.connected
        ? "Prototype browser runtime ready — Nordic Hub connected"
        : "Prototype browser runtime ready"
    );
    startAutonomousLoop();
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

  state.autonomousTimer = setInterval(async () => {
    if (!state.conversation.length) return;

    try {
      const reply = await window.PrototypeRuntime.autonomous(
        state.conversation
      );
      if (!reply) return;

      state.conversation.push({
        role: "assistant",
        content: reply,
      });
      addMessage("Prototype", reply, "prototype");
      saveConversation();
    } catch (error) {
      console.error(error);
    }
  }, state.cooldownMs);
}

function updateIdentityDetails() {
  const originalId = localStorage.getItem("prototype_original_id");
  const isOriginal = originalId === prototypeId;
  const onlinePeers = state.hub.peers.filter((peer) => peer.online);

  els.details.textContent = JSON.stringify(
    {
      runtime: "browser",
      prototype_id: prototypeId,
      prototype_name: getPrototypeName(),
      identity_saved: true,
      nordic_hub: {
        connected: state.hub.connected,
        registered: state.hub.registered,
        online_other_prototypes: onlinePeers.length,
        known_other_prototypes: state.hub.peers.length,
        last_error: state.hub.lastError,
      },
      original_id_known: Boolean(originalId),
      code_editor: isOriginal ? "enabled" : "locked",
      autonomous_cooldown_seconds: 10,
      ai_to_ai_cooldown_seconds: 0,
      external_ai_connection: false,
      mcp_required_for_chat: false,
      server_connection: false,
      codespaces_required_for_chat: false,
      model_runtime: window.PrototypeRuntime.name,
      model_status: window.PrototypeRuntime.status(),
    },
    null,
    2
  );
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
