const state = {
  conversation: [],
  autonomousTimer: null,
  cooldownMs: 10_000,
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
  localStorage.setItem("prototype_conversation", JSON.stringify(state.conversation.slice(-50)));
}

function loadConversation() {
  try {
    const saved = JSON.parse(localStorage.getItem("prototype_conversation") || "[]");
    if (Array.isArray(saved)) {
      state.conversation = saved;
      for (const item of saved) {
        addMessage(item.role === "user" ? "You" : "Prototype", item.content,
          item.role === "user" ? "user" : "prototype");
      }
    }
  } catch {
    state.conversation = [];
  }
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
    const reply = await window.PrototypeRuntime.generate(text, state.conversation);
    state.conversation.push({ role: "assistant", content: reply });
    addMessage("Prototype", reply, "prototype");
    saveConversation();
    setStatus("Prototype browser runtime ready");
    startAutonomousLoop();
  } catch (error) {
    console.error(error);
    const reply = "My browser runtime could not answer yet. The project files are still safe; please reload once the runtime is available.";
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
      const reply = await window.PrototypeRuntime.autonomous(state.conversation);
      if (!reply) return;
      state.conversation.push({ role: "assistant", content: reply });
      addMessage("Prototype", reply, "prototype");
      saveConversation();
    } catch (error) {
      console.error(error);
    }
  }, state.cooldownMs);
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

function updateIdentityDetails() {
  const originalId = localStorage.getItem("prototype_original_id");
  const isOriginal = originalId === prototypeId;
  els.details.textContent = JSON.stringify({
    runtime: "browser",
    prototype_id: prototypeId,
    identity_saved: true,
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
  }, null, 2);
}

els.form.addEventListener("submit", handleChat);
loadConversation();
setStatus("Prototype browser runtime ready");
updateIdentityDetails();
