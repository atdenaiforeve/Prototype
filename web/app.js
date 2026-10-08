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

  // Browser-native inference will plug into this function.
  const reply = await window.PrototypeRuntime.generate(text, state.conversation);
  state.conversation.push({ role: "assistant", content: reply });
  addMessage("Prototype", reply, "prototype");
  saveConversation();
  startAutonomousLoop();
}

function startAutonomousLoop() {
  if (state.autonomousTimer !== null) return;
  state.autonomousTimer = setInterval(async () => {
    if (!state.conversation.length) return;
    const reply = await window.PrototypeRuntime.autonomous(state.conversation);
    if (!reply) return;
    state.conversation.push({ role: "assistant", content: reply });
    addMessage("Prototype", reply, "prototype");
    saveConversation();
  }, state.cooldownMs);
}

function updateDetails() {
  els.details.textContent = JSON.stringify({
    runtime: "browser",
    storage: "localStorage",
    autonomous_cooldown_seconds: 10,
    external_ai_connection: false,
    server_connection: false,
    model_runtime: window.PrototypeRuntime.name,
    model_status: window.PrototypeRuntime.status(),
  }, null, 2);
}

els.form.addEventListener("submit", handleChat);
loadConversation();
setStatus("Prototype browser runtime ready");
updateDetails();
