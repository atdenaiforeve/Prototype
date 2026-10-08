// Browser-native Prototype runtime.
// This file deliberately has NO server, peer-AI, GitHub, or network connection.
// The neural model backend will be added here without changing the UI.

const Runtime = {
  name: "Prototype Browser Runtime",
  _status: "scaffold",

  status() {
    return this._status;
  },

  async generate(message, history) {
    // Safe fallback until the browser neural engine is installed.
    return "Browser runtime is ready. The neural model engine is the next part being migrated.";
  },

  async autonomous(history) {
    // The 10-second autonomous loop is preserved.
    // Returning an empty string means no autonomous message yet.
    return "";
  },
};

window.PrototypeRuntime = Runtime;
