// Browser port of Prototype's learning/feedback layer.
// Like the Python version, this records learning; it does not secretly change
// neural weights until the browser training engine is migrated.

export class LearningLoop {
  constructor(memory) {
    this.memory = memory;
  }

  async learnFromUser(message, conversationId = null) {
    if (!String(message).trim()) throw new Error("message must not be empty");
    return this.memory.remember(
      "User explicitly said: " + String(message).trim(),
      "experience",
      { conversation_id: conversationId }
    );
  }

  async recordResult(prompt, output, confidence = 0) {
    return this.memory.remember(
      JSON.stringify({ prompt, output, confidence }),
      "experience",
      { source: "inference" }
    );
  }
}
