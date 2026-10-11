// Browser-native Prototype runtime.
// This is a small corpus-based browser engine, not the trained Python Transformer.
const FILES = [
  "ai.txt", "anti_garble.txt", "basics.txt", "computers_programming.txt", "context.txt",
  "conversation.txt", "conversation_instructions.txt", "conversation_skills.txt",
  "earth.txt", "english_basics.txt", "factual_reasoning.txt", "humans.txt",
  "language.txt", "logic_reasoning.txt", "mathematics.txt", "matter_energy.txt",
  "memory_learning.txt", "nova_protocol.txt", "plants.txt", "self_model.txt",
  "space.txt", "tools_agents.txt", "weather_climate.txt"
];

const Runtime = {
  name: "Prototype Browser Language Engine",
  _status: "loading",
  _next: new Map(),
  _nextOne: new Map(),
  _answers: [],
  _sentences: [],
  _ready: null,
  _words: 0,
  _trainingFiles: 0,

  status() { return this._status; },

  async init() {
    if (this._ready) return this._ready;
    this._ready = (async () => {
      try {
        const results = await Promise.all(FILES.map(async (file) => {
          const response = await fetch("../data/training/" + file, { cache: "no-cache" });
          if (!response.ok) throw new Error("Training file " + file + " returned HTTP " + response.status);
          return response.text();
        }));
        this._trainingFiles = results.length;
        this.train(results.join("\n\n"));
        this._status = "ready";
      } catch (error) {
        console.error("[Prototype] Could not load the full training curriculum:", error);
        this._status = "partial";
        // Keep the app usable, but never pretend this tiny fallback is the trained model.
        this.train("User: hi\nAssistant: Hi! What can I help you with?\n\nUser: hello\nAssistant: Hello! What are you working on?\n\nUser: what should you say if you don't know\nAssistant: I don't know.");
      }
      return this;
    })();
    return this._ready;
  },

  normalize(text) {
    return String(text || "").toLowerCase().replace(/[^a-z0-9' ]+/g, " ").replace(/\s+/g, " ").trim();
  },

  tokenize(text) {
    return String(text || "").toLowerCase().match(/[a-z0-9']+|[.,!?;:]/g) || [];
  },

  train(text) {
    this._next.clear();
    this._nextOne.clear();
    this._answers = [];
    this._sentences = [];

    const lines = String(text).split(/\r?\n/);
    for (let i = 0; i < lines.length; i++) {
      const question = lines[i].match(/^\s*(?:User|You):\s*(.+?)\s*$/i);
      if (!question) continue;
      let j = i + 1;
      while (j < lines.length && !lines[j].trim()) j++;
      if (j >= lines.length) continue;
      const answer = lines[j].match(/^\s*(?:Assistant|Prototype):\s*(.+?)\s*$/i);
      if (answer && question[1] && answer[1]) {
        this._answers.push({
          question: question[1].trim(),
          normalized: this.normalize(question[1]),
          answer: answer[1].trim()
        });
      }
    }

    const chunks = String(text).match(/[^.!?\n]+[.!?]?/g) || [];
    this._sentences = chunks.map(s => s.trim()).filter(s => s.length >= 25 && s.length <= 260);

    const words = this.tokenize(text);
    this._words = words.length;
    for (let i = 0; i < words.length - 2; i++) {
      const first = words[i];
      const second = words[i + 1];
      const third = words[i + 2];
      const key = first + "\0" + second;
      if (!this._next.has(key)) this._next.set(key, new Map());
      const pairs = this._next.get(key);
      pairs.set(third, (pairs.get(third) || 0) + 1);
      if (!this._nextOne.has(second)) this._nextOne.set(second, new Map());
      const singles = this._nextOne.get(second);
      singles.set(third, (singles.get(third) || 0) + 1);
    }
  },

  pick(map) {
    if (!map || !map.size) return null;
    const options = [...map.entries()].sort((a, b) => b[1] - a[1]).slice(0, 8);
    const total = options.reduce((sum, item) => sum + item[1], 0);
    let pick = Math.random() * total;
    for (const [word, count] of options) {
      pick -= count;
      if (pick <= 0) return word;
    }
    return options[0][0];
  },

  quickReply(prompt) {
    const text = this.normalize(prompt);
    if (/^(hi|hello|hey|hiya|sup|yo)( prototype)?$/.test(text)) return "Hi! I'm here. What would you like to work on?";
    if (/^(you good|are you good|are you okay|you ok|are you alright|how are you)$/.test(text)) return "I'm running and ready to chat. What are we working on?";
    if (/^(are you there|prototype are you there)$/.test(text)) return "Yes, I'm here and ready to respond.";
    if (/^(thanks|thank you|cheers)$/.test(text)) return "You're welcome!";
    return null;
  },

  retrieveAnswer(prompt) {
    const normalized = this.normalize(prompt);
    if (!normalized) return null;

    const exact = this._answers.find(item => item.normalized === normalized);
    if (exact) return exact.answer;

    const stop = new Set(["a", "an", "the", "is", "are", "am", "i", "you", "me", "my", "your", "to", "of", "for", "do", "does", "did", "can", "could", "would", "should", "what", "why", "how", "when", "where", "it", "this", "that", "and", "or", "in", "on"]);
    const queryWords = new Set(normalized.split(" ").filter(word => word.length > 2 && !stop.has(word)));
    if (!queryWords.size) return null;

    let best = null;
    let bestScore = 0;
    for (const item of this._answers) {
      const candidateWords = new Set(item.normalized.split(" ").filter(word => word.length > 2 && !stop.has(word)));
      if (!candidateWords.size) continue;
      let overlap = 0;
      for (const word of queryWords) if (candidateWords.has(word)) overlap++;
      const score = overlap / Math.max(queryWords.size, candidateWords.size);
      if (score > bestScore) {
        best = item;
        bestScore = score;
      }
    }
    return bestScore >= 0.72 ? best.answer : null;
  },

  retrieveSentence(prompt) {
    const queryWords = new Set(this.normalize(prompt).split(" ").filter(word => word.length > 3));
    if (!queryWords.size) return null;
    let best = null;
    let bestScore = 0;
    for (const sentence of this._sentences) {
      const words = new Set(this.normalize(sentence).split(" ").filter(word => word.length > 3));
      if (!words.size) continue;
      let overlap = 0;
      for (const word of queryWords) if (words.has(word)) overlap++;
      const score = overlap / queryWords.size;
      if (score > bestScore) {
        best = sentence;
        bestScore = score;
      }
    }
    return bestScore >= 0.6 ? best : null;
  },

  local(prompt, history = []) {
    const retrieved = this.retrieveAnswer(prompt);
    if (retrieved) return retrieved;

    const relatedSentence = this.retrieveSentence(prompt);
    if (relatedSentence) return relatedSentence;

    // Back off from a two-token transition to a one-token transition.
    const context = [...history.slice(-4).map(item => item.content || ""), prompt].join(" ");
    const words = this.tokenize(context);
    let first = words[words.length - 2] || "prototype";
    let second = words[words.length - 1] || "is";
    const output = [];

    for (let i = 0; i < 35; i++) {
      const next = this.pick(this._next.get(first + "\0" + second))
        || this.pick(this._nextOne.get(second));
      if (!next) break;
      output.push(next);
      first = second;
      second = next;
      if (output.length >= 6 && /[.!?]/.test(next)) break;
    }

    if (output.length >= 3) {
      const result = output.join(" ").replace(/\s+([.,!?;:])/g, "$1");
      return result.charAt(0).toUpperCase() + result.slice(1);
    }

    return "I don't know the answer to that yet. Try asking in a different way.";
  },

  async generate(message, history = []) {
    await this.init();
    return this.local(message, history);
  },

  async autonomous(history = []) {
    await this.init();
    const prompts = [
      "what should we test after training",
      "what is the difference between memory and training",
      "what does a good training example look like",
      "what should you do when a question is unclear"
    ];
    const prompt = prompts[Math.floor(Math.random() * prompts.length)];
    return this.local(prompt, history);
  },

  stats() {
    return {
      status: this._status,
      learned_words: this._words,
      training_files_loaded: this._trainingFiles,
      question_answer_examples: this._answers.length,
      transition_pairs: this._next.size,
      external_ai_connection: false,
      server_connection: false,
      human_autonomous_cooldown_seconds: 10,
      ai_to_ai_cooldown_seconds: 10
    };
  }
};

window.PrototypeRuntime = Runtime;
Runtime.init();
