// Browser port of Prototype's custom tokenizer foundation.
// BPE vocabulary/merge loading will be connected to the existing training data.

export class PrototypeTokenizer {
  constructor(vocab = {}, merges = []) {
    this.vocab = vocab;
    this.merges = merges;
  }

  basicTokens(text) {
    return String(text).match(/\\w+|[^\\w\\s]/g) || [];
  }

  encode(text) {
    return this.basicTokens(text).map(token => this.vocab[token] ?? this.vocab["<UNK>"] ?? 1);
  }

  decode(ids) {
    const reverse = new Map(Object.entries(this.vocab).map(([token, id]) => [Number(id), token]));
    return ids.map(id => reverse.get(Number(id)) ?? "<UNK>").join(" ");
  }
}
