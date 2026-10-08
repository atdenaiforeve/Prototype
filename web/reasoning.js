// Browser port of Prototype's explicit structured reasoning workspace.

export class ReasoningWorkspace {
  constructor(maxHypotheses = 8, maxEvidence = 8) {
    this.maxHypotheses = maxHypotheses;
    this.maxEvidence = maxEvidence;
    this.hypotheses = [];
    this.decisions = [];
  }

  addHypothesis(text, confidence = 0.5) {
    const hypothesis = {
      text: String(text).trim(),
      confidence: Math.max(0, Math.min(1, Number(confidence))),
      evidence: [],
      status: "active",
    };
    this.hypotheses.push(hypothesis);
    this.hypotheses = this.hypotheses.slice(-this.maxHypotheses);
    return hypothesis;
  }

  addEvidence(hypothesis, evidence) {
    hypothesis.evidence.push(String(evidence).trim());
    hypothesis.evidence = hypothesis.evidence.slice(-this.maxEvidence);
  }

  choose() {
    const active = this.hypotheses.filter(h => h.status === "active");
    if (!active.length) return null;
    const chosen = active.reduce((a, b) => a.confidence >= b.confidence ? a : b);
    for (const h of active) h.status = h === chosen ? "chosen" : "rejected";
    this.decisions.push(chosen.text);
    this.decisions = this.decisions.slice(-this.maxHypotheses);
    return chosen;
  }

  snapshot() {
    return {
      hypotheses: structuredClone(this.hypotheses),
      decisions: [...this.decisions],
    };
  }

  clear() {
    this.hypotheses = [];
    this.decisions = [];
  }
}
