"use strict";
// yomikiri AI: online-learning RPS predictor with honest explanations.
// Shared by index.html (browser) and sim_balance.js (node).
// Depends on headForward/headTrainStep from cnn.js (parity-verified SGD engine).
(function (root) {
  const CNN = (typeof module !== 'undefined') ? require('./cnn.js') : root;
  const headForward = CNN.headForward, headTrainStep = CNN.headTrainStep;

  const MOVES = ['グー', 'チョキ', 'パー'];
  const EMO = ['✊', '✌️', '🖐'];
  // a beats b?
  function beats(a, b) { return (a === 0 && b === 1) || (a === 1 && b === 2) || (a === 2 && b === 0); }
  function counter(m) { return (m + 2) % 3; } // the move that beats m

  function mulberry32(seed) { let a = seed >>> 0;
    return function () { a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }

  function randomDeal(rng) { // 12 cards, each type 1..7 (no degenerate hands)
    for (;;) {
      const c = [0, 0, 0];
      for (let i = 0; i < 12; i++) c[Math.floor(rng() * 3)]++;
      if (c.every(v => v >= 1 && v <= 7)) return c;
    }
  }

  function newMatchState(pInit, aInit) {
    return { pHist: [], aHist: [], results: [], // results from player's perspective: 'win'|'lose'|'draw'
      pCards: (pInit || [4, 4, 4]).slice(), aCards: (aInit || [4, 4, 4]).slice(),
      stats: { trans1: [[0,0,0],[0,0,0],[0,0,0]], afterRes: { win: [0,0,0], lose: [0,0,0], draw: [0,0,0] },
               moves: [0, 0, 0], lossN: 0, lossChanged: 0 },
      samples: [] }; // [{x, y}] for replay within the match
  }

  const FDIM = 24;
  function features(st) {
    const x = new Float64Array(FDIM);
    const h = st.pHist, n = h.length;
    for (let k = 0; k < 3; k++) if (n > k) x[k * 3 + h[n - 1 - k]] = 1;
    if (n > 0) { const r = st.results[n - 1];
      x[9 + (r === 'win' ? 0 : r === 'lose' ? 1 : 2)] = 1; }
    for (let m = 0; m < 3; m++) { x[12 + m] = st.pCards[m] / 4; x[15 + m] = st.aCards[m] / 4; }
    x[18] = st.stats.lossN > 0 ? st.stats.lossChanged / st.stats.lossN : 0.5;
    let streak = 0; for (let i = n - 1; i >= 0 && h[i] === h[n - 1]; i--) streak++;
    x[19] = Math.min(streak, 4) / 4;
    if (n > 0) x[20 + st.aHist[n - 1]] = 1; // AI's own last move: catches counter-my-hand players
    x[23] = 1;
    return x;
  }

  function newModel(hn, seed) {
    const rng = mulberry32(seed || 42);
    const g = () => (rng() * 2 - 1); // uniform [-1,1)
    const sh = Math.sqrt(2 / FDIM), so = Math.sqrt(2 / hn);
    const W = { hid_w: [], hid_b: new Array(hn).fill(0), fc_w: [], fc_b: [0, 0, 0] };
    for (let k = 0; k < hn; k++) { const r = new Array(FDIM);
      for (let i = 0; i < FDIM; i++) r[i] = g() * sh; W.hid_w.push(r); }
    for (let o = 0; o < 3; o++) { const r = new Array(hn);
      for (let k = 0; k < hn; k++) r[k] = g() * so; W.fc_w.push(r); }
    return W;
  }

  // prediction renormalized over the cards the player actually has
  function maskRenorm(p, cards) {
    const q = [0, 0, 0]; let s = 0;
    for (let m = 0; m < 3; m++) { q[m] = cards[m] > 0 ? Math.max(p[m], 1e-9) : 0; s += q[m]; }
    if (s <= 0) return [1 / 3, 1 / 3, 1 / 3];
    for (let m = 0; m < 3; m++) q[m] /= s;
    return q;
  }

  function predictFreq(st) { // deku: frequency + card counting only
    const tot = st.stats.moves[0] + st.stats.moves[1] + st.stats.moves[2];
    const p = [0, 0, 0];
    for (let m = 0; m < 3; m++) p[m] = (st.stats.moves[m] + 1) / (tot + 3); // laplace-smoothed
    const q = maskRenorm(p, st.pCards);
    let mv = 0; for (let m = 1; m < 3; m++) if (q[m] > q[mv]) mv = m;
    return { p: q, move: mv, conf: q[mv], nn: false };
  }

  function predictNN(model, st) {
    const x = features(st);
    const pr = headForward(x, model).probs;
    const q = maskRenorm([pr[0], pr[1], pr[2]], st.pCards);
    let mv = 0; for (let m = 1; m < 3; m++) if (q[m] > q[mv]) mv = m;
    return { p: q, move: mv, conf: q[mv], nn: true, x };
  }

  // hp: {lr, steps, replay} ; extraBank: persisted [{x,y}] for laplace
  function learn(model, st, x, actual, hp, extraBank) {
    st.samples.push({ x, y: actual });
    for (let t = 0; t < hp.steps; t++) {
      const bF = [x], bY = [actual];
      const hist = st.samples;
      for (let r = 0; r < Math.min(hp.replay, hist.length - 1); r++) {
        const s = hist[hist.length - 2 - r]; bF.push(s.x); bY.push(s.y);
      }
      if (extraBank && extraBank.length) {
        for (let r = 0; r < Math.min(6, extraBank.length); r++) {
          const s = extraBank[(t * 7 + r * 13) % extraBank.length]; bF.push(s.x); bY.push(s.y);
        }
      }
      headTrainStep(model, bF, bY, hp.lr);
    }
  }

  // honest, human-readable reason for the committed prediction
  function explain(st, pred) {
    if (st.pCards.filter(c => c > 0).length === 1) {
      const only = st.pCards.findIndex(c => c > 0);
      return `残り札が${MOVES[only]}しかない。読みも何もない——詰みだ`;
    }
    const n = st.pHist.length;
    if (n >= 1) {
      const last = st.pHist[n - 1], row = st.stats.trans1[last];
      const tot = row[0] + row[1] + row[2];
      if (tot >= 3) {
        let k = 0; for (let m = 1; m < 3; m++) if (row[m] > row[k]) k = m;
        if (k === pred.move && row[k] / tot >= 0.5)
          return `${MOVES[last]}の後、お前は${Math.round(row[k] / tot * 100)}%で${MOVES[k]}を出している（${tot}回中${row[k]}回）`;
      }
      const r = st.results[n - 1];
      if (r === 'lose' && st.stats.lossN >= 3) {
        const cr = st.stats.lossChanged / st.stats.lossN;
        if (cr >= 0.7 && pred.move !== last) return `負けた直後、お前は${Math.round(cr * 100)}%の確率で手を変える。だから${MOVES[last]}は消えた`;
        if (cr <= 0.3 && pred.move === last) return `負けても手を変えない頑固さ、${Math.round((1 - cr) * 100)}%。読める`;
      }
    }
    const tot = st.stats.moves[0] + st.stats.moves[1] + st.stats.moves[2];
    if (tot >= 4) {
      let k = 0; for (let m = 1; m < 3; m++) if (st.stats.moves[m] > st.stats.moves[k]) k = m;
      if (k === pred.move && st.stats.moves[k] / tot >= 0.45)
        return `ここまでのお前の${MOVES[k]}率は${Math.round(st.stats.moves[k] / tot * 100)}%。数字は嘘をつかない`;
    }
    if (pred.conf < 0.4) return 'まだデータが薄い。……勘だ';
    return `パターン解析の結果だ。言葉にできる癖はまだ隠せているようだな`;
  }

  // ai chooses its card: EV best-response with confidence-scaled mixing
  function aiChoose(st, pred, rng) {
    const avail = [];
    for (let a = 0; a < 3; a++) if (st.aCards[a] > 0) avail.push(a);
    if (avail.length === 1) return avail[0];
    const eps = Math.max(0.05, Math.min(0.35, 0.45 - pred.conf * 0.55));
    if (rng() < eps) { // stock-balancing move
      let best = avail[0];
      for (const a of avail) if (st.aCards[a] > st.aCards[best]) best = a;
      return best;
    }
    let best = avail[0], bestEv = -1e9;
    for (const a of avail) {
      let ev = 0;
      for (let m = 0; m < 3; m++) ev += pred.p[m] * (beats(a, m) ? 1 : beats(m, a) ? -1 : 0);
      ev += st.aCards[a] * 0.02; // tiny tiebreak toward deeper stock
      if (ev > bestEv) { bestEv = ev; best = a; }
    }
    return best;
  }

  function updateAfterRound(st, pMove, aMove) {
    const result = beats(pMove, aMove) ? 'win' : beats(aMove, pMove) ? 'lose' : 'draw';
    const n = st.pHist.length;
    if (n > 0) {
      st.stats.trans1[st.pHist[n - 1]][pMove]++;
      const prevRes = st.results[n - 1];
      st.stats.afterRes[prevRes][pMove]++;
      if (prevRes === 'lose') { st.stats.lossN++; if (pMove !== st.pHist[n - 1]) st.stats.lossChanged++; }
    }
    st.stats.moves[pMove]++;
    st.pHist.push(pMove); st.aHist.push(aMove); st.results.push(result);
    st.pCards[pMove]--; st.aCards[aMove]--;
    return result;
  }

  const BOSSES = {
    deku:    { name: '見習いデク',   hn: 0,  hp: null, persist: false, declareConf: 0.95,
               intro: '「ぼ、ぼくは数を数えるだけです……」' },
    satori:  { name: '中堅サトリ',   hn: 8,  hp: { lr: 0.18, steps: 6, replay: 8 }, persist: false,
               intro: '「フフ……この試合の中で、あなたの癖を暴いてみせますよ」' },
    laplace: { name: '悪魔ラプラス', hn: 16, hp: { lr: 0.15, steps: 7, replay: 12 }, persist: true,
               intro: '「私はお前との全対戦を覚えている。逃げ場はない」' }
  };

  // ヨミキリ宣言: the AI publicly doubles down only on a real read —
  // needs data (round >= 4) and boss-specific confidence, else it keeps quiet.
  const DECLARE_CONF = 0.65;
  const DECLARE_MIN_ROUND = 3; // 0-indexed: from the 4th round
  function shouldDeclare(st, pred, boss) {
    const th = boss && boss.declareConf != null ? boss.declareConf : DECLARE_CONF;
    return st.pHist.length >= DECLARE_MIN_ROUND && pred.conf >= th;
  }

  const api = { MOVES, EMO, FDIM, DECLARE_CONF, DECLARE_MIN_ROUND, shouldDeclare,
    beats, counter, mulberry32, newMatchState, randomDeal, features,
    newModel, predictFreq, predictNN, learn, explain, aiChoose, updateAfterRound, BOSSES };
  if (typeof module !== 'undefined') module.exports = api;
  else root.YomiAI = api;
})(typeof window !== 'undefined' ? window : globalThis);
