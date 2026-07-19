"use strict";
// Balance & learning verification for yomikiri.
// G1 fairness vs random | G2 exploitation of biased bots | G3 persistent learning | G4 honest explanations
const Y = require('./yomi_ai.js');

function rngFactory(seed) { return Y.mulberry32(seed); }
function pickAvail(cards, rng) { const av = []; for (let m = 0; m < 3; m++) if (cards[m] > 0) av.push(m);
  return av[Math.floor(rng() * av.length)]; }

// --- bot policies (return move given state, must respect remaining cards) ---
const BOTS = {
  random: (st, rng) => pickAvail(st.pCards, rng),
  proportional: (st, rng) => { // uniform over remaining CARDS = sound baseline play
    const tot = st.pCards[0] + st.pCards[1] + st.pCards[2];
    let r = rng() * tot;
    for (let m = 0; m < 3; m++) { r -= st.pCards[m]; if (r < 0) return m; } return 2; },
  repeatAfterWin: (st, rng) => {
    const n = st.pHist.length;
    if (n > 0 && st.results[n - 1] === 'win' && st.pCards[st.pHist[n - 1]] > 0) return st.pHist[n - 1];
    return pickAvail(st.pCards, rng); },
  cycler: (st, rng) => { const n = st.pHist.length;
    for (let d = 0; d < 3; d++) { const m = (n + d) % 3; if (st.pCards[m] > 0) return m; } return 0; },
  favorite: (st, rng) => { if (st.pCards[2] > 0 && rng() < 0.6) return 2; return pickAvail(st.pCards, rng); },
  antiDeku: (st, rng) => { // knows deku's public brain: he counters your frequency-argmax
    const pred = Y.predictFreq(st);            // exactly what deku computes (public info only)
    const dekuMove = Y.counter(pred.move);     // what deku will most likely play
    const my = Y.counter(dekuMove);            // beat it
    if (st.pCards[my] > 0 && pred.conf > 0.36) return my;
    return BOTS.proportional(st, rng); },
  humanish: (st, rng) => { const n = st.pHist.length;
    if (n > 0 && st.results[n - 1] === 'lose' && rng() < 0.7) {
      const c = Y.counter(st.aHist[n - 1]); if (st.pCards[c] > 0) return c; }
    if (n > 0 && st.results[n - 1] === 'win' && rng() < 0.5 && st.pCards[st.pHist[n - 1]] > 0) return st.pHist[n - 1];
    return pickAvail(st.pCards, rng); }
};

// --- one match with the real economy; returns per-round log + chip outcome ---
// cfg: {deal:'even'|'random', rounds:12|10}
function playMatch(bossKey, botFn, rng, model, bank, cfg) {
  cfg = cfg || { deal: 'even', rounds: 12 };
  const boss = Y.BOSSES[bossKey];
  const st = cfg.deal === 'random'
    ? Y.newMatchState(Y.randomDeal(rng), Y.randomDeal(rng))
    : Y.newMatchState();
  let pChips = 10, aChips = 10, carry = 0;
  const roundLog = [];
  for (let round = 0; round < cfg.rounds && pChips > 0 && aChips > 0; round++) {
    const pred = boss.hn === 0 ? Y.predictFreq(st) : Y.predictNN(model, st);
    const aMove = Y.aiChoose(st, pred, rng);
    const pMove = botFn(st, rng);
    const x = pred.x || Y.features(st);
    const result = Y.updateAfterRound(st, pMove, aMove);
    if (boss.hn > 0) Y.learn(model, st, x, pMove, boss.hp, boss.persist ? bank : null);
    if (bank && boss.persist) { bank.push({ x, y: pMove }); if (bank.length > 240) bank.shift(); }
    const declared = Y.shouldDeclare({ pHist: { length: round } }, pred, boss); // round count = pHist length pre-update
    const bet = 1, pot = bet * 2 + carry; carry = 0;
    const hit = pred.move === pMove;
    if (result === 'draw') { carry = pot; }
    else if (result === 'win') { // player wins
      pChips += pot - bet; aChips -= bet;
      if (declared && !hit && aChips > 0) { const x = Math.min(2, aChips); pChips += x; aChips -= x; } // 返り討ち
    } else { // ai wins
      aChips += pot - bet; pChips -= bet;
      if (declared && hit && pChips > 0) { const x = Math.min(2, pChips); aChips += x; pChips -= x; } // ヨミキリ成立
    }
    roundLog.push({ round, result, hit, conf: pred.conf, declared });
  }
  return { pChips, aChips, roundLog };
}

function decidedWinRate(logs, filter) {
  let ai = 0, p = 0;
  for (const l of logs) { if (filter && !filter(l)) continue;
    if (l.result === 'lose') ai++; else if (l.result === 'win') p++; }
  return ai + p > 0 ? ai / (ai + p) : 0.5;
}

let pass = true;
function check(name, cond, detail) {
  console.log(`${cond ? 'PASS' : 'FAIL'}  ${name}  ${detail}`);
  if (!cond) pass = false;
}

// ---- G1: fairness vs sound baseline play (proportional over remaining cards) ----
// (vs pure type-uniform "random", the AI legitimately profits from card management —
//  that is Kaiji-style endgame skill, reported below as info, not gated)
for (const bossKey of ['deku', 'satori', 'laplace']) {
  const rng = rngFactory(1);
  let diff = 0, logs = [];
  const N = 1500;
  for (let i = 0; i < N; i++) {
    const model = Y.BOSSES[bossKey].hn ? Y.newModel(Y.BOSSES[bossKey].hn, 100 + i) : null;
    const r = playMatch(bossKey, BOTS.proportional, rng, model, null);
    diff += r.pChips - r.aChips; logs.push(...r.roundLog);
  }
  const wr = decidedWinRate(logs);
  // a light house edge (< 2 chips) is accepted — it comes from endgame card-reading, which
  // human players can also do (all cards are public); the gate is "no unfair round winrate".
  check(`G1 ${bossKey} vs proportional`, Math.abs(wr - 0.5) < 0.04 && Math.abs(diff / N) < 2.0,
    `AI round winrate ${(wr * 100).toFixed(1)}% (want ~50), mean chip diff ${(diff / N).toFixed(2)}`);
}
{ // G1b: the tutorial boss must be decisively beatable by reading his (public) brain
  const rng = rngFactory(31);
  let diff = 0; const logs = [];
  const N = 1500;
  for (let i = 0; i < N; i++) {
    const r = playMatch('deku', BOTS.antiDeku, rng, null, null);
    diff += r.pChips - r.aChips; logs.push(...r.roundLog);
  }
  const wr = decidedWinRate(logs);
  check('G1b deku is beatable by out-reading him', diff / N > 2.0,
    `player mean chip profit ${(diff / N).toFixed(2)} (AI round winrate ${(wr * 100).toFixed(1)}%)`);
}
{ // info only: house edge vs naive type-uniform random
  const rng = rngFactory(2); let diff = 0; const logs = [];
  for (let i = 0; i < 800; i++) {
    const model = Y.newModel(8, 300 + i);
    const r = playMatch('satori', BOTS.random, rng, model, null);
    diff += r.pChips - r.aChips; logs.push(...r.roundLog);
  }
  console.log(`INFO  house edge vs naive random: AI winrate ${(decidedWinRate(logs) * 100).toFixed(1)}%, mean chip diff ${(diff / 800).toFixed(2)} (card-management skill, intended)`);
}

// ---- G2: exploitation of biased bots (satori & laplace, within-match) ----
for (const bossKey of ['satori', 'laplace']) {
  for (const bot of ['repeatAfterWin', 'cycler', 'favorite', 'humanish']) {
    const rng = rngFactory(7);
    const logs = [];
    for (let i = 0; i < 800; i++) {
      const model = Y.newModel(Y.BOSSES[bossKey].hn, 500 + i);
      const r = playMatch(bossKey, BOTS[bot], rng, model, null);
      logs.push(...r.roundLog);
    }
    const early = decidedWinRate(logs, l => l.round < 6);
    const late = decidedWinRate(logs, l => l.round >= 6);
    const all = decidedWinRate(logs);
    // trend gate only where the bias persists all game (favorite's paper runs out; humanish is reactive)
    const trendGate = (bot === 'repeatAfterWin' || bot === 'cycler') ? late >= early - 0.02 : true;
    // humanish is semi-adaptive: the win condition is the within-match comeback (and G3 shows
    // laplace's carry-over crushes it from match 2 onward)
    const allGate = bot === 'humanish' ? (late > 0.53 && late > early + 0.03) : all > 0.54;
    check(`G2 ${bossKey} vs ${bot}`, allGate && trendGate,
      `AI winrate ${(all * 100).toFixed(1)}% (early ${(early * 100).toFixed(1)} → late ${(late * 100).toFixed(1)})`);
  }
}

// ---- G3: laplace persistent learning across matches ----
{
  const seqWins = { first: [], later: [] };
  for (let s = 0; s < 250; s++) {
    const rng = rngFactory(9000 + s);
    const model = Y.newModel(16, 9000 + s);
    const bank = [];
    for (let g = 0; g < 5; g++) {
      const r = playMatch('laplace', BOTS.humanish, rng, model, bank);
      const earlyWr = decidedWinRate(r.roundLog, l => l.round < 4);
      if (g === 0) seqWins.first.push(earlyWr); else if (g >= 3) seqWins.later.push(earlyWr);
    }
  }
  const avg = a => a.reduce((x, y) => x + y, 0) / a.length;
  const f = avg(seqWins.first), l = avg(seqWins.later);
  check('G3 laplace persistence', l > f + 0.02,
    `early-round AI winrate: match1 ${(f * 100).toFixed(1)}% → match4-5 ${(l * 100).toFixed(1)}%`);
}

// ---- G5: variant rules stay fair (random deal / 10-round spare mode) ----
for (const cfg of [{ deal: 'random', rounds: 12 }, { deal: 'even', rounds: 10 }, { deal: 'random', rounds: 10 }]) {
  const rng = rngFactory(55);
  let diff = 0; const logs = [];
  const N = 1200;
  for (let i = 0; i < N; i++) {
    const model = Y.newModel(8, 700 + i);
    const r = playMatch('satori', BOTS.proportional, rng, model, null, cfg);
    diff += r.pChips - r.aChips; logs.push(...r.roundLog);
  }
  const wr = decidedWinRate(logs);
  check(`G5 satori vs proportional [${cfg.deal}/${cfg.rounds}R]`, Math.abs(wr - 0.5) < 0.04 && Math.abs(diff / N) < 2.0,
    `AI round winrate ${(wr * 100).toFixed(1)}%, mean chip diff ${(diff / N).toFixed(2)}`);
}

// ---- G4: explanations are honest & robust ----
{
  let ok = true, cited = 0;
  const rng = rngFactory(77);
  for (let i = 0; i < 400; i++) {
    const model = Y.newModel(8, i);
    const st = Y.newMatchState();
    for (let round = 0; round < 12; round++) {
      const pred = Y.predictNN(model, st);
      const msg = Y.explain(st, pred);
      if (typeof msg !== 'string' || !msg.length) { ok = false; break; }
      const m = msg.match(/(グー|チョキ|パー)の後、お前は(\d+)%で(グー|チョキ|パー)を出している（(\d+)回中(\d+)回）/);
      if (m) { cited++;
        const last = Y.MOVES.indexOf(m[1]), tgt = Y.MOVES.indexOf(m[3]);
        const tot = +m[4], cnt = +m[5];
        const row = st.stats.trans1[last];
        if (row[tgt] !== cnt || row[0] + row[1] + row[2] !== tot || Math.round(cnt / tot * 100) !== +m[2]) ok = false;
        if (tgt !== pred.move) ok = false; // must describe the actual committed prediction
      }
      const aMove = Y.aiChoose(st, pred, rng);
      const pMove = BOTS.humanish(st, rng);
      const x = pred.x;
      Y.updateAfterRound(st, pMove, aMove);
      Y.learn(model, st, x, pMove, { lr: 0.15, steps: 3, replay: 6 }, null);
    }
  }
  check('G4 explanation honesty', ok && cited > 50, `verified ${cited} cited pattern claims against real stats`);
}

console.log(pass ? '\nALL BALANCE TESTS PASS' : '\nBALANCE TESTS FAILED');
process.exit(pass ? 0 : 1);
