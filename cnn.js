"use strict";
// damashi-ai inference engine: conv-relu-pool ×2 -> fc -> softmax
// Shared by index.html (browser) and test_parity.js (node).
function conv2d(inp, W, B) { // inp: [C][H][W], W: [O][C][3][3] -> relu'd output
  const C = inp.length, H = inp[0].length, Wd = inp[0][0].length;
  const O = W.length, oh = H - 2, ow = Wd - 2;
  const out = [];
  for (let o = 0; o < O; o++) {
    const m = [];
    for (let y = 0; y < oh; y++) {
      const row = new Float32Array(ow);
      for (let x = 0; x < ow; x++) {
        let s = B[o];
        for (let c = 0; c < C; c++) {
          const ic = inp[c], k = W[o][c];
          for (let ky = 0; ky < 3; ky++) {
            const ir = ic[y + ky], kr = k[ky];
            s += ir[x] * kr[0] + ir[x + 1] * kr[1] + ir[x + 2] * kr[2];
          }
        }
        row[x] = s > 0 ? s : 0; // relu
      }
      m.push(row);
    }
    out.push(m);
  }
  return out;
}
function maxpool2(inp) {
  return inp.map(ch => {
    const oh = Math.floor(ch.length / 2), ow = Math.floor(ch[0].length / 2), m = [];
    for (let y = 0; y < oh; y++) {
      const row = new Float32Array(ow);
      for (let x = 0; x < ow; x++) {
        row[x] = Math.max(ch[2*y][2*x], ch[2*y][2*x+1], ch[2*y+1][2*x], ch[2*y+1][2*x+1]);
      }
      m.push(row);
    }
    return m;
  });
}
function forward(img28, W) { // img28: [28][28] floats 0..1
  const a1 = conv2d([img28], W.conv1_w, W.conv1_b); // C1×26×26
  const p1 = maxpool2(a1);                          // C1×13×13
  const a2 = conv2d(p1, W.conv2_w, W.conv2_b);      // C2×11×11
  const p2 = maxpool2(a2);                          // C2×5×5
  const C2 = p2.length;
  let flat = new Float32Array(C2 * 25);
  for (let c = 0; c < C2; c++) for (let y = 0; y < 5; y++) for (let x = 0; x < 5; x++)
    flat[c * 25 + y * 5 + x] = p2[c][y][x];
  const feat = flat; // conv features (pre-hidden), input to the trainable head
  if (W.hid_w) { // optional hidden fc + relu
    const h = new Float32Array(W.hid_b.length);
    for (let k = 0; k < h.length; k++) {
      let s = W.hid_b[k]; const w = W.hid_w[k];
      for (let i = 0; i < flat.length; i++) s += w[i] * flat[i];
      h[k] = s > 0 ? s : 0;
    }
    flat = h;
  }
  const logits = W.fc_b.map((b, k) => {
    let s = b; const w = W.fc_w[k];
    for (let i = 0; i < flat.length; i++) s += w[i] * flat[i];
    return s;
  });
  const mx = Math.max(...logits);
  const ex = logits.map(v => Math.exp(v - mx));
  const sum = ex.reduce((a, b) => a + b, 0);
  return { probs: ex.map(v => v / sum), a1, a2, p2, feat };
}

// ---- v3 revenge system: online learning of the fc head (conv frozen) ----
// Hyperparameters mirrored in train_v3.py; change both or parity breaks.
const LEARN_HP = { steps: 40, lr: 0.04, batch: 32, advRepeat: 8, lambda: 0.02 };

function headForward(f, W) { // f: length-800 array -> {hpre, h, probs}
  const HN = W.hid_b.length, ON = W.fc_b.length;
  const hpre = new Float64Array(HN), h = new Float64Array(HN);
  for (let k = 0; k < HN; k++) {
    let s = W.hid_b[k]; const w = W.hid_w[k];
    for (let i = 0; i < f.length; i++) s += w[i] * f[i];
    hpre[k] = s; h[k] = s > 0 ? s : 0;
  }
  const logits = new Float64Array(ON);
  for (let o = 0; o < ON; o++) {
    let s = W.fc_b[o]; const w = W.fc_w[o];
    for (let k = 0; k < HN; k++) s += w[k] * h[k];
    logits[o] = s;
  }
  let mx = -Infinity;
  for (let o = 0; o < ON; o++) if (logits[o] > mx) mx = logits[o];
  let sum = 0; const probs = new Float64Array(ON);
  for (let o = 0; o < ON; o++) { probs[o] = Math.exp(logits[o] - mx); sum += probs[o]; }
  for (let o = 0; o < ON; o++) probs[o] /= sum;
  return { hpre, h, probs };
}

// anchor (optional): pristine weights + lambda. L2-pulls the head toward its original
// function so accuracy drift stays bounded no matter how many tricks it learns.
function headTrainStep(W, batchF, batchY, lr, anchor) { // one SGD step, returns mean CE loss
  const B = batchF.length, HN = W.hid_b.length, FN = batchF[0].length, ON = W.fc_b.length;
  const gHw = []; for (let k = 0; k < HN; k++) gHw.push(new Float64Array(FN));
  const gHb = new Float64Array(HN);
  const gFw = []; for (let o = 0; o < ON; o++) gFw.push(new Float64Array(HN));
  const gFb = new Float64Array(ON);
  let loss = 0;
  for (let s = 0; s < B; s++) {
    const f = batchF[s], y = batchY[s];
    const { hpre, h, probs } = headForward(f, W);
    loss += -Math.log(Math.max(probs[y], 1e-300));
    const dl = probs; dl[y] -= 1; // dLoss/dlogits (per sample, before 1/B)
    const dh = new Float64Array(HN);
    for (let o = 0; o < ON; o++) {
      const g = dl[o]; gFb[o] += g;
      const gw = gFw[o], wr = W.fc_w[o];
      for (let k = 0; k < HN; k++) { gw[k] += g * h[k]; dh[k] += wr[k] * g; }
    }
    for (let k = 0; k < HN; k++) {
      if (hpre[k] <= 0) continue;
      const g = dh[k]; gHb[k] += g;
      const gw = gHw[k];
      for (let i = 0; i < FN; i++) gw[i] += g * f[i];
    }
  }
  const sc = lr / B, la = anchor ? lr * anchor.lambda : 0;
  for (let k = 0; k < HN; k++) {
    W.hid_b[k] -= sc * gHb[k] + (la ? la * (W.hid_b[k] - anchor.W0.hid_b[k]) : 0);
    const w = W.hid_w[k], g = gHw[k], a0 = la ? anchor.W0.hid_w[k] : null;
    for (let i = 0; i < FN; i++) w[i] -= sc * g[i] + (la ? la * (w[i] - a0[i]) : 0);
  }
  for (let o = 0; o < ON; o++) {
    W.fc_b[o] -= sc * gFb[o] + (la ? la * (W.fc_b[o] - anchor.W0.fc_b[o]) : 0);
    const w = W.fc_w[o], g = gFw[o], a0 = la ? anchor.W0.fc_w[o] : null;
    for (let k = 0; k < HN; k++) w[k] -= sc * g[k] + (la ? la * (w[k] - a0[k]) : 0);
  }
  return loss / B;
}

// Incremental trainer so the browser can animate per-step (rAF) and node can loop.
// replay: REPLAY object (int-quantized). history: [{f, y}] previously learned fools.
function makeTrainer(W, advF, advY, replay, history, opts) {
  const hp = LEARN_HP, scale = replay.scale || 1;
  const earlyStop = opts && opts.earlyStop; // stop once the trick is countered (game mode)
  const anchor = (opts && opts.anchorW0) ? { W0: opts.anchorW0, lambda: (opts.lambda != null ? opts.lambda : hp.lambda) } : null;
  const rf = replay.f.map(row => { const a = new Float64Array(row.length);
    for (let i = 0; i < row.length; i++) a[i] = row[i] / scale; return a; });
  const ry = replay.y;
  const hist = history || [];
  let t = 0, lastAdvP = 0;
  return {
    hp,
    step() {
      const bF = [], bY = [];
      for (let j = 0; j < hp.batch; j++) {
        const i = (t * hp.batch + j) % ry.length;
        bF.push(rf[i]); bY.push(ry[i]);
      }
      for (let r = 0; r < hp.advRepeat; r++) { bF.push(advF); bY.push(advY); }
      for (let r = 0; r < Math.min(hist.length, 8); r++) { // keep old tricks fresh
        const hsample = hist[(t + r) % hist.length];
        bF.push(hsample.f); bY.push(hsample.y);
      }
      const loss = headTrainStep(W, bF, bY, hp.lr, anchor);
      t++;
      const advProbs = headForward(advF, W).probs;
      lastAdvP = advProbs[advY];
      return { t, loss, advProbs };
    },
    done() {
      if (t >= hp.steps) return true;
      return !!(earlyStop && t >= 10 && lastAdvP > 0.95);
    },
    consolidate(k) { // replay-only cool-down: pulls clean accuracy back up
      for (let c = 0; c < (k || 5); c++) {
        const bF = [], bY = [];
        for (let j = 0; j < hp.batch; j++) {
          const i = ((t + c) * hp.batch + j) % ry.length;
          bF.push(rf[i]); bY.push(ry[i]);
        }
        headTrainStep(W, bF, bY, hp.lr, anchor);
      }
    }
  };
}

function headAccuracy(W, featRows, ys, scale) { // eval on quantized feature rows
  let ok = 0;
  const f = new Float64Array(featRows[0].length);
  for (let s = 0; s < featRows.length; s++) {
    const row = featRows[s];
    for (let i = 0; i < row.length; i++) f[i] = row[i] / (scale || 1);
    const p = headForward(f, W).probs;
    let am = 0; for (let o = 1; o < p.length; o++) if (p[o] > p[am]) am = o;
    if (am === ys[s]) ok++;
  }
  return ok / featRows.length;
}
function cam(p2, cls, W) { // class activation map on 5×5
  const m = [];
  for (let y = 0; y < 5; y++) { const r = new Float32Array(5);
    for (let x = 0; x < 5; x++) { let s = 0;
      for (let c = 0; c < 16; c++) s += W.fc_w[cls][c*25 + y*5 + x] * p2[c][y][x];
      r[x] = s; } m.push(r); }
  return m;
}
if (typeof module !== 'undefined') module.exports = { conv2d, maxpool2, forward, cam,
  LEARN_HP, headForward, headTrainStep, makeTrainer, headAccuracy };
