/* ───────────────────────── NUMA — in-browser EEG pipeline ─────────────────────────
 * Parses a BioSemi .bdf and reproduces (as faithfully as practical in JS) the v6
 * training preprocessing, then emits the tensors the ONNX model expects.
 *
 *   parseBDF(arrayBuffer)  → { labels, units, fs[], data[] (physical, per signal) }
 *   buildInputs(arrayBuffer, meta) → { windows, bandpower, mask, N, C, T, info }
 *
 * NOTE: this is an APPROXIMATION of MNE's pipeline (IIR zero-phase filters vs MNE's
 * FIR; linear-interp resample vs FFT resample). Band power and the model itself are
 * exact. Predictions may differ slightly from the Python reference.
 */
window.NumaEEG = (() => {
  "use strict";

  const AUX_PREFIX = ["EXG", "GSR", "STATUS", "TRIG", "AIO", "ERG", "RESP", "PLET", "TEMP"];

  // ── BDF (BioSemi 24-bit) reader ─────────────────────────────────────────────
  function parseBDF(buf) {
    const dv = new DataView(buf);
    const td = new TextDecoder("ascii");
    const str = (off, len) => td.decode(new Uint8Array(buf, off, len)).trim();

    // identification byte 0 == 255 for BDF
    if (dv.getUint8(0) !== 255) throw new Error("Not a BDF file (bad identification byte).");
    const nRecords = parseInt(str(236, 8), 10);
    const recDur   = parseFloat(str(244, 8));
    const ns       = parseInt(str(252, 4), 10);
    if (!ns || ns < 1) throw new Error("BDF header reports no signals.");

    let o = 256;
    const labels = []; for (let i = 0; i < ns; i++) labels.push(str(o + i * 16, 16)); o += ns * 16;
    o += ns * 80;                                   // transducer
    const units = []; for (let i = 0; i < ns; i++) units.push(str(o + i * 8, 8)); o += ns * 8;
    const physMin = []; for (let i = 0; i < ns; i++) physMin.push(parseFloat(str(o + i * 8, 8))); o += ns * 8;
    const physMax = []; for (let i = 0; i < ns; i++) physMax.push(parseFloat(str(o + i * 8, 8))); o += ns * 8;
    const digMin = []; for (let i = 0; i < ns; i++) digMin.push(parseFloat(str(o + i * 8, 8))); o += ns * 8;
    const digMax = []; for (let i = 0; i < ns; i++) digMax.push(parseFloat(str(o + i * 8, 8))); o += ns * 8;
    o += ns * 80;                                   // prefiltering
    const nSamp = []; for (let i = 0; i < ns; i++) nSamp.push(parseInt(str(o + i * 8, 8), 10)); o += ns * 8;
    o += ns * 32;                                   // reserved

    const fs = nSamp.map((n) => n / recDur);
    const gain = physMax.map((pmax, i) => (pmax - physMin[i]) / (digMax[i] - digMin[i]));
    const data = nSamp.map((n) => new Float32Array(n * nRecords));

    // data records: per record, per signal, nSamp[s] little-endian 24-bit samples
    let p = o;
    for (let r = 0; r < nRecords; r++) {
      for (let s = 0; s < ns; s++) {
        const n = nSamp[s], out = data[s], base = r * n, g = gain[s], pmin = physMin[s], dmin = digMin[s];
        for (let k = 0; k < n; k++) {
          let v = dv.getUint8(p) | (dv.getUint8(p + 1) << 8) | (dv.getUint8(p + 2) << 16);
          if (v & 0x800000) v -= 0x1000000;        // 24-bit two's complement
          out[base + k] = (v - dmin) * g + pmin;   // → physical units
          p += 3;
        }
      }
    }
    return { labels, units, fs, data, nRecords, recDur };
  }

  // ── DSP: zero-phase biquad (RBJ) filters ────────────────────────────────────
  function biquad(type, fc, fs, Q) {
    const w0 = 2 * Math.PI * fc / fs, c = Math.cos(w0), s = Math.sin(w0), al = s / (2 * Q);
    let b0, b1, b2, a0, a1, a2;
    if (type === "lp")      { b0 = (1 - c) / 2; b1 = 1 - c;  b2 = (1 - c) / 2; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; }
    else if (type === "hp") { b0 = (1 + c) / 2; b1 = -(1 + c); b2 = (1 + c) / 2; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; }
    else /* notch */        { b0 = 1; b1 = -2 * c; b2 = 1; a0 = 1 + al; a1 = -2 * c; a2 = 1 - al; }
    return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
  }
  function applyBiquad(x, [b0, b1, b2, a1, a2]) {
    const y = new Float32Array(x.length);
    let x1 = 0, x2 = 0, y1 = 0, y2 = 0;
    for (let i = 0; i < x.length; i++) {
      const xi = x[i], yi = b0 * xi + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      x2 = x1; x1 = xi; y2 = y1; y1 = yi; y[i] = yi;
    }
    return y;
  }
  function reverse(x) { const y = new Float32Array(x.length); for (let i = 0; i < x.length; i++) y[i] = x[x.length - 1 - i]; return y; }
  function filtfilt(x, coef) { return reverse(applyBiquad(reverse(applyBiquad(x, coef)), coef)); }  // zero-phase

  function bandpassNotch(x, fs) {
    let y = filtfilt(x, biquad("hp", 1.0, fs, 0.7071));   // 1 Hz high-pass
    y = filtfilt(y, biquad("lp", 40.0, fs, 0.7071));      // 40 Hz low-pass
    y = filtfilt(y, biquad("notch", 50.0, fs, 30));       // 50 Hz notch
    return y;
  }

  // resample to 256 Hz by linear interpolation (safe: signal is band-limited <40 Hz)
  function resampleTo(x, fsIn, fsOut) {
    if (Math.abs(fsIn - fsOut) < 1e-6) return x;
    const ratio = fsIn / fsOut, Lout = Math.floor(x.length / ratio);
    const y = new Float32Array(Lout);
    for (let i = 0; i < Lout; i++) {
      const t = i * ratio, j = Math.floor(t), f = t - j;
      y[i] = j + 1 < x.length ? x[j] * (1 - f) + x[j + 1] * f : x[j];
    }
    return y;
  }

  // ── radix-2 FFT (n = 256) for band power ────────────────────────────────────
  function fftPower(re) {
    const n = re.length, im = new Float32Array(n);
    for (let i = 1, j = 0; i < n; i++) {            // bit reversal
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { const tr = re[i]; re[i] = re[j]; re[j] = tr; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cr = 1, ci = 0;
        for (let k = 0; k < len / 2; k++) {
          const a = i + k, b = i + k + len / 2;
          const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
          re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
          const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr;
        }
      }
    }
    const half = n / 2 + 1, pw = new Float32Array(half);
    for (let k = 0; k < half; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
    return pw;                                       // power, bins 0..n/2
  }

  // ── full pipeline → ONNX inputs ─────────────────────────────────────────────
  function buildInputs(buf, meta) {
    const C = meta.n_channels, T = meta.n_times, nb = meta.n_bands || 5;
    const fsOut = meta.sample_rate || 256;
    const bands = meta.bands || [[1, 4], [4, 8], [8, 13], [13, 30], [30, 40]];
    const thr = meta.artifact_thresh_v || 150e-6;
    const cap = meta.eval_max_windows || 96;

    const bdf = parseBDF(buf);

    // pick EEG channels (drop aux), preserve file order to match MNE
    const eegIdx = [];
    bdf.labels.forEach((lab, i) => {
      const L = lab.toUpperCase();
      if (!AUX_PREFIX.some((px) => L.startsWith(px))) eegIdx.push(i);
    });
    if (eegIdx.length !== C) {
      throw new Error(`Channel mismatch: model expects ${C} EEG channels, file has ${eegIdx.length} after dropping auxiliaries.`);
    }

    // unit → volts (BioSemi BDF is typically µV; MNE works in volts)
    const unitScale = (u) => {
      const s = (u || "").toLowerCase();
      if (s.includes("µv") || s.includes("uv")) return 1e-6;
      if (s.includes("mv")) return 1e-3;
      if (s === "v") return 1;
      return 1e-6;                                   // default assume µV
    };

    // filter → resample → store as channel-major matrix at fsOut
    const chans = eegIdx.map((si) => {
      const sc = unitScale(bdf.units[si]);
      let x = bdf.data[si];
      if (sc !== 1) { const y = new Float32Array(x.length); for (let i = 0; i < x.length; i++) y[i] = x[i] * sc; x = y; }
      x = bandpassNotch(x, bdf.fs[si]);
      return resampleTo(x, bdf.fs[si], fsOut);
    });
    const L = Math.min(...chans.map((c) => c.length));

    // common-average reference (subtract mean across channels per timepoint)
    for (let t = 0; t < L; t++) {
      let m = 0; for (let c = 0; c < C; c++) m += chans[c][t]; m /= C;
      for (let c = 0; c < C; c++) chans[c][t] -= m;
    }

    // window (non-overlapping T), artifact-reject, baseline, z-score
    const winList = [];
    const bs = Math.max(1, Math.floor(T * 0.2));
    for (let start = 0; start + T <= L; start += T) {
      // artifact: any channel peak-to-peak over threshold → drop window
      let bad = false;
      for (let c = 0; c < C && !bad; c++) {
        let mn = Infinity, mx = -Infinity;
        for (let k = 0; k < T; k++) { const v = chans[c][start + k]; if (v < mn) mn = v; if (v > mx) mx = v; }
        if (mx - mn > thr) bad = true;
      }
      if (bad) continue;
      const w = new Float32Array(C * T);             // (C, T) z-scored
      for (let c = 0; c < C; c++) {
        let base = 0; for (let k = 0; k < bs; k++) base += chans[c][start + k]; base /= bs;
        let mean = 0; for (let k = 0; k < T; k++) mean += (chans[c][start + k] - base); mean /= T;
        let varr = 0;
        for (let k = 0; k < T; k++) { const d = (chans[c][start + k] - base) - mean; varr += d * d; }
        const sd = Math.sqrt(varr / T) + 1e-8;
        for (let k = 0; k < T; k++) w[c * T + k] = ((chans[c][start + k] - base) - mean) / sd;
      }
      winList.push(w);
    }
    if (winList.length === 0) throw new Error("No clean EEG windows (file too short or all windows rejected as artifacts).");

    // cap to eval_max with even spacing (matches v6 eval)
    let chosen = winList;
    if (winList.length > cap) {
      chosen = [];
      for (let i = 0; i < cap; i++) chosen.push(winList[Math.round(i * (winList.length - 1) / (cap - 1))]);
    }
    const N = chosen.length;

    // band-power per window/channel (exact match to the torch BandPower module)
    const bandBins = bands.map(([lo, hi]) => {
      const idx = []; for (let k = 0; k <= T / 2; k++) if (k >= lo && k < hi) idx.push(k); return idx;
    });
    const windows = new Float32Array(N * C * T);
    const bandpower = new Float32Array(N * C * nb);
    for (let w = 0; w < N; w++) {
      const src = chosen[w];
      windows.set(src, w * C * T);
      for (let c = 0; c < C; c++) {
        const seg = new Float32Array(T);
        for (let k = 0; k < T; k++) seg[k] = src[c * T + k];
        const pw = fftPower(seg);
        for (let b = 0; b < nb; b++) {
          let sum = 0; for (const k of bandBins[b]) sum += pw[k];
          bandpower[w * C * nb + c * nb + b] = Math.log1p(sum);
        }
      }
    }
    const mask = new Float32Array(N).fill(1);
    return { windows, bandpower, mask, N, C, T, info: { totalWindows: winList.length, used: N } };
  }

  return { parseBDF, buildInputs };
})();
