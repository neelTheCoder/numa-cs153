# NUMA · Frontier Systems

A minimalist website for **non-invasive semantic neural decoding**: upload a single-instance
`.bdf` EEG recording and the NUMA v6 model guesses which of 36 imagined animals/tools it is,
then pops up the matching picture. Includes the project's mission, v6 cross-validation results,
and a walkthrough of how the decoder works.

**Inference runs entirely in the browser** via [onnxruntime-web](https://onnxruntime.ai/) — no
backend, no server, no upload limits. The `.bdf` is parsed, filtered, and classified on the
visitor's own device, so it works as a pure static site on Vercel and the file never leaves the
browser.

```
.
├── index.html        # the site
├── styles.css        # styling (minimalist, dark)
├── eeg.js            # in-browser BDF parser + preprocessing + band power
├── app.js            # orchestration: file → eeg.js → ONNX → image popup
├── manifest.json     # class → picture mapping (auto-generated from /pictures)
├── numa_v6.onnx      # ⬅ YOU ADD THIS (from export_onnx.py) — the model weights
├── model_meta.json   # ⬅ YOU ADD THIS (from export_onnx.py) — channels, classes, etc.
├── vercel.json       # static config
├── pictures/         # animals/ + tools/ images shown on prediction
└── plots/            # v6 result figures
```

## One-time setup: export the model

The trained weights live on the GPU box (`enigma-train`), not in this repo. Convert a v6
checkpoint to ONNX **once**, using `export_onnx.py` from the `numa_model` repo:

```bash
# on the machine that has the checkpoint (e.g. enigma-train):
pip install torch onnx onnxscript
python export_onnx.py --ckpt enigma_v6_fold2.pt --out numa_v6.onnx
#   → produces numa_v6.onnx  and  model_meta.json
#   (the script prints a "max|onnx - torch|" check ~1e-5 to confirm the export is faithful)
```

Copy both files into this repo's root, commit, and push:

```bash
cp numa_v6.onnx model_meta.json /path/to/numa-frontier-systems/
git add numa_v6.onnx model_meta.json && git commit -m "add model" && git push
```

Until those two files exist, the page loads fully but the upload card shows a "model not loaded"
notice.

## Deploy (Vercel)

1. Push this folder to GitHub (already wired to `numa-frontier-systems`).
2. Vercel → **New Project** → import the repo. Framework preset: **Other** (no build command).
   Deploy. Vercel serves the static files — including `numa_v6.onnx` — directly.

That's the whole deployment. No API, no environment variables.

## How a prediction works (in the browser)

1. `eeg.js` parses the BioSemi `.bdf` (24-bit), drops non-EEG channels.
2. Reproduces the training pipeline: 1–40 Hz band-pass + 50 Hz notch (zero-phase), resample to
   256 Hz, common-average reference, 1-second windows, artifact rejection, baseline + z-score.
3. Computes per-channel log band power (δ θ α β γ) — exactly matching the model.
4. `onnxruntime-web` runs the recording-level attention model over all windows → top-5 concepts.
5. The matching picture pops up.

## Notes & caveats

- **Approximate preprocessing.** The in-browser DSP uses zero-phase IIR filters and linear-interp
  resampling, which approximate MNE's FIR/FFT pipeline used in training. The band-power features
  and the model itself are exact, but predictions may differ slightly from the Python reference.
- **First load** downloads the ONNX runtime (~10 MB of wasm, cached afterward) and your model.
- Inputs must be **BioSemi-64-layout** EEG matching the model's channel count, or you'll get a
  clear "channel mismatch" error.
- **Research demonstration.** v6 is reliably above chance (≈3.6% balanced accuracy vs 2.8% chance,
  on a balanced 36-class problem) but absolute accuracy is modest by the nature of the problem.
