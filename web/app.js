/* ───────────────────────── NUMA Frontier Systems — frontend logic ─────────────────────────
 * Fully in-browser inference (no backend): parse .bdf → preprocess (eeg.js) →
 * run numa_v6.onnx with onnxruntime-web → show the predicted concept's picture.
 *
 * Requires in the repo root:
 *   numa_v6.onnx       (from export_onnx.py)
 *   model_meta.json    (from export_onnx.py — channels, window size, class order)
 *   manifest.json      (class → picture)
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const drop = $("drop"), file = $("file"), go = $("go"),
        fileName = $("fileName"), status = $("status"), error = $("error"),
        apiWarn = $("apiWarn");
  const modal = $("modal"), modalClose = $("modalClose"),
        mImg = $("mImg"), mLabel = $("mLabel"), mCat = $("mCat"),
        mConf = $("mConf"), mTopk = $("mTopk"), mWindows = $("mWindows");

  let manifest = {}, meta = null, session = null, chosen = null;
  let loadError = "";

  // single-threaded wasm — no special COOP/COEP headers needed on Vercel
  if (window.ort) { ort.env.wasm.numThreads = 1; ort.env.wasm.simd = true; }

  // ── load model + metadata + manifest up front ───────────────────────────────
  const ready = (async () => {
    try {
      const [man, mt] = await Promise.all([
        fetch("manifest.json").then((r) => r.json()),
        fetch("model_meta.json").then((r) => { if (!r.ok) throw new Error("model_meta.json missing"); return r.json(); }),
      ]);
      manifest = man; meta = mt;
      session = await ort.InferenceSession.create("numa_v6.onnx", { executionProviders: ["wasm"] });
    } catch (e) {
      loadError = "Model not loaded — add numa_v6.onnx and model_meta.json to the site (see README).";
      console.error(e);
    }
  })();

  ready.then(() => { if (loadError) { apiWarn.hidden = false; apiWarn.textContent = "⚙️ " + loadError; } });

  const pretty = (s) => s.replace(/_/g, " ");
  const softmax = (a) => { const m = Math.max(...a); const e = a.map((x) => Math.exp(x - m)); const s = e.reduce((p, q) => p + q, 0); return e.map((x) => x / s); };

  // ── file selection ──────────────────────────────────────────────────────────
  drop.addEventListener("click", () => file.click());
  file.addEventListener("change", (e) => setFile(e.target.files[0]));
  ["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "dragend"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("drag")));
  drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("drag"); setFile(e.dataTransfer.files[0]); });

  function setFile(f) {
    error.textContent = ""; status.textContent = "";
    if (!f) { chosen = null; fileName.textContent = ""; go.disabled = true; return; }
    if (!f.name.toLowerCase().endsWith(".bdf")) { chosen = null; go.disabled = true; fileName.textContent = ""; error.textContent = "Please choose a .bdf file."; return; }
    chosen = f;
    fileName.innerHTML = `Selected: <b>${escapeHtml(f.name)}</b> · ${(f.size / 1e6).toFixed(1)} MB`;
    go.disabled = false;
  }

  // ── decode ──────────────────────────────────────────────────────────────────
  go.addEventListener("click", async () => {
    if (!chosen) return;
    go.disabled = true; error.textContent = "";
    status.innerHTML = `<span class="spin"></span>Loading model…`;
    await ready;
    if (!session || !meta) { status.textContent = ""; error.textContent = loadError; go.disabled = false; return; }

    try {
      status.innerHTML = `<span class="spin"></span>Reading & filtering EEG (this runs in your browser)…`;
      const buf = await chosen.arrayBuffer();
      // yield so the spinner paints before the heavy synchronous DSP
      await new Promise((r) => setTimeout(r, 30));
      const inp = window.NumaEEG.buildInputs(buf, meta);

      status.innerHTML = `<span class="spin"></span>Running the neural decoder…`;
      const feeds = {
        windows:   new ort.Tensor("float32", inp.windows,   [1, inp.N, 1, inp.C, inp.T]),
        bandpower: new ort.Tensor("float32", inp.bandpower, [1, inp.N, inp.C * (meta.n_bands || 5)]),
        mask:      new ort.Tensor("float32", inp.mask,      [1, inp.N]),
      };
      const out = await session.run(feeds);
      const logits = Array.from(out.logits.data);
      const probs = softmax(logits);
      status.textContent = "";
      showResult(probs, inp.info);
    } catch (e) {
      status.textContent = "";
      error.textContent = "Error: " + e.message;
      console.error(e);
    } finally {
      go.disabled = false;
    }
  });

  // ── render modal ──────────────────────────────────────────────────────────────
  function showResult(probs, info) {
    const order = probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]);
    const topLabel = meta.classes[order[0][1]];
    const m = manifest[topLabel] || {};
    mLabel.textContent = pretty(topLabel);
    mCat.textContent = m.category || "concept";
    mConf.textContent = `${(order[0][0] * 100).toFixed(1)}% confidence`;

    if (m.src) { mImg.src = m.src; mImg.alt = pretty(topLabel); mImg.parentElement.style.display = "flex"; }
    else { mImg.removeAttribute("src"); mImg.parentElement.style.display = "none"; }

    mTopk.innerHTML = "";
    order.slice(0, 5).forEach(([p, i], r) => {
      const pct = (p * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "row" + (r === 0 ? " top" : "");
      row.innerHTML = `<span class="name">${escapeHtml(pretty(meta.classes[i]))}</span>` +
                      `<span class="bar"><div style="width:${pct}%"></div></span>` +
                      `<span class="pct">${pct}%</span>`;
      mTopk.appendChild(row);
    });

    const chance = (1 / meta.classes.length * 100).toFixed(1);
    mWindows.textContent = `Decision pooled over ${info.used} of ${info.totalWindows} clean EEG windows · chance ${chance}%`;
    openModal();
  }

  function openModal() { modal.hidden = false; document.body.style.overflow = "hidden"; }
  function closeModal() { modal.hidden = true; document.body.style.overflow = ""; }
  modalClose.addEventListener("click", closeModal);
  modal.addEventListener("click", (e) => { if (e.target === modal) closeModal(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !modal.hidden) closeModal(); });

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  }
})();
