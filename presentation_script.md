# NUMA — Presentation / Video Script (5–7 minutes)

Spoken narration mapped to `NUMA_presentation.pdf`. Target ~5–7 min (~900–1,000 words).
Timing cues in brackets. Maps directly to the CS 153 rubric (Problem & Insight, Execution &
Technical Work, Evaluation & Evidence, Communication, Process/Integrity).

---

### Slide 1 — Title  [0:00–0:25]
"Hi — I'm Neel, and this is **NUMA**, a one-person frontier lab for non-invasive brain
decoding. The question I set out to answer: can we read *which of 36 imagined objects* —
animals and tools — a person is thinking about, from signals recorded outside the skull,
with no surgery? And can one person, with AI tooling, run the whole research program
end-to-end. Everything I'll show is live at the link on screen."

### Slide 2 — Problem & Insight  [0:25–1:05]
"The core bottleneck in non-invasive brain–computer interfaces is **signal-to-noise**. The
skull smears the brain's weak electric fields, so EEG is incredibly *fast* but spatially
*blind*. Decoding 36 distinct concepts from a single trial is brutal — random chance is just
**2.78%**. My insight, and the bet of this project, is to **fuse two modalities with
complementary physics**: EEG for millisecond timing, and **fNIRS** — which measures blood-flow
with infrared light — for strong spatial localization of *where* the cortex is active."

### Slide 3 — Approach  [1:05–1:45]
"The naive way to combine them — flatten both and concatenate — fails badly, because you're
forcing one network to digest 2-kilohertz EEG and ~8-hertz fNIRS at once. Instead I use a
**physics-informed prior** called CHASEDOWN: it reconstructs the fNIRS blood-flow map into a
single **per-electrode spatial weight** — with *zero trainable parameters* — and multiplies it
into the EEG. So we localize cortical activity *before* any learning happens. You can see the
prior concentrates over the front-left, matching where the optodes were placed."

### Slide 4 — Architecture  [1:45–2:30]
"The model itself is built for low signal-to-noise. A **depthwise spatio-temporal backbone**
separates *when* a rhythm happens from *where* it happens, which keeps the parameter count tiny
and resists noise. Then — and this is the key piece — instead of treating the scalp as a flat
grid, I model the 64 electrodes as a **graph** using their true physical distances, and do
**localized message-passing** between neighboring electrodes. That graph is what lets the fNIRS
spatial prior flow along *real* cortical pathways instead of acting as random noise."

### Slide 5 — Key Result  [2:30–3:25]  ★ the headline
"Here's the result that answers the scientific question. EEG alone gets about **4.8%**. If you
just multiply in the fNIRS prior with no graph, accuracy actually *drops* to **2.93%** — the
prior is destructive noise without the right architecture. But route that **same** prior
through the graph network, and you jump to **9.16%** — a **+90.8% relative improvement** over
EEG alone, and it's **statistically significant at p = 0.005**. The naive version wasn't
significant — p of 0.14. So the verified finding is precise: **fNIRS plus EEG beats EEG, but
only when you explicitly model the brain's non-Euclidean geometry.** That's mechanistic
evidence that multimodal fusion is the right path — not just a number that went up."

### Slide 6 — Autoresearch  [3:25–4:10]  ★ the automation story
"Now the 'frontier lab' part. I didn't hand-tune this. I built an **autonomous agent** — an LLM
that edits the training code, runs an experiment, measures the result, and keeps or reverts the
change, on its own, in a loop. The benchmark is **frozen and un-gameable**: a fixed data split,
a fixed time budget, scored on *balanced* accuracy so it can't cheat by collapsing to one
class. It ran **26 experiments**, kept the **4** that actually helped, and ruled out **22**
dead-ends — capacity bumps, extra layers, augmentations — documenting *why* each failed. That's
how one person scales into an automated experimental program."

### Slide 7 — Iteration  [4:10–4:45]
"Execution-wise, the decoder went through **six honestly-evaluated generations**. I fixed an
early failure where the model collapsed to predicting one class, folded in the agent's winning
configuration, swapped to a stronger backbone, and finally — the biggest idea — moved from
judging noisy one-second windows to a **recording-level attention model** that *learns which
moments actually carry the concept*. That v6 model is what's deployed."

### Slide 8 — Evaluation  [4:45–5:20]
"On evaluation: every number uses **5-fold leave-one-instance-out** cross-validation, so the
model is always tested on repetitions it never saw — no leakage. The deployed EEG model is
**above chance on all five folds**, and the per-class chart shows specific concepts decoded
several times above chance. I report **balanced accuracy** throughout, which can't be faked by
guessing a common class."

### Slide 9 — Evidence & Failure Analysis  [5:20–5:55]
"I also show the failure modes honestly. The confusion matrix makes the hard cases visible, and
I document the real limitation: on this data-starved 36-class problem the representation can
collapse, which is exactly what motivates collecting more synchronous data. Single-trial 36-way
EEG is genuinely hard — the meaningful win here is *consistent, collapse-free* separation above
chance, plus the significant multimodal gain."

### Slide 10 — The Product  [5:55–6:25]
"And it's not just a notebook — it's a **live product**. You upload one recording, and the
predicted object pops up with a confidence chart. It runs **entirely in your browser** with
onnxruntime-web — no server, no upload limits, and your data never leaves your device. The model
is about 118 thousand parameters, under a megabyte, deployed as a static site on Vercel."

### Slide 11 — Impact  [6:25–6:45]
"Why it matters: a reliable non-invasive semantic decoder is a foothold toward **restoring
communication** for people who've lost speech and movement, and toward interfaces driven by
intention. And it's evidence that physics-informed, topology-aware multimodal models are the
right direction for low-signal brain decoding."

### Slide 12 & 13 — Next + Integrity  [6:45–7:05]
"Next, I'd collect a larger multimodal cohort, make the fusion trainable, and add onset-locked
epoching. On integrity: the autoresearch loop *is* an autonomous LLM agent, AI assistants
helped me implement and document, every claim is backed by cross-validation and a significance
test, and it builds on the open ENIGMA backbone and the open Rybář dataset — both cited. Public
repo, full commit history, live demo."

### Slide 14 — Close  [7:05–7:15]
"So: fNIRS plus EEG, routed through cortical topology, beats EEG alone — verified, and shipped.
Thanks — and please try the live demo."

---

## Rubric coverage cheat-sheet (for the README / video description)

- **Problem & Insight (3):** Slides 1–3 — SNR bottleneck, 36-class single-trial decoding,
  complementary-physics fusion insight.
- **Execution & Technical Work (5):** Slides 4, 6, 7, 10 — architecture, the autoresearch agent
  system, six-generation iteration, and a functional live in-browser product.
- **Evaluation & Evidence (3):** Slides 5, 8, 9 — significant +90.8% multimodal gain (p=0.005),
  5-fold CV above chance, balanced-accuracy reporting, confusion-matrix failure analysis,
  cited baselines/dataset.
- **Communication & Presentation (2):** image-forward deck, plain-language script, live demo,
  comprehensive README.
- **Process, Integrity & Disclosure (2):** Slide 13 — AI usage disclosed (agentic loop +
  assistants), ENIGMA + dataset credited, limitations discussed, public repo + commit history.
