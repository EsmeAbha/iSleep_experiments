# MM-Net — joint sleep staging and respiratory-event detection

Code, results and paper for *"A Physiologically Interpretable Multimodal Model
for Joint Sleep Staging and Respiratory-Event Detection in Subacute Ischemic
Stroke."*

A single network reads a night of polysomnography and produces both outputs at
once: a 5-class sleep stage per 30-second epoch, and a per-epoch
respiratory-event score. It is built on the iSLEEPS stroke cohort (99 patients
for staging, 96 with a complete cardiorespiratory montage), evaluated
10-fold patient-independent with a fixed fold assignment.

| | staging | respiratory |
|---|---|---|
| **MM-Net (concat, headline)** | acc **0.722** · macro-F1 **0.651** · κ **0.611** | AUC **0.711** · AP **0.337** |
| best single-EEG baseline (AttnSleep) | acc 0.686 | — |
| best cardio-feature baseline (gradient boosting) | — | AUC 0.670 |

0.773 M parameters. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) §4a — the
0.86 M figure in the results tables is the cross-attention variant's size, not
the headline model's.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest tests/ -q                    # 89 tests, no data or GPU needed (~3 s)
python scripts/verify_headline.py   # recompute the paper's numbers from saved predictions
```

`verify_headline.py` needs `results/npz/predictions.npz`, which is gitignored for
size — see [REPRODUCIBILITY.md](REPRODUCIBILITY.md) §2.

## Layout

```
code/            model, features and the reproduction engine
notebooks/       the two executed notebooks every reported number comes from
results/         summary CSVs, per-experiment JSON, and (gitignored) npz artifacts
paper/           multimodal.pdf
scripts/         verify_headline.py — one-command claim check
tests/           unit tests for the feature, fold, decoder and model code
```

### `code/`

| Module | Role |
|---|---|
| `datasets.py` | loads preprocessed per-subject `.npz`, builds the subject-disjoint folds, drops the SN15/SN28 duplicate |
| `features.py` | 23 hand-crafted features per channel — band powers, spectral entropy/edge, Hjorth, time-domain |
| `features_v2.py` | adds AASM-style event features (spindle density, slow-wave amplitude, EOG movement, EMG tone) → **188 EEG features** |
| `cardio_features.py` | 14 cardiorespiratory features (SpO₂ + desaturation depth, pulse, ECG, airflow, effort, thoraco-abdominal asynchrony) |
| `mm_feature_net.py` | **the headline model** — feature MLPs → fusion → BiLSTM → staging + apnea heads |
| `multimodal_net.py` | the raw-signal variant (CNN encoders instead of feature MLPs) |
| `mmnet_repro.py` | cached 10-fold reproduction engine; modality ablation by zeroing feature columns |

### How the model works

Each 30-second epoch becomes two feature vectors — 188 neural (EEG/EOG/EMG) and
14 cardiorespiratory. Two MLPs embed them, a fusion block combines them, and a
BiLSTM reads 20 consecutive epochs so the network sees sleep as a sequence
rather than isolated snapshots. Two heads share that representation: a 5-class
staging head and a binary respiratory head. Staging output is then smoothed by a
Viterbi decoder whose transition matrix is estimated from the training folds.

Three fusion modes make the design ablatable: `cross` (the two modalities attend
to each other), `concat` (interaction removed), and `eeg_only` (cardio removed
entirely, including the apnea head's direct cardio pathway).

**Concat is the headline, not attention.** Cross-attention scores 0.712 / 0.705
against concat's 0.722 / 0.711 — the interaction does not pay for itself here.
See VERIFICATION_LOG row 4.

### Key findings

- **The modality split is clean and physiological.** Removing all
  cardiorespiratory input drops respiratory AUC 0.711 → 0.673 while leaving
  staging untouched (Wilcoxon over 10 folds: respiratory p=0.004, staging
  p=0.91). Removing EOG does the reverse. Cardio carries breathing, ocular
  carries sleep depth.
- **SpO₂ is the single most load-bearing channel** for respiratory detection
  (AUC 0.711 → 0.681 alone).
- **Detection is strongest on the most severe events** — central apnea 0.840,
  obstructive 0.763, hypopnea 0.692.
- **Against a fair baseline the margin is modest.** Gradient boosting on the
  same 14 cardio features reaches AUC 0.670 vs MM-Net's 0.711. The contribution
  is the joint single-pass model with per-modality attribution, not detection
  supremacy — the paper is framed accordingly.
- Predicted per-patient event burden correlates with clinical AHI
  (Spearman ρ = 0.315, p = 0.002, n = 96).

## Data

The iSLEEPS recordings are access-controlled clinical polysomnography and are
**not** in this repository; `data/` is gitignored. The notebooks therefore cannot
be re-executed here — their stored outputs are the record. What *can* be checked
without the raw data is the metric layer, via `scripts/verify_headline.py`.

## Verification

- [`VERIFICATION_LOG.md`](VERIFICATION_LOG.md) — the original claim-by-claim log
  (17 claims: confirmed, corrected, or flagged).
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) — an independent second pass from
  the packaged artifacts, including where the recorded results do not hold up
  and the code issues found along the way.
