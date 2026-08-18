# Reproducibility notes

What can and cannot be re-run from this repository, and what I found when I
checked the recorded results against the saved artifacts.

`VERIFICATION_LOG.md` records the original claim-by-claim verification. This file
is a second, independent pass done from the packaged artifacts alone, and it
disagrees with the record in a few specific places. Those are listed below.

---

## 1. What you can run

| Level | Needs | Command |
|---|---|---|
| Unit tests | nothing but the deps | `pytest tests/ -q` |
| Headline verification | `results/npz/predictions.npz` | `python scripts/verify_headline.py` |
| Notebooks (read) | nothing — outputs are stored | open `notebooks/*.ipynb` |
| Notebooks (re-execute) | raw iSLEEPS PSG + GPU | not possible from this repo alone |
| Full 10-fold retrain | `data/mm_features/SN*.npz` + GPU | `python code/mmnet_repro.py` |

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest tests/ -q
python scripts/verify_headline.py
```

## 2. What is missing, and why

**Raw data is not here and should not be.** `datasets.PROC_DIR` points at
`data/processed/SN*.npz` and `mmnet_repro.FE` at `data/mm_features/SN*.npz`.
Neither directory ships. The iSLEEPS recordings are access-controlled clinical
PSG, so `data/` is gitignored. Without it you cannot retrain or re-execute the
notebooks — only inspect their stored outputs.

**`results/npz/` is gitignored for size.** It is ~172 MB: `embeddings.npz` and
`supp_artifacts.npz` are ~86 MB each. GitHub warns above 50 MB and hard-rejects
any file above 100 MB, so committing this tree would be a problem regardless.
`predictions.npz` (350 KB) is the only one `verify_headline.py` strictly needs;
the per-event-type section additionally uses `embeddings.npz` (for `sid`) and
`event_labels.npz`. Keep the originals or regenerate via
`mmnet_repro.run_config(..., save_embed=True)`.

## 3. Verification result

`python scripts/verify_headline.py` recomputes 13 recorded numbers from
`predictions.npz`. Run against the shipped artifacts:

```
staging accuracy            0.7220       0.7212   -0.0008   OK
macro F1                    0.6510       0.6528   +0.0018   OK
Cohen kappa                 0.6110       0.6086   -0.0024   OK
respiratory AUC             0.7110       0.7058   -0.0052   OFF
respiratory AP              0.3370       0.3185   -0.0185   OFF
per-class F1 [W]            0.7700       0.7760   +0.0060   OFF
per-class F1 [N1]           0.2700       0.2842   +0.0142   OFF
per-class F1 [N2]           0.7800       0.7822   +0.0022   OK
per-class F1 [N3]           0.6800       0.6702   -0.0098   OFF
per-class F1 [R]            0.7200       0.7514   +0.0314   OFF
AUC [hypopnea]              0.6920       0.6917   -0.0003   OK
AUC [obstructive_apnea]     0.7630       0.7605   -0.0025   OK
AUC [central_apnea]         0.8400       0.8159   -0.0241   OFF
```

**The three headline staging figures reproduce tightly** (≤ 0.0024), as do
hypopnea and obstructive-apnea AUC.

**Two integrity checks passed exactly**, which is the stronger result:

- The `any event` column of `event_labels.npz` is **bit-identical** to the
  trained apnea label — 0 mismatches over 89,532 epochs. This independently
  confirms VERIFICATION_LOG row 8.
- `embeddings.npz['stage']` is array-equal to `predictions.npz['y_true']`, so
  the two artifacts share one epoch ordering and can be safely joined. SN28, the
  known SN15 duplicate, is the only subject present in `event_labels.npz` and
  absent from the predictions — the de-duplication in VERIFICATION_LOG row 7
  did what it claims.

**But `results/README.md` overstates the agreement.** It says separate engine
runs "differ from the notebook in the third decimal (GPU non-determinism)". That
holds for staging accuracy, but not for four numbers: REM F1 is off by 0.031,
central-apnea AUC by 0.024, respiratory AP by 0.019, and N1 F1 by 0.014 — all
second-decimal. These are the minority classes and the precision-sensitive
metric, exactly where a small number of positives makes run-to-run variance
large (central apnea has only 831 positive epochs of 89,532). The honest
statement is *"staging reproduces to the third decimal; minority-class F1 and
respiratory AP move in the second."*

Note on method: per-event-type AUC scores one event type against **clean
non-event epochs**. That choice is not arbitrary — it reproduces hypopnea
(0.6917 vs 0.692) and obstructive (0.7605 vs 0.763) closely, whereas scoring
against all other epochs gives 0.682 / 0.733, which matches neither. So the
central-apnea gap is a real run-to-run difference, not a definitional one.

## 4. Discrepancies found in the recorded results

**a. The reported parameter count belongs to the wrong model.**
`headline_metrics.csv` records `params_million,0.86`, and `staging_benchmark.csv`
gives 0.86 to the row whose accuracy (0.721) is the concat result. Measured
directly:

| fusion | parameters |
|---|---|
| `cross` (attention) | **0.856 M** |
| `concat` (**the headline**) | **0.773 M** |
| `eeg_only` | 0.765 M |

0.86 M is the cross-attention variant. The headline model is concat —
`mmnet_repro.py` runs `run_config("headline_concat", fusion="concat")`, and
VERIFICATION_LOG row 4 states concat was made the headline with attention
demoted to an ablation. The parameter count appears to be the one figure that
switch missed. **The headline model should be reported as 0.77 M, not 0.86 M.**
Pinned by `tests/test_models.py::test_reported_0_86M_is_the_cross_model_not_the_headline_concat`.

**b. Severity-band patient counts disagree with the verification log.**
`staging_by_severity.csv` lists 14 / 22 / 23 / 37 patients (= 96, the
respiratory cohort). VERIFICATION_LOG row 10 states 15 / 24 / 23 / 38 (= 100).
Both can be right — the CSV is restricted to subjects with predictions — but
neither document says so. Worth one clarifying sentence.

## 5. Code issues worth fixing

**Fold assignment depends on input order, not just the seed.**
`make_folds` shuffles the caller's list *in place* and then strides it, so the
same `seed=42` with a differently ordered subject list produces a different
partition. Every current caller passes `sorted(data)`, so the published folds
are well defined — but anything that changes iteration order (a `glob` returning
a new order, an unsorted dict) would silently invalidate every cached per-fold
result and every number derived from it. Both behaviours are pinned in
`tests/test_datasets.py`. A one-line `sids = sorted(subjects)` inside
`make_folds` would remove the hazard.

**Stale `sys.path` entries.** `mmnet_repro.py` inserts `ROOT/model`,
`ROOT/utils` and `ROOT/processing`; `cardio_features.py` inserts `ROOT/utils`.
None of those directories exist — they are leftovers from an earlier layout. The
imports still succeed, because the modules all sit in `code/` and Python puts a
script's own directory on `sys.path`. So this is dead, misleading code rather
than breakage, but it means the intended import path is not the one being used.
`cardio_features.py`'s docstring likewise still says `python extra/cardio_features.py`.

**Importing `mmnet_repro` has a filesystem side effect.** `os.makedirs(RUNS)`
runs at module scope, so merely importing it creates `results/revision/runs/`.
Collecting the test suite is enough to trigger it. That path is gitignored, but
directory creation belongs inside `run_config`, not at import time.

**`cardio_features.cardio_feats` docstring contradicts the code.** It documents
thoraco-abdominal asynchrony as `|corr(Thorax, Abdomen)|`, but the implementation
stores the *signed* correlation. Signed is the more useful choice — it separates
paradoxical from synchronous breathing, which is the obstructive-event marker —
so the docstring is what is wrong. Pinned in `tests/test_cardio_features.py`.

**Minor:** `cardio_features.run()` takes a `subs` parameter it never uses.

## 6. Test suite

88 tests, no data or GPU required, ~3 s:

| File | Covers |
|---|---|
| `test_features.py` | feature widths (the 188 the network hard-codes), band-power response to known tones, amplitude scaling, degenerate/flat channels |
| `test_datasets.py` | fold disjointness, full cohort coverage, each subject tested exactly once, determinism, SN28 exclusion |
| `test_cardio_features.py` | the 14 cardio columns, desaturation depth, asynchrony sign, and that `mmnet_repro.CARD_GROUPS` still tiles those columns correctly |
| `test_hmm.py` | Viterbi against brute-force enumeration, spike smoothing, prior handling |
| `test_models.py` | output shapes, pinned parameter counts, gradient flow, and that `eeg_only` is genuinely cardio-free |

The `eeg_only` test is the one that matters most for the paper's argument: the
apnea head reads a direct cardio pathway, so "remove the cardio modality" is
only honest if that pathway is zeroed too. The test asserts that scaling the
cardio input by 100x moves neither output by a float.

`test_models.py` skips as a group if torch is absent, so the other 60-odd tests
still run in a light environment.
