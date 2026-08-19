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

**a. The reported parameter count belonged to the wrong model. — FIXED**
`headline_metrics.csv` recorded `params_million,0.86`, and
`staging_benchmark.csv` gave 0.86 to the row whose accuracy (0.721) is the
concat result. Measured directly:

| fusion | parameters |
|---|---|
| `cross` (attention) | **0.856 M** |
| `concat` (**the headline**) | **0.773 M** |
| `eeg_only` | 0.765 M |

0.86 M is the cross-attention variant. The headline model is concat —
`mmnet_repro.py` runs `run_config("headline_concat", fusion="concat")`, and
VERIFICATION_LOG row 4 states concat was made the headline with attention
demoted to an ablation. The parameter count was the one figure that switch
missed.

The engine cache settles it. `results/engine_cache_per_fold/*.json` records the
config each run used:

| run | fusion | card_drop |
|---|---|---|
| `headline_concat` | `concat` | — |
| `attention_cross` | `cross` | — |
| `neural_only` | **`concat`** | `['all']` |

So `staging_benchmark.csv`'s "MM-Net (neural only)" row is *not* a smaller
network — it is the same concat architecture with the cardio feature columns
zeroed, exactly as `mmnet_repro`'s docstring describes ablation ("the model
architecture is unchanged; the information is removed"). Both MM-Net rows are
therefore the same architecture and must report the same size.

Corrected to **0.773 M** in both files. Two tests now read the CSVs and compare
them against a freshly constructed model, so the tables cannot drift from the
code again:
`test_reported_parameter_count_matches_the_headline_architecture` and
`test_both_benchmark_rows_share_the_concat_architecture`.

If `0.86 M` also appears in `paper/multimodal.pdf`, it needs the same correction
there — the repo and the paper now disagree.

**b. Severity-band patient counts disagree with the verification log.**
`staging_by_severity.csv` lists 14 / 22 / 23 / 37 patients (= 96, the
respiratory cohort). VERIFICATION_LOG row 10 states 15 / 24 / 23 / 38 (= 100).
Both can be right — the CSV is restricted to subjects with predictions — but
neither document says so. Worth one clarifying sentence.

## 5. Code issues — all fixed

**Fold assignment depended on input order, not just the seed. — FIXED**
`make_folds` shuffled the caller's list *in place* and then strided it, so the
same `seed=42` with a differently ordered subject list produced a different
partition. Every caller passes `sorted(data)`, so the published folds were well
defined — but anything that changed iteration order (a `glob` returning a new
order, an unsorted dict) would have silently invalidated every cached per-fold
result while still looking deterministic.

`make_folds` now sorts its input before shuffling. **The published folds are
byte-identical** — verified by regenerating the k=5 and k=10 partitions before
and after the change and diffing them. Fold 0's test subjects
(`[4, 33, 36, 43, 51, 64, 76, 85, 96, 100]`) are now pinned as a regression
guard, and `test_partition_depends_on_the_seed_alone` checks reversed, rotated
and shuffled input all produce the same partition.

**Stale `sys.path` entries. — FIXED** `mmnet_repro.py` inserted `ROOT/model`,
`ROOT/utils` and `ROOT/processing`; `cardio_features.py` inserted `ROOT/utils`.
None of those directories exist — leftovers from an earlier layout. The imports
still succeeded, because the modules all sit in `code/` and Python puts a
script's own directory on `sys.path`, so this was dead, misleading code rather
than breakage. Both now add `code/` explicitly, which also makes them importable
as modules and not only runnable as scripts. `cardio_features.py`'s docstring no
longer says `python extra/cardio_features.py`.

**A silent fallback to a duplicate fold builder. — FIXED** `mmnet_repro.py`
wrapped `from datasets import make_folds` in a bare `try/except` that fell back
to a second, inline copy of the fold logic. The two happened to agree on fold
membership, but a silent fallback to a duplicate implementation in the module
that *defines the evaluation protocol* is the wrong failure mode — if they ever
diverged, every reported metric would quietly describe a different split. The
import is now explicit and fails loudly. CI asserts
`mmnet_repro.make_folds is datasets.make_folds`.

**Importing `mmnet_repro` had a filesystem side effect. — FIXED**
`os.makedirs(RUNS)` ran at module scope, so merely importing it created
`results/revision/runs/`; collecting the test suite was enough to trigger it.
Moved inside `run_config`, and CI asserts the directory is absent after import.

**`cardio_feats` docstring contradicted the code. — FIXED** It documented
thoraco-abdominal asynchrony as `|corr(Thorax, Abdomen)|`, but the implementation
stores the *signed* correlation. Signed is the correct choice — +1 is
synchronous breathing and −1 paradoxical, and it is the negative end that marks
an obstructive event, so taking the absolute value would fold two opposite
physiological states onto each other. The docstring was what was wrong; it now
says so explicitly. Behaviour pinned in `tests/test_cardio_features.py`.

**Minor. — FIXED** `cardio_features.run()` took a `subs` parameter it never
used; dropped, along with an unused `compute_sample_weight` import.

## 6. Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request, on Python 3.10
and 3.12:

1. `pytest tests/ -q`
2. `scripts/verify_headline.py` — only when `results/npz/predictions.npz` is
   present. It is gitignored, so on a normal runner this step reports that and
   moves on rather than failing the build; it does real work on a self-hosted
   runner or one that restores the artifacts from a cache.
3. An import-hygiene check guarding the fixes above: the reproduction modules
   import cleanly, `mmnet_repro` uses the shared fold builder, and importing it
   creates no directories.

## 7. Test suite

89 tests, no data or GPU required, ~3 s:

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
