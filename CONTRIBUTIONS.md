# Contributions

Per-member contributions for the MM-Net manuscript. Each entry names specific
sections, experiments, figures or code, and is traceable to the commit history
of this repository (`github.com/EsmeAbha/iSleep_experiments`).

---

## Esme Abha — reproducibility, verification and release engineering

Owned the reproducibility and verification layer of the project. Specifically:

- **Packaged the delivered archive into a working repository** (`code/`,
  `notebooks/`, `results/`, `paper/`, `scripts/`, `tests/`), with `.gitignore`,
  `requirements.txt` and a full README. Kept 172 MB of derived `.npz` artifacts
  and a 169 MB source archive out of git history — both exceed GitHub's limits
  and are regenerable. *(commits `4167f9e`, `e9ccb5a`)*

- **Wrote `scripts/verify_headline.py`**, which recomputes 13 reported numbers
  from the saved 10-fold predictions and diffs them against `results/*.csv`. It
  requires no GPU, no training and no access to the clinical recordings, and
  exits non-zero on drift so it runs in CI. *(commit `4167f9e`)*

- **Wrote the 89-test suite** in `tests/` covering the feature extractors, fold
  construction, the Viterbi stage smoother (validated against brute-force
  enumeration) and both network definitions. The load-bearing test asserts that
  the `eeg_only` ablation is genuinely cardio-free — the apnea head has a direct
  cardio pathway, so the modality ablation is only valid if that path is zeroed.
  *(commits `4167f9e`, `ccf0666`, `dbbca9c`)*

- **Found and corrected a parameter-count error in the results tables.**
  `headline_metrics.csv` and `staging_benchmark.csv` reported 0.86 M parameters;
  0.86 M is the cross-attention variant's size, while the headline concat model
  has 0.773 M. Established from the engine cache that `neural_only` is
  `fusion=concat` with the cardio columns zeroed — the same architecture — so
  both benchmark rows must report the same size. Confirmed independently by
  `3_figure_hypnogram.ipynb`, which prints `params: 773254`. Corrected both files
  and added tests that compare the CSVs against a constructed model so the tables
  cannot drift from the code again. *(commit `dbbca9c`)*

- **Fixed a latent reproducibility hazard in fold construction.** `make_folds()`
  shuffled the caller's list in place, making the caller's iteration order part
  of the fold assignment; any change in ordering would have silently invalidated
  every cached per-fold result while still appearing deterministic. Verified the
  published folds were unchanged by regenerating the k=5 and k=10 partitions
  before and after the fix and diffing them, then pinned fold 0's test subjects
  as a regression guard. *(commit `ccf0666`)*

- **Cleaned up import hygiene in the reproduction modules** — removed stale
  `sys.path` entries pointing at directories that no longer exist, removed a
  silent fallback to a duplicate copy of `make_folds` inside the module that
  defines the evaluation protocol, and moved directory creation out of module
  scope so importing no longer has filesystem side effects. *(commit `a19ac91`)*

- **Set up continuous integration** (`.github/workflows/ci.yml`): the test suite
  on Python 3.10 and 3.12, the headline verification when artifacts are present,
  and an import-hygiene check. *(commit `9e4b454`)*

- **Wrote `REPRODUCIBILITY.md`**, an independent second verification pass from
  the packaged artifacts. It confirms two integrity properties exactly — the
  event-label column is bit-identical to the trained apnea label (0 mismatches
  over 89,532 epochs), and the embedding and prediction artifacts share one epoch
  ordering — and documents where recorded results do not hold up, including that
  `results/README.md`'s "third decimal" agreement claim understates the true
  spread (REM F1 by 0.031, central-apnea AUC by 0.024, respiratory AP by 0.019).
  *(commits `4167f9e`, `e43aecd`)*

- **Documented the environment and result traceability** in the README: hardware
  (NVIDIA RTX 2060), hosts, recorded Python versions and runtimes, and a map from
  each reported result to the notebook and code cell that produced it — including
  an explicit statement of what does *not* trace to a notebook cell.
  *(commits `1a93327`, `ac342ff`)*

---

## [Member 2 — full name]

*Replace with specific contributions: sections written, experiments run, figures
and tables produced, code authored. Follow the brief's example — "ran the
ablation study and produced Tables 6–7 and Figure 5", not "helped with
experiments". Each item should be checkable against the commit history.*

## [Member 3 — full name]

*As above.*

## [Member 4 — full name]

*As above.*

---

## Note on tooling

AI assistance (Claude) was used for the packaging, test-suite, verification-script
and documentation work listed above; the sessions are included in the submission's
evidence of work. Commit `b5801a1` records this co-authorship in the git history.
All reported experimental results were produced by the notebooks in `notebooks/`,
and every correction listed above was verified against the saved artifacts before
being applied.
