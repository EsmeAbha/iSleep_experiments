#!/usr/bin/env python3
"""
verify_headline.py -- recompute the paper's headline numbers from the saved
artifacts and diff them against the values recorded in results/*.csv.

This is the one check a reader can run WITHOUT the raw iSLEEPS recordings or a
GPU: it never trains anything, it only re-derives metrics from the pooled
10-fold test-set predictions that the reproduction engine saved.

    python scripts/verify_headline.py
    python scripts/verify_headline.py --tol 0.005 --results-dir results

Exit status is 0 when every claim is within tolerance, 1 otherwise, so this can
be wired into CI.

Requires: results/npz/predictions.npz  (headline staging + respiratory metrics)
Optional: results/npz/embeddings.npz   (adds per-event-type AUC; needs `sid`)
          results/npz/event_labels.npz
Both npz files are gitignored for size -- see REPRODUCIBILITY.md.
"""
import argparse
import csv
import os
import sys

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             cohen_kappa_score, f1_score, roc_auc_score)

CLASS_NAMES = ["W", "N1", "N2", "N3", "R"]
# event_labels.npz column order, established by matching column sums against the
# counts recorded in results/per_event_type_auc.csv (11605 / 1661 / 831).
# Column 0 ("any event") is bit-identical to predictions.npz:apnea_true.
EVENT_COLS = {"any": 0, "hypopnea": 1, "obstructive_apnea": 2, "central_apnea": 3}


def _fail(msg):
    print("ERROR: " + msg, file=sys.stderr)
    raise SystemExit(2)


def read_csv_map(path, key_col, val_col):
    """{key: float(value)} from a two-column-of-interest CSV."""
    out = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row[key_col]] = float(row[val_col])
            except (ValueError, KeyError, TypeError):
                continue          # non-numeric rows (e.g. the free-text `note`)
    return out


def subject_blocks(sid):
    """Yield (subject_id, start, end) for each contiguous run in `sid`."""
    if len(sid) == 0:
        return
    edges = np.flatnonzero(np.diff(sid)) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(sid)]])
    for a, b in zip(starts, ends):
        yield int(sid[a]), int(a), int(b)


class Report:
    """Collects claim-vs-recomputed rows and prints one aligned table."""

    def __init__(self, tol):
        self.tol = tol
        self.rows = []

    def check(self, label, claimed, actual, tol=None):
        tol = self.tol if tol is None else tol
        delta = actual - claimed
        ok = abs(delta) <= tol
        self.rows.append((label, claimed, actual, delta, ok, tol))
        return ok

    def render(self):
        w = max(len(r[0]) for r in self.rows) + 2
        header = "metric".ljust(w) + "paper".rjust(9) + "recomputed".rjust(13)
        print("\n" + header + "delta".rjust(10) + "   status")
        print("-" * (w + 45))
        for label, claimed, actual, delta, ok, tol in self.rows:
            status = "OK" if ok else "OFF  (tol %g)" % tol
            print("%s%9.4f%13.4f%+10.4f   %s"
                  % (label.ljust(w), claimed, actual, delta, status))
        bad = [r for r in self.rows if r[4] is False]
        print("-" * (w + 45))
        print("%d metrics checked, %d outside tolerance" % (len(self.rows), len(bad)))
        return bad


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--tol", type=float, default=0.005,
                    help="absolute tolerance for a claim to count as reproduced "
                         "(default 0.005; GPU non-determinism moves the 3rd decimal)")
    args = ap.parse_args()

    res = args.results_dir
    npz_dir = os.path.join(res, "npz")
    pred_path = os.path.join(npz_dir, "predictions.npz")
    if not os.path.exists(pred_path):
        _fail(pred_path + " not found. It is gitignored for size; "
              "see REPRODUCIBILITY.md for how to obtain or regenerate it.")

    P = np.load(pred_path)
    y_true, y_pred = P["y_true"], P["y_pred"]
    apnea_true, apnea_score = P["apnea_true"], P["apnea_score"]

    print("loaded {:,} pooled test epochs from {}".format(len(y_true), pred_path))
    print("stage distribution: " + ", ".join(
        "{}={:,}".format(c, n)
        for c, n in zip(CLASS_NAMES, np.bincount(y_true, minlength=5))))
    print("respiratory-event prevalence: {:.4f} ({:,} positive epochs)".format(
        apnea_true.mean(), int(apnea_true.sum())))

    rep = Report(args.tol)

    # ---- headline staging + respiratory -------------------------------------
    claims = read_csv_map(os.path.join(res, "headline_metrics.csv"), "metric", "value")
    rep.check("staging accuracy", claims["staging_accuracy"],
              accuracy_score(y_true, y_pred))
    rep.check("macro F1", claims["macro_f1"],
              f1_score(y_true, y_pred, average="macro", zero_division=0))
    rep.check("Cohen kappa", claims["cohen_kappa"], cohen_kappa_score(y_true, y_pred))
    rep.check("respiratory AUC", claims["respiratory_auc"],
              roc_auc_score(apnea_true, apnea_score))
    rep.check("respiratory AP", claims["respiratory_average_precision"],
              average_precision_score(apnea_true, apnea_score))

    n_claim = int(claims["n_epochs"])
    if n_claim != len(y_true):
        print("\nNOTE: headline_metrics.csv says n_epochs={:,} but the saved "
              "predictions contain {:,}".format(n_claim, len(y_true)))

    # ---- per-class F1 --------------------------------------------------------
    pcf_claim = read_csv_map(os.path.join(res, "per_class_f1.csv"), "stage", "f1")
    pcf = f1_score(y_true, y_pred, average=None, labels=range(5), zero_division=0)
    for name, val in zip(CLASS_NAMES, pcf):
        rep.check("per-class F1 [%s]" % name, pcf_claim[name], float(val))

    # ---- per-event-type AUC (needs sid to align event labels to scores) ------
    emb_path = os.path.join(npz_dir, "embeddings.npz")
    ev_path = os.path.join(npz_dir, "event_labels.npz")
    if os.path.exists(emb_path) and os.path.exists(ev_path):
        sid = np.load(emb_path)["sid"]
        EV = np.load(ev_path)
        # Rebuild the per-epoch event matrix in prediction order. `sid` is grouped
        # by subject in fold/test order, so walk contiguous runs rather than sort.
        blocks = []
        for s, a, b in subject_blocks(sid):
            key = "SN%d" % s
            if key not in EV:
                _fail("%s in embeddings.sid but missing from event_labels.npz" % key)
            block = EV[key]
            if len(block) != b - a:
                _fail("%s: event_labels has %d rows, predictions have %d"
                      % (key, len(block), b - a))
            blocks.append(block)
        events = np.concatenate(blocks)

        # Cross-check: column 0 must equal the trained apnea label exactly.
        mismatches = int((events[:, EVENT_COLS["any"]] != apnea_true).sum())
        print("\nevent-label integrity: 'any event' column vs trained apnea label "
              "-> {} mismatches over {:,} epochs".format(mismatches, len(apnea_true)))

        ev_csv = os.path.join(res, "per_event_type_auc.csv")
        auc_claim = read_csv_map(ev_csv, "event_type", "auc")
        n_claim_ev = read_csv_map(ev_csv, "event_type", "positive_epochs")
        negatives = apnea_true == 0
        for etype in ("hypopnea", "obstructive_apnea", "central_apnea"):
            col = events[:, EVENT_COLS[etype]] > 0
            # this event type versus clean non-event epochs
            mask = col | negatives
            auc = roc_auc_score(col[mask].astype(int), apnea_score[mask])
            rep.check("AUC [%s]" % etype, auc_claim[etype], float(auc))
            if int(col.sum()) != int(n_claim_ev[etype]):
                print("NOTE: %s positive epochs: csv=%d recomputed=%d"
                      % (etype, int(n_claim_ev[etype]), int(col.sum())))
    else:
        print("\nskipping per-event-type AUC: needs results/npz/embeddings.npz "
              "and event_labels.npz (gitignored for size)")

    bad = rep.render()
    if bad:
        print("\nMetrics outside tolerance are listed above. A small offset is expected "
              "for artifacts saved by a different run than the one reported in the paper "
              "(GPU non-determinism); see REPRODUCIBILITY.md.")
        return 1
    print("\nAll claims reproduced within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
