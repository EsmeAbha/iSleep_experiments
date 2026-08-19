"""Tests for the fold builder (code/datasets.py).

Fold construction is the single most consequential piece of plumbing in the
project: every headline number is "10-fold patient-independent, seed 42". If
folds ever overlapped between train and test, every metric in the paper would be
optimistic. These tests need no data on disk.
"""
import random

import numpy as np
import pytest

from datasets import CHANNELS, CLASS_NAMES, DUPLICATE_DROP, make_folds

SUBJECTS = [s for s in range(1, 101) if s != 28]   # the real cohort shape


class TestMakeFolds:
    @pytest.mark.parametrize("k", [5, 10])
    def test_returns_k_folds(self, k):
        assert len(make_folds(SUBJECTS, k, seed=42)) == k

    @pytest.mark.parametrize("k", [5, 10])
    def test_train_and_test_are_disjoint(self, k):
        for train, test in make_folds(SUBJECTS, k, seed=42):
            assert set(train).isdisjoint(test), "patient leakage between train and test"

    @pytest.mark.parametrize("k", [5, 10])
    def test_train_plus_test_covers_the_cohort(self, k):
        for train, test in make_folds(SUBJECTS, k, seed=42):
            assert set(train) | set(test) == set(SUBJECTS)

    @pytest.mark.parametrize("k", [5, 10])
    def test_every_subject_is_tested_exactly_once(self, k):
        """The pooled 89,532-epoch prediction set depends on this: each subject
        must contribute test predictions once and only once."""
        counts = {}
        for _, test in make_folds(SUBJECTS, k, seed=42):
            for s in test:
                counts[s] = counts.get(s, 0) + 1
        assert set(counts) == set(SUBJECTS)
        assert set(counts.values()) == {1}

    def test_is_deterministic_for_a_fixed_seed(self):
        assert make_folds(SUBJECTS, 10, seed=42) == make_folds(SUBJECTS, 10, seed=42)

    def test_different_seeds_give_different_partitions(self):
        assert make_folds(SUBJECTS, 10, seed=42) != make_folds(SUBJECTS, 10, seed=7)

    def test_partition_depends_on_the_seed_alone(self):
        """make_folds sorts its input before shuffling, so the caller's iteration
        order is not part of the fold assignment. Before that fix, a glob
        returning files in a new order would have silently reshuffled the folds
        and invalidated every cached per-fold result while still looking
        deterministic. See REPRODUCIBILITY.md.
        """
        expected = make_folds(sorted(SUBJECTS), 10, seed=42)
        for label, order in [
            ("reversed", list(reversed(sorted(SUBJECTS)))),
            ("rotated", SUBJECTS[37:] + SUBJECTS[:37]),
            ("shuffled", random.Random(9).sample(SUBJECTS, len(SUBJECTS))),
        ]:
            assert make_folds(order, 10, seed=42) == expected, \
                "fold assignment changed for %s input order" % label

    def test_published_fold_assignment_is_unchanged(self):
        """Regression guard on the exact partition every reported number uses.

        Fold 0's test subjects under the published protocol (10-fold, seed 42,
        cohort minus the SN28 duplicate). If this ever changes, the cached
        per-fold results and the paper's metrics no longer describe the same
        split.
        """
        _, test0 = make_folds(SUBJECTS, 10, seed=42)[0]
        assert test0 == [4, 33, 36, 43, 51, 64, 76, 85, 96, 100]

    @pytest.mark.parametrize("k", [5, 10])
    def test_folds_are_balanced_within_one_subject(self, k):
        sizes = [len(test) for _, test in make_folds(SUBJECTS, k, seed=42)]
        assert max(sizes) - min(sizes) <= 1

    def test_returns_sorted_ids(self):
        for train, test in make_folds(SUBJECTS, 10, seed=42):
            assert train == sorted(train)
            assert test == sorted(test)


class TestCohortConstants:
    def test_sn28_is_dropped_as_a_duplicate(self):
        """SN28 is byte-identical to SN15; keeping both would double-count a
        patient across folds (VERIFICATION_LOG row 7)."""
        assert 28 in DUPLICATE_DROP

    def test_channel_and_class_constants(self):
        assert CHANNELS == ["C4:M1", "C3:M2", "O2:M1", "O1:M2"]
        assert CLASS_NAMES == ["W", "N1", "N2", "N3", "R"]
