"""Tests for the fold builder (code/datasets.py).

Fold construction is the single most consequential piece of plumbing in the
project: every headline number is "10-fold patient-independent, seed 42". If
folds ever overlapped between train and test, every metric in the paper would be
optimistic. These tests need no data on disk.
"""
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

    def test_partition_is_stable_for_sorted_input(self):
        """The seed alone does not fix the partition -- the *input order* is part
        of it, because make_folds shuffles the caller's list in place before
        striding it. Every caller in the repo passes `sorted(data)`, so the
        published folds are well defined; this test pins that contract.
        """
        a = make_folds(sorted(SUBJECTS), 10, seed=42)
        b = make_folds(sorted(SUBJECTS), 10, seed=42)
        assert a == b

    def test_fold_assignment_depends_on_input_order(self):
        """Companion to the test above, documenting the fragility explicitly:
        the same seed with a differently ordered subject list yields a DIFFERENT
        partition. Anything that changes iteration order (a glob returning a new
        order, a dict no longer sorted) would silently invalidate every cached
        per-fold result. Callers must keep sorting. See REPRODUCIBILITY.md.
        """
        assert make_folds(sorted(SUBJECTS), 10, seed=42) \
            != make_folds(list(reversed(sorted(SUBJECTS))), 10, seed=42)

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
