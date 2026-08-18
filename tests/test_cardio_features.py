"""Tests for the cardiorespiratory feature block (code/cardio_features.py).

These 14 features are what the whole cardio argument rests on: the modality
ablation in the paper works by zeroing *column groups* of this vector, so the
column order here and the group indices in mmnet_repro.CARD_GROUPS must agree.
A silent reordering would mislabel every ablation row in Table 3.
"""
import numpy as np
import pytest

from cardio_features import CARD, CTX, cardio_feats, context

N_CARDIO_FEATS = 14
T = 750             # 30 s epoch at 25 Hz
I = {c: k for k, c in enumerate(CARD)}


def blank(n=4):
    return np.zeros((n, len(CARD), T), np.float32)


class TestCardioFeats:
    def test_shape_is_14_columns(self):
        assert cardio_feats(blank(6)).shape == (6, N_CARDIO_FEATS)

    def test_channel_order_is_the_documented_one(self):
        assert CARD == ["ECG", "Flow", "Thorax", "Abdomen", "Effort", "SpO2", "Pulse"]

    def test_no_nan_on_all_zero_input(self):
        """Missing cardio channels are stored as zeros for subjects without a
        complete montage -- corrcoef on a constant channel is NaN and must be
        squashed, or the whole feature matrix poisons training."""
        assert np.isfinite(cardio_feats(blank())).all()

    def test_spo2_summary_columns(self):
        x = blank(1)
        x[0, I["SpO2"]] = np.linspace(90, 98, T)
        f = cardio_feats(x)[0]
        assert f[0] == pytest.approx(94.0, abs=0.1)     # mean
        assert f[1] == pytest.approx(90.0, abs=0.1)     # min
        assert f[2] > 0                                  # std

    def test_desaturation_depth_is_median_minus_min(self):
        """Column 3 is the desaturation-depth feature -- the single most
        physiologically load-bearing input to the respiratory head."""
        x = blank(1)
        sp = np.full(T, 97.0, np.float32)
        sp[:50] = 85.0                                   # a desaturation dip
        x[0, I["SpO2"]] = sp
        f = cardio_feats(x)[0]
        assert f[3] == pytest.approx(np.median(sp) - sp.min(), abs=1e-4)
        assert f[3] == pytest.approx(12.0, abs=1e-4)

    def test_pulse_columns_track_mean_and_variability(self):
        x = blank(1)
        x[0, I["Pulse"]] = np.full(T, 60.0, np.float32)
        f = cardio_feats(x)[0]
        assert f[4] == pytest.approx(60.0)
        assert f[5] == pytest.approx(0.0, abs=1e-6)

    def test_line_length_grows_with_signal_activity(self):
        """Line-length (cols 7, 9) is amplitude x rate -- a faster wave at the
        same amplitude must score higher."""
        slow, fast = blank(1), blank(1)
        tt = np.arange(T) / 25.0
        slow[0, I["Flow"]] = np.sin(2 * np.pi * 0.2 * tt)
        fast[0, I["Flow"]] = np.sin(2 * np.pi * 2.0 * tt)
        assert cardio_feats(fast)[0, 9] > cardio_feats(slow)[0, 9]

    def test_thoracoabdominal_asynchrony_is_signed_correlation(self):
        """In-phase breathing -> +1, paradoxical (anti-phase) -> -1.

        NOTE: the module docstring describes this feature as |corr(Thorax,
        Abdomen)|, but the implementation stores the *signed* correlation. The
        signed form is the more useful one (it distinguishes paradoxical from
        synchronous breathing, which is the obstructive-event marker), so this
        test pins the actual behaviour; the docstring is what is wrong.
        """
        tt = np.arange(T) / 25.0
        wave = np.sin(2 * np.pi * 0.25 * tt).astype(np.float32)

        in_phase = blank(1)
        in_phase[0, I["Thorax"]] = wave
        in_phase[0, I["Abdomen"]] = wave
        assert cardio_feats(in_phase)[0, 13] == pytest.approx(1.0, abs=1e-4)

        anti_phase = blank(1)
        anti_phase[0, I["Thorax"]] = wave
        anti_phase[0, I["Abdomen"]] = -wave
        assert cardio_feats(anti_phase)[0, 13] == pytest.approx(-1.0, abs=1e-4)

    def test_asynchrony_is_zero_when_a_channel_is_flat(self):
        x = blank(1)
        x[0, I["Thorax"]] = np.sin(np.arange(T) / 10.0)
        # Abdomen left flat -> corrcoef undefined -> guarded to 0.0
        assert cardio_feats(x)[0, 13] == 0.0


class TestAblationGroupsMatchColumns:
    """mmnet_repro zeroes cardio columns by group name; the indices there must
    stay in sync with the column order produced above."""

    def test_groups_partition_all_14_columns(self):
        from mmnet_repro import CARD_GROUPS
        covered = sorted(i for g in CARD_GROUPS.values() for i in g)
        assert covered == list(range(N_CARDIO_FEATS)), "ablation groups must tile the vector"

    def test_no_column_appears_in_two_groups(self):
        from mmnet_repro import CARD_GROUPS
        flat = [i for g in CARD_GROUPS.values() for i in g]
        assert len(flat) == len(set(flat))

    def test_group_indices_point_at_the_right_features(self):
        from mmnet_repro import CARD_GROUPS
        assert CARD_GROUPS["spo2"] == [0, 1, 2, 3]        # mean/min/std/desat-depth
        assert CARD_GROUPS["pulse_hrv"] == [4, 5]         # pulse mean/std
        assert CARD_GROUPS["ecg"] == [6, 7]               # ECG std / line-length
        assert CARD_GROUPS["airflow"] == [8, 9]           # Flow std / line-length
        assert CARD_GROUPS["effort"] == [10, 11, 12, 13]  # thorax/abdomen/effort + asynchrony


class TestContextStacking:
    def test_width_is_2k_plus_1_times_features(self):
        F = np.arange(20 * 3, dtype=np.float32).reshape(20, 3)
        assert context(F, k=3).shape == (20, 3 * (2 * 3 + 1))

    def test_row_count_is_preserved(self):
        F = np.zeros((17, 5), np.float32)
        assert context(F, k=CTX).shape[0] == 17

    def test_centre_slice_is_the_original_row(self):
        """With k=3 the centre block (columns 3F:4F) must be the unshifted row."""
        F = np.arange(10 * 4, dtype=np.float32).reshape(10, 4)
        C = context(F, k=3)
        np.testing.assert_array_equal(C[:, 3 * 4:4 * 4], F)

    def test_edges_are_padded_by_repetition_not_zeros(self):
        F = np.ones((5, 2), np.float32)
        assert (context(F, k=2) == 1.0).all()
