"""Tests for the hand-crafted feature extractors (code/features.py, features_v2.py).

These run on synthetic signals, so they need neither the iSLEEPS recordings nor a
GPU. They pin down the two properties the downstream models depend on: the feature
vector has the width the networks are constructed for (188 EEG features), and the
spectral features actually respond to the frequency content they claim to measure.
"""
import numpy as np
import pytest

from features import BANDS, extract_features
from features_v2 import event_features, extract_features_v2

FS = 100
T = 3000            # 30 s epoch at 100 Hz
N_BASE_PER_CH = 23  # base features per channel (see features.py docstring)


def sine(freq, n_epochs, n_ch, amp=1.0, fs=FS, t=T):
    """[n, C, T] pure tone -- power should land in exactly one band."""
    tt = np.arange(t) / fs
    wave = amp * np.sin(2 * np.pi * freq * tt)
    return np.tile(wave, (n_epochs, n_ch, 1)).astype(np.float32)


class TestExtractFeatures:
    def test_shape_is_channels_times_23(self):
        x = sine(10, 4, 4)
        F, names = extract_features(x, fs=FS)
        assert F.shape == (4, 4 * N_BASE_PER_CH)
        assert len(names) == F.shape[1]

    def test_names_are_unique(self):
        _, names = extract_features(sine(10, 2, 4), fs=FS)
        assert len(set(names)) == len(names)

    def test_no_nan_or_inf(self):
        F, _ = extract_features(sine(10, 4, 4), fs=FS)
        assert np.isfinite(F).all()

    def test_deterministic(self):
        x = sine(10, 3, 4)
        a, _ = extract_features(x, fs=FS)
        b, _ = extract_features(x, fs=FS)
        np.testing.assert_array_equal(a, b)

    def test_flat_signal_does_not_blow_up(self):
        """All-zero channels occur in this cohort (dropped/disconnected leads)."""
        F, _ = extract_features(np.zeros((2, 4, T), np.float32), fs=FS)
        assert np.isfinite(F).all()

    @pytest.mark.parametrize("band_name,freq", [
        ("delta", 2.0), ("theta", 6.0), ("alpha", 10.0), ("beta", 22.0)])
    def test_relative_band_power_peaks_in_the_matching_band(self, band_name, freq):
        """A pure tone must put most relative power in the band containing it.

        This is the check that would catch a wrong `fs`, a bad Welch window, or a
        band-edge indexing slip -- the failure modes that silently degrade staging.
        """
        F, names = extract_features(sine(freq, 2, 1), fs=FS)
        rel = {b: F[0, names.index("%s_rel_c0" % b)] for b, _, _ in BANDS}
        assert rel[band_name] == pytest.approx(max(rel.values()))
        assert rel[band_name] > 0.5

    def test_amplitude_features_scale_with_amplitude(self):
        small, _ = extract_features(sine(10, 1, 1, amp=1.0), fs=FS)
        large, names = extract_features(sine(10, 1, 1, amp=3.0), fs=FS)
        for feat in ("std_c0", "rms_c0", "ptp_c0"):
            i = names.index(feat)
            assert large[0, i] == pytest.approx(3.0 * small[0, i], rel=1e-3)

    def test_relative_band_power_sums_to_about_one(self):
        F, names = extract_features(sine(10, 1, 1), fs=FS)
        total = sum(F[0, names.index("%s_rel_c0" % b)] for b, _, _ in BANDS)
        # BANDS covers 0.5-30 Hz of a 0-50 Hz spectrum, and sigma overlaps beta,
        # so this is a sanity bound rather than an exact identity.
        assert 0.0 < total <= 1.5


class TestEventFeatures:
    def test_shape_and_names(self):
        x = np.zeros((5, 7, T), np.float32)
        F, names = event_features(x, fs=FS)
        assert F.shape[0] == 5
        assert F.shape[1] == len(names)
        assert len(set(names)) == len(names)

    def test_covers_eeg_eog_and_emg(self):
        _, names = event_features(np.zeros((1, 7, T), np.float32), fs=FS)
        assert sum("spindle" in n for n in names) == 12   # 3 features x 4 EEG channels
        assert sum(n.startswith("sw_") for n in names) == 8
        assert sum(n.startswith("eog") for n in names) == 4
        assert sum(n.startswith("emg") for n in names) == 3

    def test_emg_tonic_level_tracks_muscle_amplitude(self):
        """EMG log-RMS is the REM-atonia discriminator -- it must be monotone."""
        rs = np.random.default_rng(0)
        quiet = np.zeros((1, 7, T), np.float32)
        loud = np.zeros((1, 7, T), np.float32)
        quiet[0, 6] = rs.normal(0, 0.1, T)
        loud[0, 6] = rs.normal(0, 5.0, T)
        fq, names = event_features(quiet, fs=FS)
        fl, _ = event_features(loud, fs=FS)
        i = names.index("emg_logrms")
        assert fl[0, i] > fq[0, i]

    def test_eog_movement_energy_tracks_eye_movement(self):
        rs = np.random.default_rng(1)
        still = np.zeros((1, 7, T), np.float32)
        moving = np.zeros((1, 7, T), np.float32)
        moving[0, 4] = rs.normal(0, 3.0, T)
        fs_, names = event_features(still, fs=FS)
        fm, _ = event_features(moving, fs=FS)
        i = names.index("eog_mov_c4")
        assert fm[0, i] > fs_[0, i]


class TestExtractFeaturesV2:
    def test_total_width_is_188(self):
        """The 188 here is load-bearing: MMFeatureNet(n_eeg=188) and the ablation
        masks in mmnet_repro.eeg_modality_masks() both hard-code it."""
        F, names = extract_features_v2(np.zeros((2, 7, T), np.float32), fs=FS)
        assert F.shape == (2, 188)
        assert len(names) == 188

    def test_is_base_features_then_event_features(self):
        x = np.zeros((2, 7, T), np.float32)
        base, base_names = extract_features(x, fs=FS)
        F, names = extract_features_v2(x, fs=FS)
        np.testing.assert_allclose(F[:, :base.shape[1]], base, rtol=1e-6)
        assert names[:len(base_names)] == base_names

    def test_no_nan_on_degenerate_input(self):
        F, _ = extract_features_v2(np.zeros((1, 7, T), np.float32), fs=FS)
        assert np.isfinite(F).all()
