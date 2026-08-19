"""Tests for the two network definitions (code/mm_feature_net.py, multimodal_net.py).

The claims these pin down are the ones the ablation table depends on:

  * the fusion switch actually changes what the model can see, and
  * `eeg_only` is genuinely cardio-free -- the apnea head reads a direct cardio
    pathway, so "remove the cardio modality" is only honest if that pathway is
    zeroed too. mm_feature_net does this explicitly; this test is what keeps it
    true if the forward pass is ever refactored.

Skipped as a group when torch is not installed, so the rest of the suite still
runs in a light environment.
"""
import csv
import os

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from mm_feature_net import MMFeatureNet          # noqa: E402
from multimodal_net import MultimodalSleepNet    # noqa: E402

FUSIONS = ["cross", "concat", "eeg_only"]
B, L = 2, 20
N_EEG, N_CARD = 188, 14


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(0)


def feats(b=B, ln=L):
    return torch.randn(b, ln, N_EEG), torch.randn(b, ln, N_CARD)


class TestMMFeatureNet:
    @pytest.mark.parametrize("fusion", FUSIONS)
    def test_output_shapes(self, fusion):
        model = MMFeatureNet(fusion=fusion)
        stage, apnea = model(*feats())
        assert stage.shape == (B, L, 5)     # 5 sleep stages per epoch
        assert apnea.shape == (B, L)        # binary respiratory logit per epoch

    @pytest.mark.parametrize("fusion", FUSIONS)
    def test_outputs_are_finite(self, fusion):
        stage, apnea = MMFeatureNet(fusion=fusion)(*feats())
        assert torch.isfinite(stage).all() and torch.isfinite(apnea).all()

    @pytest.mark.parametrize("ln", [1, 5, 20, 37])
    def test_accepts_any_sequence_length(self, ln):
        """Inference pads to a multiple of L but the last window can be short."""
        stage, apnea = MMFeatureNet(fusion="concat")(*feats(ln=ln))
        assert stage.shape == (B, ln, 5)

    @pytest.mark.parametrize("fusion,millions", [
        ("cross", 0.856), ("concat", 0.773), ("eeg_only", 0.765)])
    def test_parameter_counts_are_pinned(self, fusion, millions):
        """Pin each variant's size so an architecture change has to be deliberate."""
        n = sum(p.numel() for p in MMFeatureNet(fusion=fusion).parameters())
        assert n / 1e6 == pytest.approx(millions, abs=0.001)

    def test_reported_parameter_count_matches_the_headline_architecture(self):
        """Ties results/headline_metrics.csv to the actual model.

        This value was 0.86 M, which is the *cross-attention* variant's size. The
        headline is concat (mmnet_repro runs run_config("headline_concat",
        fusion="concat")), so the reported figure belonged to a model that is an
        ablation, not the headline -- the one number the attention -> concat
        switch in VERIFICATION_LOG row 4 missed. Corrected to 0.773 M and pinned
        here so a future architecture change cannot silently stale the table.
        """
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "results", "headline_metrics.csv")
        with open(csv_path, newline="", encoding="utf-8") as fh:
            reported = {r["metric"]: r["value"] for r in csv.DictReader(fh)}
        measured = sum(p.numel() for p in MMFeatureNet(fusion="concat").parameters()) / 1e6
        assert float(reported["params_million"]) == pytest.approx(measured, abs=0.001)

    def test_both_benchmark_rows_share_the_concat_architecture(self):
        """staging_benchmark.csv's two MM-Net rows must report the same size.

        "neural only" is not a different network: the engine cache records it as
        fusion=concat with card_drop=['all'], i.e. the same architecture with the
        cardio feature columns zeroed. Different parameter counts on those two
        rows would mean the ablation changed the model rather than the input.
        """
        csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "results", "staging_benchmark.csv")
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = {r["model"]: r for r in csv.DictReader(fh)}
        measured = sum(p.numel() for p in MMFeatureNet(fusion="concat").parameters()) / 1e6
        for name in ("MM-Net (neural only)", "MM-Net (neural+cardio)"):
            assert float(rows[name]["params_M"]) == pytest.approx(measured, abs=0.001), name

    def test_cross_fusion_has_more_parameters_than_concat(self):
        cross = sum(p.numel() for p in MMFeatureNet(fusion="cross").parameters())
        concat = sum(p.numel() for p in MMFeatureNet(fusion="concat").parameters())
        assert cross > concat

    def test_eeg_only_ignores_the_cardio_input_entirely(self):
        """The load-bearing ablation guarantee: with fusion='eeg_only', changing
        the cardio features must not move either output by even a float."""
        model = MMFeatureNet(fusion="eeg_only").eval()
        feeg, fcard = feats()
        with torch.no_grad():
            a_stage, a_apnea = model(feeg, fcard)
            b_stage, b_apnea = model(feeg, torch.randn_like(fcard) * 100.0)
        torch.testing.assert_close(a_stage, b_stage)
        torch.testing.assert_close(a_apnea, b_apnea)

    @pytest.mark.parametrize("fusion", ["cross", "concat"])
    def test_cardio_input_does_change_the_apnea_head(self, fusion):
        """Converse of the above -- if cardio made no difference in the full
        model, the modality ablation would be measuring nothing."""
        model = MMFeatureNet(fusion=fusion).eval()
        feeg, fcard = feats()
        with torch.no_grad():
            a = model(feeg, fcard)[1]
            b = model(feeg, torch.randn_like(fcard) * 100.0)[1]
        assert not torch.allclose(a, b)

    def test_eval_mode_is_deterministic(self):
        """Dropout is 0.3; inference must not be stochastic or the cached
        per-fold metrics would not be comparable between runs."""
        model = MMFeatureNet(fusion="concat").eval()
        feeg, fcard = feats()
        with torch.no_grad():
            torch.testing.assert_close(model(feeg, fcard)[0], model(feeg, fcard)[0])

    def test_gradients_reach_both_encoders(self):
        model = MMFeatureNet(fusion="cross")
        stage, apnea = model(*feats())
        (stage.sum() + apnea.sum()).backward()
        for name, enc in (("eeg", model.eeg_enc), ("card", model.card_enc)):
            grads = [p.grad for p in enc.parameters() if p.grad is not None]
            assert grads, "no gradient reached the %s encoder" % name
            assert any(g.abs().sum() > 0 for g in grads), \
                "%s encoder received only zero gradients" % name

    def test_rejects_wrong_feature_width(self):
        model = MMFeatureNet(fusion="concat")
        with pytest.raises(RuntimeError):
            model(torch.randn(B, L, N_EEG + 1), torch.randn(B, L, N_CARD))


class TestMultimodalSleepNet:
    """The raw-signal variant (reported at 0.655 in staging_benchmark.csv)."""

    @pytest.mark.parametrize("fusion", FUSIONS)
    def test_output_shapes(self, fusion):
        model = MultimodalSleepNet(fusion=fusion)
        eeg = torch.randn(1, 4, 7, 3000)     # 100 Hz, 30 s
        card = torch.randn(1, 4, 7, 750)     # 25 Hz, 30 s
        stage, apnea = model(eeg, card)
        assert stage.shape == (1, 4, 5)
        assert apnea.shape == (1, 4)

    def test_eeg_only_ignores_cardio(self):
        model = MultimodalSleepNet(fusion="eeg_only").eval()
        eeg = torch.randn(1, 2, 7, 3000)
        with torch.no_grad():
            a = model(eeg, torch.randn(1, 2, 7, 750))[0]
            b = model(eeg, torch.randn(1, 2, 7, 750) * 50.0)[0]
        torch.testing.assert_close(a, b)

    def test_parameter_count_is_about_the_reported_0_95M(self):
        n = sum(p.numel() for p in MultimodalSleepNet(fusion="cross").parameters())
        assert 0.8e6 < n < 1.1e6
