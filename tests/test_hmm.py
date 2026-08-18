"""Tests for the Viterbi stage smoother (mmnet_repro._hmm).

Every staging number in the paper is post-HMM: the network emits per-epoch
probabilities and this decoder imposes the sleep-transition structure on top. A
bug here (an off-by-one in the backtrace, argmax over the wrong axis) would move
the headline accuracy without raising an error anywhere, so it is worth checking
against brute force.
"""
import itertools

import numpy as np
import pytest

from mmnet_repro import NC, _hmm


def brute_force(A_log, pi_log, logp):
    """Exhaustive best path -- exponential, only usable for tiny T."""
    T = logp.shape[0]
    best, best_score = None, -np.inf
    for path in itertools.product(range(NC), repeat=T):
        score = pi_log[path[0]] + logp[0, path[0]]
        for t in range(1, T):
            score += A_log[path[t - 1], path[t]] + logp[t, path[t]]
        if score > best_score:
            best_score, best = score, path
    return np.array(best)


def uniform_params():
    A = np.full((NC, NC), 1.0 / NC)
    pi = np.full(NC, 1.0 / NC)
    return np.log(A), np.log(pi)


class TestViterbi:
    def test_output_shape_and_range(self):
        A_log, pi_log = uniform_params()
        logp = np.log(np.full((12, NC), 1.0 / NC))
        path = _hmm(A_log, pi_log, logp)
        assert path.shape == (12,)
        assert path.min() >= 0 and path.max() < NC

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_matches_brute_force_on_short_sequences(self, seed):
        rs = np.random.default_rng(seed)
        A = rs.random((NC, NC)) + 0.1
        A /= A.sum(1, keepdims=True)
        pi = rs.random(NC) + 0.1
        pi /= pi.sum()
        emissions = rs.random((5, NC)) + 0.1
        emissions /= emissions.sum(1, keepdims=True)
        path = _hmm(np.log(A), np.log(pi), np.log(emissions))
        np.testing.assert_array_equal(path, brute_force(np.log(A), np.log(pi),
                                                        np.log(emissions)))

    def test_with_uniform_transitions_it_follows_the_emissions(self):
        """No transition preference -> the decoder must reduce to per-epoch argmax."""
        A_log, pi_log = uniform_params()
        rs = np.random.default_rng(7)
        emissions = rs.random((30, NC))
        emissions /= emissions.sum(1, keepdims=True)
        path = _hmm(A_log, pi_log, np.log(emissions))
        np.testing.assert_array_equal(path, emissions.argmax(1))

    def test_strong_self_transitions_suppress_a_single_spike(self):
        """Sleep stages persist for minutes; a one-epoch flicker against a
        confident run should be smoothed away. This is what the HMM is for."""
        A = np.full((NC, NC), 0.001)
        np.fill_diagonal(A, 1.0 - 0.001 * (NC - 1))
        pi = np.full(NC, 1.0 / NC)
        emissions = np.full((15, NC), 0.02)
        emissions[:, 2] = 0.92          # a confident run of N2
        emissions[7, 2] = 0.45          # one wobbly epoch
        emissions[7, 4] = 0.47          # ...leaning REM
        emissions /= emissions.sum(1, keepdims=True)
        assert emissions.argmax(1)[7] == 4      # raw argmax would flip to REM
        path = _hmm(np.log(A), np.log(pi), np.log(emissions))
        assert (path == 2).all()                # the decoder holds N2

    def test_prior_decides_when_emissions_are_uninformative(self):
        A = np.full((NC, NC), 1.0 / NC)
        pi = np.full(NC, 0.01)
        pi[3] = 1.0 - 0.01 * (NC - 1)
        emissions = np.full((6, NC), 1.0 / NC)
        path = _hmm(np.log(A), np.log(pi), np.log(emissions))
        assert path[0] == 3

    def test_single_epoch_sequence(self):
        A_log, pi_log = uniform_params()
        emissions = np.full((1, NC), 0.1)
        emissions[0, 1] = 0.6
        path = _hmm(A_log, pi_log, np.log(emissions))
        assert path.tolist() == [1]

    def test_deterministic(self):
        rs = np.random.default_rng(11)
        A_log, pi_log = uniform_params()
        emissions = rs.random((40, NC))
        emissions /= emissions.sum(1, keepdims=True)
        a = _hmm(A_log, pi_log, np.log(emissions))
        b = _hmm(A_log, pi_log, np.log(emissions))
        np.testing.assert_array_equal(a, b)
