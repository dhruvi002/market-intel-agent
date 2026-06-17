"""Tests for mia_eval.stats — bootstrap confidence intervals."""

from __future__ import annotations

from mia_eval.stats import MeanCI, bootstrap_ci, mean_ci


class TestBootstrapCI:
    def test_empty_returns_zero_interval(self):
        ci = bootstrap_ci([])
        assert ci.mean == 0.0 and ci.low == 0.0 and ci.high == 0.0 and ci.n == 0

    def test_single_value_degenerate(self):
        ci = bootstrap_ci([0.7])
        assert ci.mean == 0.7 and ci.low == 0.7 and ci.high == 0.7 and ci.n == 1

    def test_mean_is_sample_mean(self):
        vals = [0.2, 0.4, 0.6, 0.8]
        ci = bootstrap_ci(vals)
        assert abs(ci.mean - 0.5) < 1e-9

    def test_interval_brackets_mean(self):
        vals = [0.1, 0.3, 0.5, 0.7, 0.9]
        ci = bootstrap_ci(vals)
        assert ci.low <= ci.mean <= ci.high

    def test_deterministic_with_seed(self):
        vals = [0.1, 0.5, 0.9, 0.3, 0.7]
        a = bootstrap_ci(vals, seed=123)
        b = bootstrap_ci(vals, seed=123)
        assert a.low == b.low and a.high == b.high

    def test_seed_none_is_nondeterministic_path(self):
        # seed=None must not raise and must still bracket the mean.
        vals = [0.1, 0.5, 0.9, 0.3, 0.7]
        ci = bootstrap_ci(vals, seed=None, n_resamples=2000)
        assert ci.low <= ci.mean <= ci.high

    def test_constant_values_zero_width(self):
        ci = bootstrap_ci([0.5, 0.5, 0.5, 0.5])
        assert ci.low == 0.5 and ci.high == 0.5

    def test_confidence_width_monotonic(self):
        vals = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        narrow = bootstrap_ci(vals, confidence=0.80, seed=7)
        wide = bootstrap_ci(vals, confidence=0.99, seed=7)
        assert (wide.high - wide.low) >= (narrow.high - narrow.low)

    def test_margin_property(self):
        import pytest

        ci = MeanCI(mean=0.5, low=0.4, high=0.65, confidence=0.95, n=10)
        assert ci.margin == pytest.approx(0.15)

    def test_str_format(self):
        ci = MeanCI(mean=0.742, low=0.681, high=0.803, confidence=0.95, n=10)
        assert str(ci) == "0.742 [0.681, 0.803]"

    def test_mean_ci_alias(self):
        ci = mean_ci([0.4, 0.6], seed=5)
        assert abs(ci.mean - 0.5) < 1e-9
