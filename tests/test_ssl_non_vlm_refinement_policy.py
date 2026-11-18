import unittest

from src.utils.ssl_non_vlm_refinement_policy import (
    adaptive_class_threshold,
    linear_quantile,
)


class TestLinearQuantile(unittest.TestCase):

    def test_linear_interpolation(self):
        result = linear_quantile(
            [0.0, 1.0, 2.0, 3.0],
            0.25,
        )
        self.assertAlmostEqual(result, 0.75)

    def test_single_value(self):
        result = linear_quantile(
            [0.8],
            0.80,
        )
        self.assertAlmostEqual(result, 0.8)

    def test_empty_values_are_rejected(self):
        with self.assertRaises(ValueError):
            linear_quantile([], 0.80)

    def test_non_finite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            linear_quantile(
                [0.8, float("nan")],
                0.80,
            )


class TestAdaptiveClassThreshold(unittest.TestCase):

    def test_insufficient_history_uses_global_threshold(self):
        result = adaptive_class_threshold(
            [0.80] * 31
        )
        self.assertAlmostEqual(result, 0.90)

    def test_exact_minimum_history_uses_history_quantile(self):
        result = adaptive_class_threshold(
            [0.80] * 32
        )

        history_weight = 32 / (32 + 30)

        expected = (
            history_weight * 0.80
            + (1.0 - history_weight) * 0.90
        )

        self.assertAlmostEqual(
            result,
            expected,
        )

    def test_minimum_threshold_is_enforced(self):
        result = adaptive_class_threshold(
            [0.0] * 100
        )
        self.assertAlmostEqual(result, 0.70)

    def test_maximum_threshold_is_enforced(self):
        result = adaptive_class_threshold(
            [1.0] * 100
        )
        self.assertAlmostEqual(result, 0.90)

    def test_non_finite_score_is_rejected_even_with_short_history(self):
        scores = (
            [0.80] * 30
            + [float("nan")]
        )

        with self.assertRaises(ValueError):
            adaptive_class_threshold(
                scores
            )

    def test_score_above_one_is_rejected(self):
        with self.assertRaises(ValueError):
            adaptive_class_threshold(
                [0.80, 1.01]
            )

    def test_score_below_zero_is_rejected(self):
        with self.assertRaises(ValueError):
            adaptive_class_threshold(
                [0.80, -0.01]
            )

    def test_invalid_target_quantile_is_rejected_with_short_history(self):
        with self.assertRaises(ValueError):
            adaptive_class_threshold(
                [0.80],
                target_quantile=1.1,
            )


if __name__ == "__main__":
    unittest.main()
