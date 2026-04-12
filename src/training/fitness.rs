// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-12

//! GA-compatible fitness scoring for trained agents.
//!
//! Provides a scalar fitness in `[0, 1]` computed from win/draw/loss
//! rates and max curriculum depth reached. Designed for use with
//! genetic algorithms (roulette, tournament, rank selection).

const WEIGHT_PERFORMANCE: f64 = 0.55;
const WEIGHT_DEPTH: f64 = 0.40;
const WEIGHT_BALANCE: f64 = 0.05;
const MAX_DEPTH_NORMALIZER: f64 = 8.0;

/// GA-compatible fitness score for a trained agent.
///
/// All components are normalized to `[0, 1]`. The combined score
/// uses weights that prioritize performance (55%) and depth (40%)
/// over balance (5%).
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fitness {
    /// `win_rate + draw_rate`, in `[0, 1]`.
    pub performance: f64,
    /// `(max_depth - 1) / 8`, in `[0, 1]`.
    pub depth_score: f64,
    /// `1 - |win_rate - draw_rate|`, in `[0, 1]`.
    pub balance: f64,
}

impl Fitness {
    /// Computes a Fitness instance from scoring results.
    ///
    /// # Parameters
    /// * `win_rate` - Fraction of games won, in `[0, 1]`.
    /// * `draw_rate` - Fraction of games drawn, in `[0, 1]`.
    /// * `max_depth` - Highest curriculum depth reached, in `[1, 9]`.
    ///
    /// Returns `Fitness { performance: 0, depth_score: 0, balance: 0 }`
    /// if any rate input is NaN or infinite.
    pub fn from_scores(win_rate: f64, draw_rate: f64, max_depth: usize) -> Self {
        if !win_rate.is_finite() || !draw_rate.is_finite() {
            return Self {
                performance: 0.0,
                depth_score: 0.0,
                balance: 0.0,
            };
        }
        let performance = (win_rate + draw_rate).clamp(0.0, 1.0);
        let depth_clamped = max_depth.saturating_sub(1) as f64;
        let depth_score = (depth_clamped / MAX_DEPTH_NORMALIZER).clamp(0.0, 1.0);
        let balance = (1.0 - (win_rate - draw_rate).abs()).clamp(0.0, 1.0);
        Self {
            performance,
            depth_score,
            balance,
        }
    }

    /// Weighted combined score: `0.55 * perf + 0.40 * depth + 0.05 * balance`.
    ///
    /// Returns a value in `[0, 1]`. Higher is better.
    #[must_use]
    pub fn combined(&self) -> f64 {
        WEIGHT_PERFORMANCE * self.performance
            + WEIGHT_DEPTH * self.depth_score
            + WEIGHT_BALANCE * self.balance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fitness_optimal_perfect_50_50_d9_equals_1_0() {
        let fit = Fitness::from_scores(0.5, 0.5, 9);
        assert!(
            (fit.combined() - 1.0).abs() < 1e-6,
            "Expected 1.0, got {}",
            fit.combined()
        );
    }

    #[test]
    fn test_fitness_optimal_functional_0_99_d9_close_to_0_945() {
        let fit = Fitness::from_scores(0.0, 0.99, 9);
        let combined = fit.combined();
        // Formula: 0.55 * 0.99 + 0.40 * 1.0 + 0.05 * (1 - 0.99) = 0.5445 + 0.40 + 0.0005 = 0.945
        assert!(
            (combined - 0.945).abs() < 1e-3,
            "Expected ~0.945, got {}",
            combined
        );
    }

    #[test]
    fn test_fitness_perfect_d7_equals_0_900() {
        let fit = Fitness::from_scores(0.5, 0.5, 7);
        let combined = fit.combined();
        // Formula: 0.55 * 1.0 + 0.40 * 0.75 + 0.05 * 1.0 = 0.55 + 0.30 + 0.05 = 0.90
        assert!(
            (combined - 0.90).abs() < 1e-6,
            "Expected 0.90, got {}",
            combined
        );
    }

    #[test]
    fn test_fitness_collapsed_d6_100_loss() {
        let fit = Fitness::from_scores(0.0, 0.0, 6);
        let combined = fit.combined();
        // Formula: 0.55 * 0.0 + 0.40 * 0.625 + 0.05 * 1.0 = 0.0 + 0.25 + 0.05 = 0.30
        assert!(
            (combined - 0.30).abs() < 1e-6,
            "Expected 0.30, got {}",
            combined
        );
    }

    #[test]
    fn test_fitness_stalled_d2() {
        let fit = Fitness::from_scores(0.5, 0.0, 2);
        let combined = fit.combined();
        // Formula: 0.55 * 0.5 + 0.40 * 0.125 + 0.05 * 0.5 = 0.275 + 0.05 + 0.025 = 0.35
        assert!(
            (combined - 0.35).abs() < 1e-6,
            "Expected 0.35, got {}",
            combined
        );
    }

    #[test]
    fn test_fitness_offensive_bias_penalized() {
        let unbalanced = Fitness::from_scores(0.9, 0.0, 9).combined();
        let balanced = Fitness::from_scores(0.45, 0.45, 9).combined();
        assert!(
            balanced > unbalanced,
            "Balanced should score higher than unbalanced: {} vs {}",
            balanced,
            unbalanced
        );
    }

    #[test]
    fn test_fitness_bounded_0_1_for_all_valid_inputs() {
        for w_tenths in 0..=10 {
            for d_tenths in 0..=(10 - w_tenths) {
                for depth in 1..=9 {
                    let w = (w_tenths as f64) / 10.0;
                    let d = (d_tenths as f64) / 10.0;
                    let combined = Fitness::from_scores(w, d, depth).combined();
                    assert!(
                        (0.0..=1.0).contains(&combined),
                        "Out of bounds: w={}, d={}, depth={}, combined={}",
                        w,
                        d,
                        depth,
                        combined
                    );
                }
            }
        }
    }

    #[test]
    fn test_fitness_ordering_d9_functional_beats_d7_perfect() {
        let d9_functional = Fitness::from_scores(0.0, 0.99, 9).combined();
        let d7_perfect = Fitness::from_scores(0.5, 0.5, 7).combined();
        assert!(
            d9_functional > d7_perfect,
            "D=9 functional ({}) should beat D=7 perfect ({})",
            d9_functional,
            d7_perfect
        );
    }

    #[test]
    fn test_fitness_nan_input_returns_zero() {
        let fit = Fitness::from_scores(f64::NAN, 0.5, 9);
        assert_eq!(fit.combined(), 0.0);
        let fit2 = Fitness::from_scores(0.5, f64::NAN, 9);
        assert_eq!(fit2.combined(), 0.0);
    }

    #[test]
    fn test_weights_sum_to_one() {
        let sum = WEIGHT_PERFORMANCE + WEIGHT_DEPTH + WEIGHT_BALANCE;
        assert!(
            (sum - 1.0).abs() < f64::EPSILON,
            "Weights must sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_fitness_depth_1_contributes_zero() {
        let fit = Fitness::from_scores(1.0, 0.0, 1);
        // depth_score = (1-1)/8 = 0
        assert!((fit.depth_score - 0.0).abs() < 1e-9);
    }
}
