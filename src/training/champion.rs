// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-12

//! Champion search: iterates N training sessions, scores candidates,
//! persists the best individual found.

use std::error::Error;
use std::fs;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pc_rl_core::pc_actor_critic::PcActorCritic;
use pc_rl_core::serializer::{load_agent, save_agent};
use pc_rl_core::CpuLinAlg;
use rand::Rng;

use crate::training::continuous::ContinuousTrainer;
use crate::training::fitness::{score_vs_minimax, Fitness};
use crate::utils::config::AppConfig;

/// Result of a complete champion search.
#[derive(Debug, Clone)]
pub struct ChampionResult {
    /// Best confirmed fitness across all iterations.
    pub champion_fitness: f64,
    /// Max depth of the champion.
    pub champion_depth: usize,
    /// Iteration that produced the champion (1-indexed).
    pub champion_iteration: usize,
    /// Seed of the champion's session.
    pub champion_seed: u64,
    /// Summary of all iterations run.
    pub iterations: Vec<IterationSummary>,
}

/// Summary of a single champion search iteration.
#[derive(Debug, Clone)]
pub struct IterationSummary {
    /// Iteration index (1-indexed).
    pub iteration: usize,
    /// Random seed used for this training session.
    pub seed: u64,
    /// Peak fitness observed during the session (via running scoring).
    pub peak_fitness: f64,
    /// Depth reached at the end of the session.
    pub final_depth: usize,
    /// Depth reached at the peak fitness point.
    pub peak_depth: usize,
    /// Whether this iteration replaced the champion.
    pub replaced_champion: bool,
}

/// Champion search driver.
///
/// Runs `n_iterations` independent training sessions, evaluates each
/// candidate with running and final scoring rounds, and persists the
/// best-scoring snapshot to `champion.output_path`.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use std::sync::atomic::AtomicBool;
/// use pc_tictactoe::training::champion::ChampionFinder;
/// use pc_tictactoe::utils::config::AppConfig;
///
/// let config = AppConfig::default();
/// let stop = Arc::new(AtomicBool::new(false));
/// let mut finder = ChampionFinder::new(config, stop);
/// let result = finder.find().expect("champion search failed");
/// println!("Champion depth: {}", result.champion_depth);
/// ```
pub struct ChampionFinder {
    base_config: AppConfig,
    stop_flag: Arc<AtomicBool>,
}

impl ChampionFinder {
    /// Creates a new `ChampionFinder` from the base config.
    ///
    /// # Parameters
    /// * `base_config` - Application configuration (cloned per iteration).
    /// * `stop_flag` - Atomic flag; set to `true` to abort after the current iteration.
    #[must_use]
    pub fn new(base_config: AppConfig, stop_flag: Arc<AtomicBool>) -> Self {
        Self {
            base_config,
            stop_flag,
        }
    }

    /// Runs the champion search and returns the result.
    ///
    /// Each iteration: (1) picks a random seed, (2) trains a fresh agent
    /// using [`ContinuousTrainer`], (3) evaluates peak fitness via running
    /// scoring with `assessment_games_running`, (4) confirms the peak with
    /// a full scoring pass using `assessment_games_final`, (5) replaces
    /// the champion if the confirmed fitness is higher.
    ///
    /// # Errors
    ///
    /// Returns an error if agent construction, file I/O, or snapshot
    /// save/load fails.
    #[must_use = "call find() and inspect the returned ChampionResult"]
    pub fn find(&mut self) -> Result<ChampionResult, Box<dyn Error>> {
        let champion_cfg = self.base_config.champion.clone();
        let mut champion_fitness = -1.0_f64;
        let mut champion_depth = 0_usize;
        let mut champion_iteration = 0_usize;
        let mut champion_seed = 0_u64;
        let mut iterations: Vec<IterationSummary> = Vec::new();

        let mut rng = rand::thread_rng();

        for iter_idx in 0..champion_cfg.n_iterations {
            if self.stop_flag.load(Ordering::Acquire) {
                break;
            }

            let iteration = iter_idx + 1;
            let seed: u64 = rng.gen();
            eprintln!(
                "Iter {iteration}/{n} seed={seed}...",
                n = champion_cfg.n_iterations
            );

            let mut config = self.base_config.clone();
            config.training.seed = seed;

            let agent_cfg = config.to_agent_config()?;
            let agent = PcActorCritic::new(CpuLinAlg::new(), agent_cfg, seed)?;
            let mut trainer = ContinuousTrainer::new(agent, &config, self.stop_flag.clone());

            let mut peak_fitness = 0.0_f64;
            let mut peak_depth = 1_usize;
            let mut has_snapshot = false;
            // Use the output path stem to avoid collisions when multiple
            // ChampionFinder instances run in parallel (e.g., during tests).
            let output_stem = std::path::Path::new(&champion_cfg.output_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("champion");
            let snapshot_path = format!("tmp_{output_stem}_peak_{iteration}.json");

            while !trainer.should_stop() {
                trainer.run_single_episode_pub();

                if trainer
                    .episode_count()
                    .is_multiple_of(champion_cfg.assessment_interval)
                {
                    let (w, d, _l) = score_vs_minimax(
                        trainer.agent_mut(),
                        champion_cfg.assessment_depth,
                        champion_cfg.assessment_games_running,
                    );
                    let fitness = Fitness::from_scores(w, d, trainer.current_depth()).combined();

                    if fitness > peak_fitness {
                        peak_fitness = fitness;
                        peak_depth = trainer.current_depth();
                        save_agent(
                            trainer.agent(),
                            &snapshot_path,
                            trainer.episode_count(),
                            None,
                        )?;
                        has_snapshot = true;
                    }
                }
            }

            let final_depth = trainer.current_depth();
            let mut replaced = false;

            if has_snapshot && peak_depth >= champion_cfg.min_depth_filter {
                // Load the snapshot and run full-accuracy scoring
                let (mut snapshot_agent, _) = load_agent(&snapshot_path, CpuLinAlg::new())?;
                let (w, d, _l) = score_vs_minimax(
                    &mut snapshot_agent,
                    champion_cfg.assessment_depth,
                    champion_cfg.assessment_games_final,
                );
                let confirmed_fitness = Fitness::from_scores(w, d, peak_depth).combined();

                eprintln!(
                    "  peak_fitness={peak_fitness:.4} depth={peak_depth} confirmed={confirmed_fitness:.4}"
                );

                if confirmed_fitness > champion_fitness {
                    champion_fitness = confirmed_fitness;
                    champion_depth = peak_depth;
                    champion_iteration = iteration;
                    champion_seed = seed;
                    fs::rename(&snapshot_path, &champion_cfg.output_path)?;
                    replaced = true;
                    eprintln!(
                        "  NEW CHAMPION: fitness={confirmed_fitness:.4} depth={peak_depth} saved to {}",
                        champion_cfg.output_path
                    );
                } else if fs::metadata(&snapshot_path).is_ok() {
                    fs::remove_file(&snapshot_path)?;
                }
            } else if has_snapshot && fs::metadata(&snapshot_path).is_ok() {
                fs::remove_file(&snapshot_path)?;
            }

            iterations.push(IterationSummary {
                iteration,
                seed,
                peak_fitness,
                final_depth,
                peak_depth,
                replaced_champion: replaced,
            });
        }

        Ok(ChampionResult {
            champion_fitness,
            champion_depth,
            champion_iteration,
            champion_seed,
            iterations,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tiny_config() -> AppConfig {
        let mut config = AppConfig::default();
        config.training.episodes = 50;
        config.continuous.max_episodes = 50;
        config.champion.n_iterations = 2;
        config.champion.assessment_games_running = 5;
        config.champion.assessment_games_final = 10;
        config.champion.assessment_interval = 25;
        config.champion.assessment_depth = 1;
        config.champion.output_path = "test_champion_ut.json".to_string();
        config.training.log_interval = 0;
        config
    }

    #[test]
    fn test_champion_finder_runs_n_iterations() {
        let config = make_tiny_config();
        let stop = Arc::new(AtomicBool::new(false));
        let mut finder = ChampionFinder::new(config, stop);
        let result = finder.find().expect("find should succeed");
        assert_eq!(result.iterations.len(), 2);
        // Cleanup
        let _ = fs::remove_file("test_champion_ut.json");
    }

    #[test]
    fn test_champion_finder_respects_stop_flag() {
        let mut config = make_tiny_config();
        config.champion.output_path = "test_champion_stop.json".to_string();
        config.champion.n_iterations = 100;
        let stop = Arc::new(AtomicBool::new(true));
        let mut finder = ChampionFinder::new(config, stop);
        let result = finder.find().expect("find should succeed");
        assert_eq!(result.iterations.len(), 0);
        let _ = fs::remove_file("test_champion_stop.json");
    }

    #[test]
    fn test_champion_finder_saves_output_file_when_depth_reached() {
        let mut config = make_tiny_config();
        config.champion.output_path = "test_champion_saved.json".to_string();
        let stop = Arc::new(AtomicBool::new(false));
        let mut finder = ChampionFinder::new(config, stop);
        let _ = finder.find().expect("find should succeed");
        // Even short sessions produce at least a peak at depth 1, which
        // passes min_depth_filter=0 -> the champion file should exist.
        assert!(std::path::Path::new("test_champion_saved.json").exists());
        let _ = fs::remove_file("test_champion_saved.json");
    }
}
