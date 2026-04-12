// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-04-12

//! Stress test: loads a champion, enables CL, runs against random-depth
//! opponents, logs fitness drift over time.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pc_rl_core::pc_actor_critic::PcActorCritic;
use pc_rl_core::serializer::{load_agent, save_agent};
use pc_rl_core::{CpuLinAlg, PlasticityState};
use rand::Rng;

use crate::env::minimax::MinimaxPlayer;
use crate::env::tictactoe::{GameResult, Player, TicTacToe};
use crate::training::fitness::{score_vs_minimax, Fitness};
use crate::utils::config::{AppConfig, StressTestSection};
use crate::utils::metrics::{GameOutcome, Metrics};

/// Threshold for delta classification: deltas within this range are `Stable`.
const STATUS_DELTA_THRESHOLD: f64 = 0.01;

/// Minimax depth used for scoring during stress test.
///
/// Fixed at 9 (perfect play) so that drift measurements are anchored to
/// the strongest possible opponent — this lets the CSV's fitness column
/// be compared across runs even when `opponent_depth_max` differs.
const STRESS_SCORING_DEPTH: usize = 9;

/// CSV header for the stress test log file.
const CSV_HEADER: &str = "episode,opponent_depths_seen,fitness,win_rate,draw_rate,loss_rate,delta_vs_baseline,delta_vs_previous,status,actor_state,actor_fast,actor_slow,critic_state,critic_fast,critic_slow,hysteresis_transitions";

/// Classification of a single scoring checkpoint vs the previous one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StressStatus {
    /// First measurement — no previous to compare against.
    Baseline,
    /// Fitness improved by more than [`STATUS_DELTA_THRESHOLD`].
    Improved,
    /// Fitness changed by at most [`STATUS_DELTA_THRESHOLD`].
    Stable,
    /// Fitness degraded by more than [`STATUS_DELTA_THRESHOLD`].
    Degraded,
}

impl fmt::Display for StressStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            StressStatus::Baseline => "BASELINE",
            StressStatus::Improved => "IMPROVED",
            StressStatus::Stable => "STABLE",
            StressStatus::Degraded => "DEGRADED",
        };
        f.write_str(s)
    }
}

/// Classifies a delta vs the previous scoring result.
///
/// # Parameters
/// * `delta_previous` — `current_fitness - previous_fitness`.
///
/// # Returns
/// [`StressStatus::Improved`] if delta > threshold,
/// [`StressStatus::Degraded`] if delta < -threshold,
/// [`StressStatus::Stable`] otherwise.
#[must_use]
pub fn classify_status(delta_previous: f64) -> StressStatus {
    if delta_previous > STATUS_DELTA_THRESHOLD {
        StressStatus::Improved
    } else if delta_previous < -STATUS_DELTA_THRESHOLD {
        StressStatus::Degraded
    } else {
        StressStatus::Stable
    }
}

/// Single row logged during the stress test.
#[derive(Debug, Clone)]
pub struct StressLogEntry {
    /// Episode number at the time of logging.
    pub episode: u64,
    /// Serialized depth histogram snapshot (e.g. `"D1:245 D3:198"`).
    pub depth_histogram_snapshot: String,
    /// Combined fitness score.
    pub fitness: f64,
    /// Win rate over assessment games.
    pub win_rate: f64,
    /// Draw rate over assessment games.
    pub draw_rate: f64,
    /// Loss rate over assessment games.
    pub loss_rate: f64,
    /// Fitness minus the baseline fitness at episode 0.
    pub delta_vs_baseline: f64,
    /// Fitness minus the fitness at the previous checkpoint.
    pub delta_vs_previous: f64,
    /// Classification relative to the previous checkpoint.
    pub status: StressStatus,
    /// Actor plasticity state: `'P'` (Plastic) or `'F'` (Frozen).
    pub actor_state: char,
    /// Actor fast EWMA value.
    pub actor_fast: f64,
    /// Actor slow EWMA value.
    pub actor_slow: f64,
    /// Critic plasticity state: `'P'` (Plastic) or `'F'` (Frozen).
    pub critic_state: char,
    /// Critic fast EWMA value.
    pub critic_fast: f64,
    /// Critic slow EWMA value.
    pub critic_slow: f64,
    /// Cumulative actor+critic hysteresis FROZEN↔PLASTIC transitions (M2 only).
    pub hysteresis_transitions: u64,
}

/// Final summary of a stress test run.
#[derive(Debug, Clone)]
pub struct StressResult {
    /// Fitness measured before any CL training begins.
    pub baseline_fitness: f64,
    /// Fitness at the last scoring checkpoint.
    pub final_fitness: f64,
    /// Maximum fitness observed across all checkpoints.
    pub max_fitness: f64,
    /// Minimum fitness observed across all checkpoints.
    pub min_fitness: f64,
    /// Number of checkpoints classified as `Improved`.
    pub improvements: u64,
    /// Number of checkpoints classified as `Stable`.
    pub stable: u64,
    /// Number of checkpoints classified as `Degraded`.
    pub degradations: u64,
    /// Total number of training episodes completed.
    pub total_episodes: u64,
    /// All logged entries (baseline + every assessment checkpoint).
    pub history: Vec<StressLogEntry>,
}

/// Renders the sorted opponent-depth histogram as `"D1:count D3:count ..."`.
///
/// Returns `"-"` for an empty histogram.
///
/// # Parameters
/// * `histogram` — Map from minimax depth to number of games played at that depth.
#[must_use]
pub fn format_depth_histogram(histogram: &BTreeMap<usize, u64>) -> String {
    if histogram.is_empty() {
        return "-".to_string();
    }
    histogram
        .iter()
        .map(|(d, c)| format!("D{d}:{c}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Stress test driver.
///
/// Loads a champion agent from disk, enables CL by running `step_masked`,
/// and periodically evaluates fitness against minimax depth-9 to detect drift.
/// All measurements are appended to a CSV log file.
///
/// Episodes alternate the agent between Player::One and Player::Two (matching
/// score_vs_minimax) so that drift measurements are not contaminated by an
/// untrained side policy.
///
/// Drift measurement isolates inference (`act` via `score_vs_minimax`) from
/// training (`step_masked` via `run_stress_episode`) by calling
/// `agent.reset_step()` around each scoring round. This ensures the CSV's
/// fitness column reflects training-side drift only, not scoring-side
/// contamination.
///
/// # Examples
///
/// ```no_run
/// use std::sync::{Arc, atomic::AtomicBool};
/// use pc_tictactoe::utils::config::AppConfig;
/// use pc_tictactoe::training::stress_test::StressTester;
///
/// let cfg = AppConfig::default();
/// let stress_cfg = cfg.stress_test.clone();
/// let stop = Arc::new(AtomicBool::new(false));
/// // Requires a valid champion file on disk:
/// // let result = StressTester::new(cfg, stress_cfg, stop).unwrap().run().unwrap();
/// ```
pub struct StressTester {
    agent: PcActorCritic,
    stress_config: StressTestSection,
    stop_flag: Arc<AtomicBool>,
    depth_histogram: BTreeMap<usize, u64>,
    hysteresis_transitions: u64,
    prev_actor_state: PlasticityState,
    prev_critic_state: PlasticityState,
    metrics: Metrics,
}

impl StressTester {
    /// Creates a new stress tester by loading the champion from disk.
    ///
    /// # Parameters
    /// * `base_config` — Application configuration (curriculum, agent hyper-params).
    /// * `stress_config` — Stress-test-specific settings (paths, intervals, depths).
    /// * `stop_flag` — Shared flag; set to `true` to interrupt the run gracefully.
    ///
    /// # Errors
    /// Returns an error if the champion file cannot be loaded.
    #[must_use = "call .run() to execute the stress test"]
    pub fn new(
        base_config: AppConfig,
        stress_config: StressTestSection,
        stop_flag: Arc<AtomicBool>,
    ) -> Result<Self, Box<dyn Error>> {
        let (agent, _metadata) = load_agent(&stress_config.champion_path, CpuLinAlg::new())?;
        let metrics = Metrics::new(base_config.curriculum.window_size);
        Ok(Self {
            agent,
            stress_config,
            stop_flag,
            depth_histogram: BTreeMap::new(),
            hysteresis_transitions: 0,
            prev_actor_state: PlasticityState::Plastic,
            prev_critic_state: PlasticityState::Plastic,
            metrics,
        })
    }

    /// Runs the stress test and returns the summary.
    ///
    /// Writes a CSV row for the baseline and every `assessment_interval` episodes.
    /// Saves the post-CL agent to `stress_config.output_agent_path` on completion.
    ///
    /// # Errors
    /// Returns an error on CSV write failures or agent save failures.
    #[must_use = "check the StressResult for drift analysis"]
    pub fn run(&mut self) -> Result<StressResult, Box<dyn Error>> {
        let file = File::create(&self.stress_config.log_path)?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "{CSV_HEADER}")?;

        // Baseline scoring
        let (bw, bd, bl) = score_vs_minimax(
            &mut self.agent,
            STRESS_SCORING_DEPTH,
            self.stress_config.assessment_games,
        );
        let baseline_fitness = Fitness::from_scores(bw, bd, STRESS_SCORING_DEPTH).combined();
        let mut prev_fitness = baseline_fitness;
        let mut max_fitness = baseline_fitness;
        let mut min_fitness = baseline_fitness;
        let mut improvements = 0_u64;
        let mut stable = 0_u64;
        let mut degradations = 0_u64;

        // Capture initial CL state
        let (a_state, a_fast, a_slow, c_state, c_fast, c_slow) =
            Self::capture_cl_state(&self.agent);
        self.prev_actor_state = Self::char_to_state(a_state);
        self.prev_critic_state = Self::char_to_state(c_state);

        let baseline_entry = StressLogEntry {
            episode: 0,
            depth_histogram_snapshot: format_depth_histogram(&self.depth_histogram),
            fitness: baseline_fitness,
            win_rate: bw,
            draw_rate: bd,
            loss_rate: bl,
            delta_vs_baseline: 0.0,
            delta_vs_previous: 0.0,
            status: StressStatus::Baseline,
            actor_state: a_state,
            actor_fast: a_fast,
            actor_slow: a_slow,
            critic_state: c_state,
            critic_fast: c_fast,
            critic_slow: c_slow,
            hysteresis_transitions: 0,
        };
        let mut history = vec![baseline_entry.clone()];
        Self::write_csv_row(&mut writer, &baseline_entry)?;
        writer.flush()?;
        Self::print_terminal_row(&baseline_entry);

        let mut rng = rand::thread_rng();
        let mut episode: u64 = 0;

        while !self.stop_flag.load(Ordering::Acquire) {
            let opp_depth = rng.gen_range(
                self.stress_config.opponent_depth_min..=self.stress_config.opponent_depth_max,
            );
            *self.depth_histogram.entry(opp_depth).or_insert(0) += 1;

            // Alternate sides per episode: even episodes → P1, odd → P2.
            // Matches the side distribution used by score_vs_minimax so that
            // training and evaluation see the same policy coverage.
            let agent_side = if episode.is_multiple_of(2) {
                Player::One
            } else {
                Player::Two
            };
            let mut minimax = MinimaxPlayer::new(opp_depth);
            self.run_stress_episode(&mut minimax, agent_side);

            episode += 1;

            // Track hysteresis transitions (M2 only)
            let (a_state, _, _, c_state, _, _) = Self::capture_cl_state(&self.agent);
            let new_actor = Self::char_to_state(a_state);
            let new_critic = Self::char_to_state(c_state);
            if new_actor != self.prev_actor_state {
                self.hysteresis_transitions += 1;
                self.prev_actor_state = new_actor;
            }
            if new_critic != self.prev_critic_state {
                self.hysteresis_transitions += 1;
                self.prev_critic_state = new_critic;
            }

            if episode.is_multiple_of(self.stress_config.assessment_interval as u64) {
                let entry = self.record_scoring(episode, baseline_fitness, prev_fitness)?;
                Self::write_csv_row(&mut writer, &entry)?;
                writer.flush()?;
                Self::print_terminal_row(&entry);

                prev_fitness = entry.fitness;
                max_fitness = max_fitness.max(entry.fitness);
                min_fitness = min_fitness.min(entry.fitness);
                match entry.status {
                    StressStatus::Improved => improvements += 1,
                    StressStatus::Stable => stable += 1,
                    StressStatus::Degraded => degradations += 1,
                    StressStatus::Baseline => {}
                }
                history.push(entry);
            }

            if self.stress_config.max_episodes > 0
                && episode >= self.stress_config.max_episodes as u64
            {
                break;
            }
        }

        // Final scoring if last episode was not already a checkpoint
        let final_fitness = if episode > 0
            && !episode.is_multiple_of(self.stress_config.assessment_interval as u64)
        {
            let entry = self.record_scoring(episode, baseline_fitness, prev_fitness)?;
            Self::write_csv_row(&mut writer, &entry)?;
            writer.flush()?;
            Self::print_terminal_row(&entry);
            let f = entry.fitness;
            max_fitness = max_fitness.max(f);
            min_fitness = min_fitness.min(f);
            match entry.status {
                StressStatus::Improved => improvements += 1,
                StressStatus::Stable => stable += 1,
                StressStatus::Degraded => degradations += 1,
                StressStatus::Baseline => {}
            }
            history.push(entry);
            f
        } else {
            prev_fitness
        };

        save_agent(
            &self.agent,
            &self.stress_config.output_agent_path,
            episode as usize,
            None,
        )?;

        Ok(StressResult {
            baseline_fitness,
            final_fitness,
            max_fitness,
            min_fitness,
            improvements,
            stable,
            degradations,
            total_episodes: episode,
            history,
        })
    }

    /// Scores the agent and builds a [`StressLogEntry`] for a given episode.
    fn record_scoring(
        &mut self,
        episode: u64,
        baseline_fitness: f64,
        prev_fitness: f64,
    ) -> Result<StressLogEntry, Box<dyn Error>> {
        // Defensive isolation: reset per-step bookkeeping before scoring so that
        // the inference-only act() calls in score_vs_minimax cannot contaminate
        // the next training step_masked(). The exact drift signal we are
        // measuring depends on this isolation.
        self.agent.reset_step();
        let (w, d, l) = score_vs_minimax(
            &mut self.agent,
            STRESS_SCORING_DEPTH,
            self.stress_config.assessment_games,
        );
        // Defensive isolation: clear any inference-side cache so the next
        // run_stress_episode starts from a clean per-step state.
        self.agent.reset_step();
        let fitness = Fitness::from_scores(w, d, STRESS_SCORING_DEPTH).combined();
        let delta_baseline = fitness - baseline_fitness;
        let delta_prev = fitness - prev_fitness;
        let status = classify_status(delta_prev);
        let (a_state, a_fast, a_slow, c_state, c_fast, c_slow) =
            Self::capture_cl_state(&self.agent);
        Ok(StressLogEntry {
            episode,
            depth_histogram_snapshot: format_depth_histogram(&self.depth_histogram),
            fitness,
            win_rate: w,
            draw_rate: d,
            loss_rate: l,
            delta_vs_baseline: delta_baseline,
            delta_vs_previous: delta_prev,
            status,
            actor_state: a_state,
            actor_fast: a_fast,
            actor_slow: a_slow,
            critic_state: c_state,
            critic_fast: c_fast,
            critic_slow: c_slow,
            hysteresis_transitions: self.hysteresis_transitions,
        })
    }

    /// Reads CL state from the agent and returns `(actor_state_char, actor_fast,
    /// actor_slow, critic_state_char, critic_fast, critic_slow)`.
    ///
    /// Returns `('P', 0.0, 0.0, 'P', 0.0, 0.0)` when hysteresis is disabled.
    ///
    /// Accesses `ClState` via `agent.to_cl_state()`. The hysteresis fields are
    /// `HysteresisStateSerialized` (from `pc_rl_core::serializer`), which has
    /// the same `.state`, `.fast.value`, `.slow.value` fields as the runtime
    /// `HysteresisState` but in serializable form.
    fn capture_cl_state(agent: &PcActorCritic) -> (char, f64, f64, char, f64, f64) {
        let cl = agent.to_cl_state();
        let mut a_state = 'P';
        let mut a_fast = 0.0;
        let mut a_slow = 0.0;
        let mut c_state = 'P';
        let mut c_fast = 0.0;
        let mut c_slow = 0.0;
        if let Some(cl) = cl {
            if let Some(ref ah) = cl.actor_hysteresis {
                a_state = match ah.state {
                    PlasticityState::Plastic => 'P',
                    PlasticityState::Frozen => 'F',
                };
                a_fast = ah.fast.value;
                a_slow = ah.slow.value;
            }
            if let Some(ref ch) = cl.critic_hysteresis {
                c_state = match ch.state {
                    PlasticityState::Plastic => 'P',
                    PlasticityState::Frozen => 'F',
                };
                c_fast = ch.fast.value;
                c_slow = ch.slow.value;
            }
        }
        (a_state, a_fast, a_slow, c_state, c_fast, c_slow)
    }

    /// Converts a state character (`'P'` / `'F'`) to [`PlasticityState`].
    fn char_to_state(c: char) -> PlasticityState {
        if c == 'F' {
            PlasticityState::Frozen
        } else {
            PlasticityState::Plastic
        }
    }

    /// Runs one training episode via `step_masked` (TD-0 continuous learning).
    ///
    /// The agent always plays as `agent_side`. The opponent (minimax) moves
    /// whenever `env.current_player() != agent_side`.
    ///
    /// The terminal reward is sent in the final `step_masked` call with
    /// `terminal = true`. The `expect` calls here are unreachable in correct
    /// game logic (minimax and valid_actions always agree on legal moves).
    fn run_stress_episode(&mut self, minimax: &mut MinimaxPlayer, agent_side: Player) {
        let mut env = TicTacToe::new();
        self.agent.reset_step();

        // If agent plays as Player::Two, opponent moves first
        if agent_side == Player::Two && !env.is_terminal() {
            let opp = minimax.choose_action(&env);
            env.step(opp).expect("minimax chose valid action");
        }

        while !env.is_terminal() {
            let state = env.board_as_f64(agent_side);
            let valid = env.valid_actions();
            let action = self
                .agent
                .step_masked(&state, &valid, 0.0, false)
                .expect("step_masked failed with valid actions");
            env.step(action).expect("agent chose valid action");

            if !env.is_terminal() && env.current_player() != agent_side {
                let opp = minimax.choose_action(&env);
                env.step(opp).expect("minimax chose valid action");
            }
        }

        // Terminal step: send actual reward so TD target is correct
        let terminal_reward = env.reward(agent_side);
        let final_state = env.board_as_f64(agent_side);
        let final_valid: Vec<usize> = (0..9).collect();
        self.agent
            .step_masked(&final_state, &final_valid, terminal_reward, true)
            .expect("terminal step_masked failed during stress test");

        let outcome = match env.result() {
            GameResult::Win(p) if p == agent_side => GameOutcome::Win,
            GameResult::Win(_) => GameOutcome::Loss,
            GameResult::Draw => GameOutcome::Draw,
            GameResult::InProgress => GameOutcome::Draw,
        };
        self.metrics.record(outcome);
    }

    /// Writes one CSV data row to `writer`.
    fn write_csv_row<W: Write>(
        writer: &mut W,
        entry: &StressLogEntry,
    ) -> Result<(), Box<dyn Error>> {
        let delta_baseline = format!("{:+.4}", entry.delta_vs_baseline);
        let delta_prev = format!("{:+.4}", entry.delta_vs_previous);
        writeln!(
            writer,
            "{ep},{hist},{fit:.4},{w:.3},{d:.3},{l:.3},{db},{dp},{status},{asch},{af:.4},{asl:.4},{csch},{cf:.4},{csl:.4},{tr}",
            ep = entry.episode,
            hist = entry.depth_histogram_snapshot,
            fit = entry.fitness,
            w = entry.win_rate,
            d = entry.draw_rate,
            l = entry.loss_rate,
            db = delta_baseline,
            dp = delta_prev,
            status = entry.status,
            asch = entry.actor_state,
            af = entry.actor_fast,
            asl = entry.actor_slow,
            csch = entry.critic_state,
            cf = entry.critic_fast,
            csl = entry.critic_slow,
            tr = entry.hysteresis_transitions
        )?;
        Ok(())
    }

    /// Prints a one-line summary to stderr for live monitoring.
    fn print_terminal_row(entry: &StressLogEntry) {
        eprintln!(
            "[ep {ep:>7}] fit={fit:.4} ({db:+.4} base, {dp:+.4} prev) {status:<8} | A={asch}(f={af:.4} s={asl:.4}) C={csch}(f={cf:.4} s={csl:.4}) | trans={tr}",
            ep = entry.episode,
            fit = entry.fitness,
            db = entry.delta_vs_baseline,
            dp = entry.delta_vs_previous,
            status = entry.status.to_string(),
            asch = entry.actor_state,
            af = entry.actor_fast,
            asl = entry.actor_slow,
            csch = entry.critic_state,
            cf = entry.critic_fast,
            csl = entry.critic_slow,
            tr = entry.hysteresis_transitions
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_status_baseline_threshold() {
        assert_eq!(classify_status(0.005), StressStatus::Stable);
        assert_eq!(classify_status(-0.005), StressStatus::Stable);
        assert_eq!(classify_status(0.02), StressStatus::Improved);
        assert_eq!(classify_status(-0.02), StressStatus::Degraded);
    }

    #[test]
    fn test_format_depth_histogram_empty() {
        let h = BTreeMap::new();
        assert_eq!(format_depth_histogram(&h), "-");
    }

    #[test]
    fn test_format_depth_histogram_sorted() {
        let mut h = BTreeMap::new();
        h.insert(7, 312_u64);
        h.insert(1, 245_u64);
        h.insert(3, 198_u64);
        h.insert(9, 245_u64);
        assert_eq!(format_depth_histogram(&h), "D1:245 D3:198 D7:312 D9:245");
    }

    #[test]
    fn test_csv_header_matches_spec() {
        assert!(CSV_HEADER.starts_with("episode,opponent_depths_seen,fitness"));
        assert!(CSV_HEADER.ends_with("hysteresis_transitions"));
        // 16 columns total
        assert_eq!(CSV_HEADER.split(',').count(), 16);
    }

    #[test]
    fn test_stress_status_display() {
        assert_eq!(StressStatus::Baseline.to_string(), "BASELINE");
        assert_eq!(StressStatus::Improved.to_string(), "IMPROVED");
        assert_eq!(StressStatus::Stable.to_string(), "STABLE");
        assert_eq!(StressStatus::Degraded.to_string(), "DEGRADED");
    }
}
