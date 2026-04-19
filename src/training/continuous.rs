// Author: Julian Bolivar
// Version: 2.0.0
// Date: 2026-04-10

//! Continuous training loop using the unified `step_masked()` API.
//!
//! Implements [`ContinuousTrainer`] which runs until a stop flag is set
//! or a maximum episode count is reached. Each agent step uses
//! [`PcActorCritic::step_masked()`] for combined inference and TD(0)
//! learning — surprise-driven plasticity is handled internally by the
//! core library. Includes curriculum learning with depth advancement.
//!
//! # Phase 2 self-recovery orchestration (TODO)
//!
//! The Phase 2 APIs shipped by `pc-rl-core` — `replay_learn`,
//! `seal_replay_training_memories`, `clear_recent_memories`,
//! `rollback_soft`, `rollback_hard`, `champion_update` — are NOT yet wired
//! into this trainer loop. Setting the Phase 2 TOML fields
//! (`replay_training_capacity`, `distillation_lambda_polyak`,
//! `distillation_lambda_frozen`, etc.) currently:
//!
//! - allocates the replay buffer and distillation anchors inside the core,
//! - lets `step_masked()` auto-record transitions and apply KL distillation
//!   gradients against live anchors,
//! - but never calls `replay_learn()` to consume the buffer, and
//! - never calls `champion_update()` / `rollback_*()` for recovery.
//!
//! "Compiles and runs" is not "integrated". A future phase must hook these
//! APIs into the trainer (likely gated by fitness-drift detection via the
//! curriculum advancement signal) before the self-recovery behavior
//! described in the `pc-rl-core` CHANGELOG is actually exercised.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pc_rl_core::pc_actor_critic::PcActorCritic;
use rand::Rng;

use crate::env::minimax::MinimaxPlayer;
use crate::env::tictactoe::{GameResult, Player, TicTacToe};
use crate::utils::metrics::{GameOutcome, Metrics};

/// Continuous trainer using the unified `step_masked()` API with curriculum learning.
///
/// Runs episodes until stopped via `stop_flag` or `max_episodes`.
/// Each agent step calls [`PcActorCritic::step_masked()`] which performs
/// inference, action selection, and TD(0) learning in a single call.
/// Surprise-based plasticity scaling is handled internally by the core library.
///
/// Curriculum learning advances the minimax opponent depth when the
/// agent's non-loss rate exceeds `advance_threshold` over a sliding window.
///
/// # Examples
///
/// ```no_run
/// use std::sync::Arc;
/// use std::sync::atomic::AtomicBool;
/// use pc_rl_core::pc_actor_critic::PcActorCritic;
/// use pc_rl_core::CpuLinAlg;
/// use pc_tictactoe::training::continuous::ContinuousTrainer;
/// use pc_tictactoe::utils::config::AppConfig;
///
/// let config = AppConfig::default();
/// let agent_config = config.to_agent_config().unwrap();
/// let agent = PcActorCritic::new(CpuLinAlg::new(), agent_config, 42).unwrap();
/// let stop = Arc::new(AtomicBool::new(false));
/// let mut trainer = ContinuousTrainer::new(agent, &config, stop);
/// trainer.train();
/// ```
pub struct ContinuousTrainer {
    /// The reinforcement learning agent.
    agent: PcActorCritic,
    /// The game environment.
    env: TicTacToe,
    /// The minimax opponent.
    minimax: MinimaxPlayer,
    /// Sliding-window metrics tracker.
    metrics: Metrics,
    /// Atomic flag for Ctrl+C shutdown.
    stop_flag: Arc<AtomicBool>,
    /// Maximum episodes before automatic stop.
    max_episodes: usize,
    /// Total episodes trained.
    episode_count: usize,
    /// Total agent steps across all episodes.
    step_count: usize,
    /// Current minimax search depth (curriculum level).
    current_depth: usize,
    /// Non-loss rate threshold to advance curriculum depth.
    advance_threshold: f64,
    /// How often to print progress (every N episodes). 0 = silent.
    log_interval: usize,
    /// Collected log lines for programmatic access.
    log_lines: Vec<String>,
    /// Use random side assignment instead of alternating.
    random_side: bool,
    /// RNG for random side selection.
    rng: rand::rngs::ThreadRng,
    /// Agent side for the current episode (set by run_episode, read by episode_outcome).
    last_agent_side: Player,
    /// True iff `config.agent.replay_training_capacity > 0` at construction.
    /// Cached to avoid per-iteration config lookups and coupling to agent internals.
    replay_enabled: bool,
    /// Number of episodes between `replay_learn` invocations (from `config.training.replay_interval`).
    /// Ignored when `replay_enabled == false`.
    /// Used in Task 5 (replay trigger); suppressed until then.
    #[allow(dead_code)]
    replay_interval: usize,
    /// Batch size for each `replay_learn` call (from `config.agent.replay_batch_size`).
    /// Used in Task 5 (replay trigger); suppressed until then.
    #[allow(dead_code)]
    replay_batch_size: usize,
    /// `true` after the first successful `seal_replay_training_memories()` call.
    /// Warmup gate — `replay_learn` is not invoked until this is `true`.
    training_memories_sealed: bool,
    /// Counter of successful (`Ok`) `replay_learn` invocations. Diagnostic.
    replay_invocations: usize,
    /// Counter of seal attempts (both Ok and Err). Increments inside the
    /// `if !sealed && replay_enabled` block. Once the first attempt succeeds
    /// and the flag flips, subsequent advances skip the block — counter stays at 1.
    /// Used for precise idempotency verification in Scenario 4.3.
    seal_attempts: usize,
}

impl ContinuousTrainer {
    /// Creates a new continuous trainer.
    ///
    /// # Parameters
    ///
    /// * `agent` - The PC Actor-Critic agent to train.
    /// * `config` - Application configuration.
    /// * `stop_flag` - Atomic flag; set to `true` to stop after current episode.
    pub fn new(
        agent: PcActorCritic,
        config: &crate::utils::config::AppConfig,
        stop_flag: Arc<AtomicBool>,
    ) -> Self {
        Self {
            agent,
            env: TicTacToe::new(),
            minimax: MinimaxPlayer::new(1),
            metrics: Metrics::new(config.curriculum.window_size),
            stop_flag,
            max_episodes: config.continuous.max_episodes,
            episode_count: 0,
            step_count: 0,
            current_depth: 1,
            advance_threshold: config.curriculum.advance_threshold,
            log_interval: config.training.log_interval,
            log_lines: Vec::new(),
            random_side: config.continuous.random_side,
            rng: rand::thread_rng(),
            last_agent_side: Player::One,
            replay_enabled: config.agent.replay_training_capacity > 0,
            replay_interval: config.training.replay_interval,
            replay_batch_size: config.agent.replay_batch_size,
            training_memories_sealed: false,
            replay_invocations: 0,
            seal_attempts: 0,
        }
    }

    /// Runs the continuous training loop.
    ///
    /// Continues until `stop_flag` is set or `max_episodes` is reached.
    /// Each episode uses `step_masked()` for unified inference and learning,
    /// then checks curriculum advancement.
    pub fn train(&mut self) {
        while !self.stop_flag.load(Ordering::Acquire) && self.episode_count < self.max_episodes {
            self.run_episode();

            // Record outcome (before incrementing episode_count, since
            // run_episode used the current count to determine agent_side)
            let outcome = self.episode_outcome();
            self.metrics.record(outcome);

            // Check curriculum advancement (only after window is full)
            let prev_depth = self.current_depth;
            let non_loss_rate = self.metrics.win_rate() + self.metrics.draw_rate();
            if self.metrics.count() >= self.metrics.window_size()
                && non_loss_rate > self.advance_threshold
                && self.current_depth < 9
            {
                self.current_depth += 1;
                self.minimax = MinimaxPlayer::new(self.current_depth);
                self.metrics.reset();
                self.try_seal_on_first_advance(prev_depth);
            }

            self.episode_count += 1;

            if self.log_interval > 0 && self.episode_count.is_multiple_of(self.log_interval) {
                let cl_info = self.cl_status_string();
                let line = format!(
                    "[ep {:>6}/{total}] win={win:.1}% loss={loss:.1}% draw={draw:.1}% | depth={depth} steps={steps}{cl}",
                    self.episode_count,
                    total = self.max_episodes,
                    win = self.metrics.win_rate() * 100.0,
                    loss = self.metrics.loss_rate() * 100.0,
                    draw = self.metrics.draw_rate() * 100.0,
                    depth = self.current_depth,
                    steps = self.step_count,
                    cl = cl_info,
                );
                eprintln!("{line}");
                self.log_lines.push(line);
            }
            if prev_depth != self.current_depth {
                let line = format!(
                    "  >> Curriculum advanced: depth {} -> {}",
                    prev_depth, self.current_depth
                );
                eprintln!("{line}");
                self.log_lines.push(line);
            }
        }
    }

    /// Runs a single episode using `step_masked()` for unified inference and TD(0) learning.
    ///
    /// The agent alternates sides each episode. On each agent turn,
    /// `step_masked()` learns from the previous transition and selects
    /// the next action. A final terminal call with the game reward
    /// completes the episode.
    fn run_episode(&mut self) {
        self.env.reset();
        self.agent.reset_step();

        let agent_side = if self.random_side {
            if self.rng.gen_bool(0.5) {
                Player::One
            } else {
                Player::Two
            }
        } else if self.episode_count.is_multiple_of(2) {
            Player::One
        } else {
            Player::Two
        };

        // Store for episode_outcome()
        self.last_agent_side = agent_side;

        // If agent is Player Two, let opponent move first
        if agent_side == Player::Two && !self.env.is_terminal() {
            let opp_action = self.minimax.choose_action(&self.env);
            self.env.step(opp_action).unwrap();
        }

        // Guard: if game ended before agent could act (defensive; can't happen
        // in standard TicTacToe but protects against future rule variants)
        if self.env.is_terminal() {
            let terminal_reward = self.env.reward(agent_side);
            let state = self.env.board_as_f64(agent_side);
            let valid: Vec<usize> = (0..9).collect();
            let _ = self
                .agent
                .step_masked(&state, &valid, terminal_reward, true);
            self.step_count += 1;
            return;
        }

        while !self.env.is_terminal() {
            // Agent's turn
            let state = self.env.board_as_f64(agent_side);
            let valid = self.env.valid_actions();
            let action = self
                .agent
                .step_masked(&state, &valid, 0.0, false)
                .expect("step_masked failed: valid_actions was non-empty but returned Err");
            self.step_count += 1;
            self.env.step(action).expect("agent chose invalid action");

            // Opponent's turn (if game not over)
            if !self.env.is_terminal() && self.env.current_player() != agent_side {
                let opp_action = self.minimax.choose_action(&self.env);
                self.env.step(opp_action).unwrap();
            }
        }

        // Terminal step: send final reward with terminal=true
        let terminal_reward = self.env.reward(agent_side);
        let final_state = self.env.board_as_f64(agent_side);
        let final_valid: Vec<usize> = (0..9).collect();
        let _ = self
            .agent
            .step_masked(&final_state, &final_valid, terminal_reward, true)
            .expect("terminal step_masked failed unexpectedly");
        self.step_count += 1;
    }

    /// Determines the game outcome from the agent's perspective.
    fn episode_outcome(&self) -> GameOutcome {
        let agent_side = self.last_agent_side;
        debug_assert!(
            self.env.is_terminal(),
            "episode_outcome called on non-terminal game"
        );
        match self.env.result() {
            GameResult::Win(p) if p == agent_side => GameOutcome::Win,
            GameResult::Win(_) => GameOutcome::Loss,
            GameResult::Draw => GameOutcome::Draw,
            GameResult::InProgress => GameOutcome::Draw,
        }
    }

    /// Returns the total number of episodes trained.
    pub fn episode_count(&self) -> usize {
        self.episode_count
    }

    /// Returns the total number of agent steps across all episodes.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Returns the current curriculum depth.
    pub fn current_depth(&self) -> usize {
        self.current_depth
    }

    /// Returns the count of successful `replay_learn` invocations since construction.
    ///
    /// Primarily for testing and diagnostic purposes.
    pub fn replay_invocations(&self) -> usize {
        self.replay_invocations
    }

    /// Returns `true` after the first successful `seal_replay_training_memories()` call.
    ///
    /// Acts as the warmup gate for `replay_learn`; `false` until the seal succeeds.
    /// Primarily for testing and diagnostic purposes.
    pub fn training_memories_sealed(&self) -> bool {
        self.training_memories_sealed
    }

    /// Returns the count of seal attempts (Ok + Err combined).
    ///
    /// After the first Ok, the `training_memories_sealed` flag prevents re-entry;
    /// `seal_attempts` stays at 1 if the first attempt succeeded. If the first
    /// attempt returned Err, this can increment until a subsequent advance succeeds.
    ///
    /// Primarily for testing (Scenario 4.3 idempotency verification).
    pub fn seal_attempts(&self) -> usize {
        self.seal_attempts
    }

    /// Returns `true` when `config.agent.replay_training_capacity > 0` at construction.
    ///
    /// Lets callers determine whether the trainer was built with Phase 2 active
    /// without re-reading the config or coupling to agent internals.
    ///
    /// Primarily for testing and diagnostic purposes.
    pub fn replay_enabled(&self) -> bool {
        self.replay_enabled
    }

    /// Returns a reference to the metrics tracker.
    pub fn metrics(&self) -> &Metrics {
        &self.metrics
    }

    /// Returns the collected log lines.
    pub fn log_lines(&self) -> &[String] {
        &self.log_lines
    }

    /// Returns a CL status string for diagnostic logging.
    ///
    /// Shows actor/critic plasticity state and EWMA values when
    /// hysteresis is enabled. Returns empty string when CL is disabled.
    fn cl_status_string(&self) -> String {
        let cl = self.agent.to_cl_state();
        let Some(cl) = cl else {
            return String::new();
        };
        let mut parts = Vec::new();
        if let Some(ref ah) = cl.actor_hysteresis {
            let state_ch = match ah.state {
                pc_rl_core::PlasticityState::Plastic => "P",
                pc_rl_core::PlasticityState::Frozen => "F",
            };
            parts.push(format!(
                " actor={state_ch}(f={:.4} s={:.4})",
                ah.fast.value, ah.slow.value
            ));
        }
        if let Some(ref ch) = cl.critic_hysteresis {
            let state_ch = match ch.state {
                pc_rl_core::PlasticityState::Plastic => "P",
                pc_rl_core::PlasticityState::Frozen => "F",
            };
            parts.push(format!(
                " critic={state_ch}(f={:.4} s={:.4})",
                ch.fast.value, ch.slow.value
            ));
        }
        parts.join("")
    }

    /// Attempts to seal replay training memories on the first curriculum advance.
    ///
    /// No-op if already sealed or replay is not enabled. On `Err`, logs a
    /// warning and leaves `training_memories_sealed` false so the next advance
    /// retries. On `Ok`, flips `training_memories_sealed = true`; subsequent
    /// advances skip this block entirely (idempotency via flag guard).
    ///
    /// `seal_attempts` increments before the `match` so both `Ok` and `Err`
    /// outcomes are counted. After the first `Ok` the flag prevents re-entry,
    /// keeping the counter at exactly 1.
    ///
    /// # Parameters
    ///
    /// * `prev_depth` - Curriculum depth before the advance (used in log message).
    fn try_seal_on_first_advance(&mut self, prev_depth: usize) {
        if self.training_memories_sealed || !self.replay_enabled {
            return;
        }
        // Increment BEFORE the match to count attempts (Ok + Err).
        self.seal_attempts += 1;
        match self.agent.seal_replay_training_memories() {
            Ok(()) => {
                self.training_memories_sealed = true;
                let line = format!(
                    "[ep {}] replay training memories sealed (curriculum advance {}->{})",
                    self.episode_count, prev_depth, self.current_depth,
                );
                eprintln!("{line}");
                self.log_lines.push(line);
            }
            Err(e) => {
                // Log-warn-skip: retry on next advance; sealed stays false.
                let line = format!(
                    "[ep {}] seal_replay_training_memories failed: {} (will retry next advance)",
                    self.episode_count, e,
                );
                eprintln!("{line}");
                self.log_lines.push(line);
            }
        }
    }

    /// Returns a reference to the agent.
    pub fn agent(&self) -> &PcActorCritic {
        &self.agent
    }

    /// Returns `true` when training should stop.
    ///
    /// Stops when either the stop flag is set or the episode count has
    /// reached `max_episodes`. Callers driving external loops (e.g.
    /// `ChampionFinder`) should check this before each iteration.
    #[must_use]
    pub fn should_stop(&self) -> bool {
        self.stop_flag.load(Ordering::Acquire) || self.episode_count >= self.max_episodes
    }

    /// Runs a single episode and records its outcome.
    ///
    /// Increments `episode_count`, records the episode outcome in the
    /// metrics tracker, and checks curriculum advancement using the same
    /// logic as [`Self::train()`]. Does **not** check the stop flag,
    /// print progress, or emit log lines — callers are responsible for
    /// those concerns.
    ///
    /// # Contract
    ///
    /// After this call `episode_count` is exactly one greater than before.
    /// Curriculum depth may have advanced if the advancement criterion was met.
    pub fn train_one_episode(&mut self) {
        self.run_episode();

        let outcome = self.episode_outcome();
        self.metrics.record(outcome);

        let non_loss_rate = self.metrics.win_rate() + self.metrics.draw_rate();
        if self.metrics.count() >= self.metrics.window_size()
            && non_loss_rate > self.advance_threshold
            && self.current_depth < 9
        {
            self.current_depth += 1;
            self.minimax = MinimaxPlayer::new(self.current_depth);
            self.metrics.reset();
        }

        self.episode_count += 1;
    }

    /// Returns a mutable reference to the agent.
    ///
    /// Used by external scorers (e.g. `ChampionFinder::score_vs_minimax`)
    /// that need to call `act()` without modifying trainer state.
    #[must_use]
    pub fn agent_mut(&mut self) -> &mut PcActorCritic {
        &mut self.agent
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::config::AppConfig;
    use pc_rl_core::CpuLinAlg;

    fn make_continuous_trainer(max_episodes: usize) -> ContinuousTrainer {
        let mut config = AppConfig::default();
        config.continuous.max_episodes = max_episodes;
        let agent_config = config.to_agent_config().unwrap();
        let agent = PcActorCritic::new(CpuLinAlg::new(), agent_config, 42).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        ContinuousTrainer::new(agent, &config, stop)
    }

    fn make_cl_trainer_with_hysteresis(max_episodes: usize) -> ContinuousTrainer {
        let mut config = AppConfig::default();
        config.continuous.max_episodes = max_episodes;
        config.agent.actor_hysteresis = true;
        config.agent.critic_hysteresis = true;
        config.agent.scale_floor = 0.0;
        let agent_config = config.to_agent_config().unwrap();
        let agent = PcActorCritic::new(CpuLinAlg::new(), agent_config, 42).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        ContinuousTrainer::new(agent, &config, stop)
    }

    #[test]
    fn test_one_episode_completes_without_panic() {
        let mut trainer = make_continuous_trainer(1);
        trainer.train();
        assert_eq!(trainer.episode_count(), 1);
    }

    #[test]
    fn test_max_episodes_stops_training() {
        let mut trainer = make_continuous_trainer(3);
        trainer.train();
        assert_eq!(trainer.episode_count(), 3);
    }

    #[test]
    fn test_step_count_positive_after_training() {
        let mut trainer = make_continuous_trainer(5);
        trainer.train();
        assert!(
            trainer.step_count() > 0,
            "Expected positive step count after training"
        );
    }

    #[test]
    fn test_curriculum_advances_with_low_threshold() {
        let mut config = AppConfig::default();
        config.continuous.max_episodes = 2000;
        config.curriculum.advance_threshold = 0.0;
        config.curriculum.window_size = 1;
        let agent_config = config.to_agent_config().unwrap();
        let agent = PcActorCritic::new(CpuLinAlg::new(), agent_config, 42).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let mut trainer = ContinuousTrainer::new(agent, &config, stop);
        trainer.train();
        assert!(
            trainer.current_depth() > 1,
            "Curriculum should advance with threshold=0.0"
        );
    }

    #[test]
    fn test_hysteresis_enabled_completes_without_panic() {
        let mut trainer = make_cl_trainer_with_hysteresis(10);
        trainer.train();
        assert_eq!(trainer.episode_count(), 10);
    }

    #[test]
    fn test_both_sides_complete_without_panic() {
        // Runs 4 episodes: even episodes agent is Player::One, odd is Player::Two.
        // Verifies both sides execute without panic.
        let mut trainer = make_continuous_trainer(4);
        trainer.train();
        assert_eq!(trainer.episode_count(), 4);
    }

    #[test]
    fn test_should_stop_false_initially() {
        let trainer = make_continuous_trainer(100);
        assert!(!trainer.should_stop());
    }

    #[test]
    fn test_should_stop_true_after_stop_flag() {
        let mut config = AppConfig::default();
        config.continuous.max_episodes = 100;
        let agent_config = config.to_agent_config().unwrap();
        let agent = PcActorCritic::new(CpuLinAlg::new(), agent_config, 42).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let trainer = ContinuousTrainer::new(agent, &config, stop.clone());
        stop.store(true, Ordering::SeqCst);
        assert!(trainer.should_stop());
    }

    #[test]
    fn test_train_one_episode_increments_count() {
        let mut trainer = make_continuous_trainer(100);
        assert_eq!(trainer.episode_count(), 0);
        trainer.train_one_episode();
        assert_eq!(trainer.episode_count(), 1);
    }

    #[test]
    fn test_agent_mut_returns_same_agent() {
        let mut trainer = make_continuous_trainer(100);
        let ptr1 = trainer.agent() as *const _;
        let ptr2 = trainer.agent_mut() as *const _;
        assert_eq!(ptr1, ptr2);
    }

    /// Builds a `ContinuousTrainer` with the Phase 2 replay fields configured.
    ///
    /// # Parameters
    ///
    /// * `replay_training_capacity` - Capacity of compartment A (0 = replay disabled).
    /// * `replay_interval` - Episodes between `replay_learn` calls.
    /// * `advance_threshold` - Non-loss rate to advance curriculum depth.
    /// * `window_size` - Sliding-window size for curriculum metrics.
    /// * `max_episodes` - Episode cap for the training loop.
    fn build_test_trainer(
        replay_training_capacity: usize,
        replay_interval: usize,
        advance_threshold: f64,
        window_size: usize,
        max_episodes: usize,
    ) -> ContinuousTrainer {
        let mut config = AppConfig::default();
        config.agent.replay_training_capacity = replay_training_capacity;
        config.agent.replay_recent_capacity = if replay_training_capacity > 0 { 128 } else { 0 };
        config.training.replay_interval = replay_interval;
        config.curriculum.advance_threshold = advance_threshold;
        config.curriculum.window_size = window_size;
        config.continuous.max_episodes = max_episodes;
        config.training.seed = 42;

        let agent_config = config.to_agent_config().unwrap();
        let agent = PcActorCritic::new(CpuLinAlg::new(), agent_config, 42).unwrap();
        ContinuousTrainer::new(agent, &config, Arc::new(AtomicBool::new(false)))
    }

    #[test]
    fn test_trainer_construction_phase2_off_initial_state() {
        let trainer = build_test_trainer(
            /* replay_training_capacity */ 0, /* replay_interval */ 100,
            /* advance_threshold */ 0.95, /* window_size */ 100,
            /* max_episodes */ 200,
        );
        assert_eq!(trainer.replay_invocations(), 0);
        assert!(!trainer.training_memories_sealed());
        assert_eq!(trainer.seal_attempts(), 0);
        assert!(
            !trainer.replay_enabled(),
            "replay_enabled should be false when capacity=0"
        );
    }

    #[test]
    fn test_trainer_construction_phase2_on_initial_state() {
        let trainer = build_test_trainer(
            /* replay_training_capacity */ 256, /* replay_interval */ 50,
            /* advance_threshold */ 0.30, /* window_size */ 20,
            /* max_episodes */ 100,
        );
        assert_eq!(trainer.replay_invocations(), 0);
        assert!(!trainer.training_memories_sealed());
        assert_eq!(trainer.seal_attempts(), 0);
        assert!(
            trainer.replay_enabled(),
            "replay_enabled should be true when capacity>0"
        );
    }

    #[test]
    fn test_scenario_4_3_seal_only_once_on_first_advance() {
        // Given: Phase 2 active, trivially easy curriculum (threshold=0.0, window=1)
        // so the first episode always triggers the first advance.
        let mut trainer = build_test_trainer(
            /* replay_training_capacity */ 256, /* replay_interval */ 100,
            /* advance_threshold */ 0.0, /* window_size */ 1,
            /* max_episodes */ 200,
        );
        // When: run until max_episodes (expecting multiple advances)
        trainer.train();
        // Then: sealed == true after the first advance
        assert!(
            trainer.training_memories_sealed(),
            "sealed should be true after curriculum advance"
        );
        // Idempotency check: seal_attempts counter increments inside the
        // `if !sealed` block. After the first Ok, the flag prevents re-entry
        // on subsequent advances — counter stays at 1 exact.
        assert_eq!(
            trainer.seal_attempts(),
            1,
            "seal_attempts should be exactly 1 — idempotency guaranteed by the flag guard"
        );
    }

    #[test]
    fn test_scenario_4_4_sealed_false_before_first_advance() {
        // Given: Phase 2 active, advance_threshold impossible in short tests
        let mut trainer = build_test_trainer(
            /* replay_training_capacity */ 256, /* replay_interval */ 5,
            /* advance_threshold */ 0.999, /* window_size */ 100,
            /* max_episodes */ 50,
        );
        // When: run 50 episodes without advance
        trainer.train();
        // Then: sealed remains false, no seal attempts
        assert!(
            !trainer.training_memories_sealed(),
            "sealed should remain false without advance"
        );
        assert_eq!(trainer.seal_attempts(), 0);
    }
}
