// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! Clap-based CLI with subcommands for training, playing, evaluating, and
//! benchmarking the PC Actor-Critic agent on Tic-Tac-Toe.
//!
//! # Subcommands
//!
//! - **train** — Run episodic or continuous training.
//! - **play** — Interactive text-based game against the agent.
//! - **evaluate** — Win/draw/loss statistics vs minimax at a given depth.
//! - **benchmark** — Timing and throughput metrics for training.

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand};

use pc_rl_core::pc_actor::SelectionMode;
use pc_rl_core::pc_actor_critic::PcActorCritic;
use pc_rl_core::serializer::{load_agent, save_agent};
use pc_rl_core::CpuLinAlg;

use crate::env::minimax::MinimaxPlayer;
use crate::env::tictactoe::{GameResult, Player, TicTacToe};
use crate::training::continuous::ContinuousTrainer;
use crate::training::trainer::Trainer;
use crate::utils::config::AppConfig;

/// PC-TicTacToe: Predictive Coding Actor-Critic for Tic-Tac-Toe.
#[derive(Parser)]
#[command(name = "pc_tictactoe", version, about)]
pub struct Cli {
    /// Subcommand to execute.
    #[command(subcommand)]
    pub command: Command,
}

/// Available subcommands.
#[derive(Subcommand)]
pub enum Command {
    /// Train the agent against minimax opponents.
    Train(TrainArgs),
    /// Play an interactive game against the agent.
    Play(PlayArgs),
    /// Evaluate the agent vs minimax and print statistics.
    Evaluate(EvaluateArgs),
    /// Benchmark training throughput.
    Benchmark(BenchmarkArgs),
    /// Run lambda sweep experiment with random seeds.
    Experiment(ExperimentArgs),
    /// Generate default config.toml with optimal parameters.
    Init(InitArgs),
    /// Test a fixed config across N random seeds for statistical stability.
    SeedTest(SeedTestArgs),
    /// Search for the best champion across N training sessions.
    FindChampion(FindChampionArgs),
    /// Stress-test a champion against random-depth opponents.
    StressTest(StressTestArgs),
}

/// Arguments for the train subcommand.
#[derive(Parser)]
pub struct TrainArgs {
    /// Number of training episodes.
    #[arg(long, short)]
    pub episodes: Option<usize>,
    /// Path to TOML configuration file.
    #[arg(long, short, default_value = "config.toml")]
    pub config: String,
    /// Use continuous learning mode instead of episodic.
    #[arg(long)]
    pub continuous: bool,
    /// Maximum episodes for continuous mode.
    #[arg(long)]
    pub max_episodes: Option<usize>,
    /// Target win rate for curriculum advancement.
    #[arg(long)]
    pub target_winrate: Option<f64>,
    /// Blend factor: 1.0 = pure backprop, 0.0 = pure local PC, intermediate = hybrid.
    #[arg(long)]
    pub local_lambda: Option<f64>,
    /// Enable residual skip connections between same-dimension hidden layers.
    #[arg(long)]
    pub residual: bool,
    /// Initial ReZero scaling factor for residual connections.
    #[arg(long)]
    pub rezero_init: Option<f64>,
    /// Enable actor hysteresis (FROZEN/PLASTIC state machine).
    #[arg(long)]
    pub actor_hysteresis: bool,
    /// Enable critic hysteresis (FROZEN/PLASTIC state machine).
    #[arg(long)]
    pub critic_hysteresis: bool,
    /// EWC regularization strength (0.0 = disabled).
    #[arg(long)]
    pub ewc_lambda: Option<f64>,
    /// Scale floor for surprise-driven learning rate (0.0 = true freeze).
    #[arg(long)]
    pub scale_floor: Option<f64>,
}

/// Arguments for the play subcommand.
#[derive(Parser)]
pub struct PlayArgs {
    /// Path to a saved model file.
    #[arg(long, short)]
    pub model: Option<String>,
    /// Play as first player (agent goes second).
    #[arg(long)]
    pub first: bool,
}

/// Arguments for the evaluate subcommand.
#[derive(Parser)]
pub struct EvaluateArgs {
    /// Path to a saved model file.
    #[arg(long, short)]
    pub model: Option<String>,
    /// Number of evaluation games.
    #[arg(long, short, default_value = "100")]
    pub games: usize,
    /// Minimax search depth for the opponent.
    #[arg(long, short, default_value = "9")]
    pub depth: usize,
}

/// Arguments for the experiment subcommand.
#[derive(Parser)]
pub struct ExperimentArgs {
    /// Number of repetitions (random seeds).
    #[arg(long, short)]
    pub n: usize,
    /// Path to TOML configuration file.
    #[arg(long, short, default_value = "config.toml")]
    pub config: String,
    /// Parameter to sweep: "lambda" (default).
    #[arg(long, short, default_value = "lambda")]
    pub sweep: String,
}

/// Arguments for the seed-test subcommand.
#[derive(Parser)]
pub struct SeedTestArgs {
    /// Number of runs (random seeds).
    #[arg(long, short)]
    pub n: usize,
    /// Path to TOML configuration file.
    #[arg(long, short, default_value = "config.toml")]
    pub config: String,
    /// Use continuous learning mode (step_masked) instead of episodic.
    #[arg(long)]
    pub continuous: bool,
}

/// Arguments for the init subcommand.
#[derive(Parser)]
pub struct InitArgs {
    /// Output path for the generated config file.
    #[arg(long, short, default_value = "config.toml")]
    pub output: String,
}

/// Arguments for the benchmark subcommand.
#[derive(Parser)]
pub struct BenchmarkArgs {
    /// Path to a saved model file.
    #[arg(long, short)]
    pub model: Option<String>,
    /// Number of training episodes for the benchmark.
    #[arg(long, short, default_value = "100")]
    pub episodes: usize,
}

/// Arguments for the find-champion subcommand.
#[derive(Parser)]
pub struct FindChampionArgs {
    /// Path to TOML configuration file.
    #[arg(long, short, default_value = "config_champion.toml")]
    pub config: String,
    /// Override n_iterations from config.
    #[arg(long)]
    pub iterations: Option<usize>,
}

/// Arguments for the stress-test subcommand.
#[derive(Parser)]
pub struct StressTestArgs {
    /// Path to TOML configuration file.
    #[arg(long, short, default_value = "config_stress.toml")]
    pub config: String,
    /// Override champion_path from config.
    #[arg(long)]
    pub champion: Option<String>,
    /// Override max_episodes from config.
    #[arg(long)]
    pub max_episodes: Option<usize>,
}

/// Runs the train subcommand.
///
/// Loads config, creates agent + trainer, trains, and saves the final model.
///
/// # Arguments
///
/// * `args` - Training arguments from CLI.
///
/// # Errors
///
/// Returns an error on config/IO failures.
pub fn run_train(args: TrainArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = AppConfig::load(Path::new(&args.config))?;
    config.apply_cli_overrides(args.episodes, None);

    if let Some(wr) = args.target_winrate {
        config.curriculum.advance_threshold = wr;
    }

    if let Some(lambda) = args.local_lambda {
        config.agent.actor.local_lambda = lambda;
    }

    if args.residual {
        config.agent.actor.residual = true;
    }

    if let Some(ri) = args.rezero_init {
        config.agent.actor.rezero_init = ri;
    }

    if args.actor_hysteresis {
        config.agent.actor_hysteresis = true;
    }
    if args.critic_hysteresis {
        config.agent.critic_hysteresis = true;
    }
    if let Some(ewc) = args.ewc_lambda {
        config.agent.ewc_lambda = ewc;
    }
    if let Some(sf) = args.scale_floor {
        config.agent.scale_floor = sf;
    }

    config.validate()?;
    let agent_config = config.to_agent_config()?;
    let agent = PcActorCritic::new(CpuLinAlg::new(), agent_config, config.training.seed)?;

    if args.continuous {
        if let Some(max_ep) = args.max_episodes {
            config.continuous.max_episodes = max_ep;
        }
        let stop_flag = Arc::new(AtomicBool::new(false));
        let flag = stop_flag.clone();
        let _ = ctrlc::set_handler(move || {
            flag.store(true, Ordering::SeqCst);
        });
        let mut trainer = ContinuousTrainer::new(agent, &config, stop_flag);
        trainer.train();
        save_agent(
            trainer.agent(),
            "model.json",
            config.continuous.max_episodes,
            None,
        )?;
    } else {
        let episodes = config.training.episodes;
        let mut trainer = Trainer::new(agent, &config);
        trainer.train(episodes);
        save_agent(trainer.agent(), "model.json", episodes, None)?;
    }

    println!("Training complete. Model saved to model.json");
    Ok(())
}

/// Runs the play subcommand.
///
/// Loads a model (or creates a fresh agent) and plays an interactive game.
///
/// # Arguments
///
/// * `args` - Play arguments from CLI.
///
/// # Errors
///
/// Returns an error on IO/model failures.
pub fn run_play(args: PlayArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = if let Some(path) = &args.model {
        let (agent, _) = load_agent(path, CpuLinAlg::new())?;
        agent
    } else {
        let config = AppConfig::default();
        let agent_config = config.to_agent_config()?;
        PcActorCritic::new(CpuLinAlg::new(), agent_config, 42)?
    };

    let mut env = TicTacToe::new();
    let human_side = if args.first { Player::One } else { Player::Two };

    println!("You are {human_side:?}. Board positions 0-8:");
    println!(" 0 | 1 | 2 ");
    println!(" ---------  ");
    println!(" 3 | 4 | 5 ");
    println!(" ---------  ");
    println!(" 6 | 7 | 8 ");
    println!();

    let stdin = io::stdin();

    while !env.is_terminal() {
        if env.current_player() == human_side {
            print_board(&env);
            print!("Your move (0-8): ");
            io::stdout().flush()?;
            let mut line = String::new();
            stdin.lock().read_line(&mut line)?;
            let action: usize = match line.trim().parse() {
                Ok(a) => a,
                Err(_) => {
                    println!("Invalid input. Enter a number 0-8.");
                    continue;
                }
            };
            if let Err(e) = env.step(action) {
                println!("Invalid move: {e}. Try again.");
                continue;
            }
        } else {
            let state = env.board_as_f64(env.current_player());
            let valid = env.valid_actions();
            let (action, _) = agent.act(&state, &valid, SelectionMode::Play)?;
            println!("Agent plays: {action}");
            env.step(action).unwrap();
        }
    }

    print_board(&env);
    match env.result() {
        GameResult::Win(p) if p == human_side => println!("You win!"),
        GameResult::Win(_) => println!("Agent wins!"),
        GameResult::Draw => println!("Draw!"),
        GameResult::InProgress => unreachable!(),
    }

    Ok(())
}

/// Prints the current board state to stdout.
///
/// # Arguments
///
/// * `env` - The TicTacToe environment.
fn print_board(env: &TicTacToe) {
    let board = env.board_as_f64(Player::One);
    for row in 0..3 {
        for col in 0..3 {
            let idx = row * 3 + col;
            let ch = if board[idx] > 0.5 {
                "X"
            } else if board[idx] < -0.5 {
                "O"
            } else {
                "."
            };
            if col < 2 {
                print!(" {ch} |");
            } else {
                println!(" {ch} ");
            }
        }
        if row < 2 {
            println!("-----------");
        }
    }
    println!();
}

/// Runs the evaluate subcommand.
///
/// Plays N games of agent vs minimax and prints win/draw/loss statistics.
///
/// # Arguments
///
/// * `args` - Evaluate arguments from CLI.
///
/// # Errors
///
/// Returns an error on model loading failures.
pub fn run_evaluate(args: EvaluateArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut agent = if let Some(path) = &args.model {
        let (agent, _) = load_agent(path, CpuLinAlg::new())?;
        agent
    } else {
        let config = AppConfig::default();
        let agent_config = config.to_agent_config()?;
        PcActorCritic::new(CpuLinAlg::new(), agent_config, 42)?
    };

    if args.games == 0 {
        return Err("--games must be > 0".into());
    }
    let mut minimax = MinimaxPlayer::new(args.depth);
    let mut wins = 0usize;
    let mut draws = 0usize;
    let mut losses = 0usize;

    for game_idx in 0..args.games {
        let mut env = TicTacToe::new();
        let agent_side = if game_idx.is_multiple_of(2) {
            Player::One
        } else {
            Player::Two
        };

        while !env.is_terminal() {
            if env.current_player() == agent_side {
                let state = env.board_as_f64(agent_side);
                let valid = env.valid_actions();
                let (action, _) = agent.act(&state, &valid, SelectionMode::Play)?;
                env.step(action).unwrap();
            } else {
                let action = minimax.choose_action(&env);
                env.step(action).unwrap();
            }
        }

        match env.result() {
            GameResult::Win(p) if p == agent_side => wins += 1,
            GameResult::Win(_) => losses += 1,
            GameResult::Draw => draws += 1,
            GameResult::InProgress => {}
        }
    }

    println!(
        "Evaluation: {games} games vs minimax depth {depth}",
        games = args.games,
        depth = args.depth
    );
    println!(
        "  Wins:   {wins} ({:.1}%)",
        100.0 * wins as f64 / args.games as f64
    );
    println!(
        "  Draws:  {draws} ({:.1}%)",
        100.0 * draws as f64 / args.games as f64
    );
    println!(
        "  Losses: {losses} ({:.1}%)",
        100.0 * losses as f64 / args.games as f64
    );

    Ok(())
}

/// Runs the benchmark subcommand.
///
/// Times training for a given number of episodes and reports throughput.
///
/// # Arguments
///
/// * `args` - Benchmark arguments from CLI.
///
/// # Errors
///
/// Returns an error on config/model failures.
pub fn run_benchmark(args: BenchmarkArgs) -> Result<(), Box<dyn std::error::Error>> {
    let agent = if let Some(path) = &args.model {
        let (agent, _) = load_agent(path, CpuLinAlg::new())?;
        agent
    } else {
        let config = AppConfig::default();
        let agent_config = config.to_agent_config()?;
        PcActorCritic::new(CpuLinAlg::new(), agent_config, 42)?
    };

    let config = AppConfig::default();
    let mut trainer = Trainer::new(agent, &config);

    let start = Instant::now();
    trainer.train(args.episodes);
    let elapsed = start.elapsed();

    let eps_per_sec = args.episodes as f64 / elapsed.as_secs_f64();
    println!(
        "Benchmark: {ep} episodes in {elapsed:.2?} ({eps_per_sec:.1} episodes/sec)",
        ep = args.episodes
    );

    Ok(())
}

/// Generates a default config.toml with optimal parameters.
///
/// # Arguments
///
/// * `args` - Init arguments from CLI.
///
/// # Errors
///
/// Returns an error if the output file cannot be written.
pub fn run_init(args: InitArgs) -> Result<(), Box<dyn std::error::Error>> {
    let config = DEFAULT_CONFIG_TOML;

    if Path::new(&args.output).exists() {
        eprintln!("Warning: {} already exists. Overwriting.", args.output);
    }

    std::fs::write(&args.output, config)?;
    println!("Config written to {}", args.output);
    Ok(())
}

/// Default configuration TOML with optimal parameters.
const DEFAULT_CONFIG_TOML: &str = r#"[agent]
gamma = 0.99
surprise_low = 0.02
surprise_high = 0.15
adaptive_surprise = true
surprise_buffer_size = 400
entropy_coeff = 0.0

# TD(n): n-step returns (0 = standard TD(0), n > 0 = accumulate n steps)
td_steps = 0

# Continuous Learning (all disabled by default for backward compatibility)
# M1 — Scale range: controls surprise-driven learning rate scaling
scale_floor = 0.0              # 0.0 = true weight freeze when surprise is low
scale_ceil = 2.0               # Max learning rate multiplier when surprise is high

# M2 — Hysteresis: dual-EWMA FROZEN/PLASTIC state machines
actor_hysteresis = false
actor_fast_window = 20
actor_slow_window = 100
actor_wake_fraction = 0.5
actor_sleep_fraction = 0.3
critic_hysteresis = false
critic_fast_window = 20
critic_slow_window = 100
critic_wake_fraction = 0.5
critic_sleep_fraction = 0.3
actor_wakes_critic = true
actor_wakes_critic_threshold = 1000
critic_wakes_actor = true
critic_wakes_actor_threshold = 1000

# M3 — Consolidation decay: per-layer learning rate modulation
consolidation_decay = 1.0      # 1.0 = no decay (all layers equal)
critic_consolidation_decay = 1.0
adaptive_consolidation = false
consolidation_ema_beta = 0.99
consolidation_sigmoid_k = 10.0
consolidation_error_threshold = 0.05

# M4 — EWC: Elastic Weight Consolidation
ewc_lambda = 0.0               # 0.0 = disabled (zero overhead)
fisher_decay = 0.9
fisher_ema_beta = 0.99
logits_reversal = false

[agent.actor]
input_size = 9
output_size = 9
output_activation = "linear"
alpha = 0.03
tol = 0.01
min_steps = 1
max_steps = 5
lr_weights = 0.005
synchronous = true
temperature = 1.0
local_lambda = 0.99
residual = false
rezero_init = 0.001

[[agent.actor.hidden_layers]]
size = 27
activation = "tanh"

[agent.critic]
# input_size = actor.input_size + sum(hidden layer sizes)
# 9 + 27 = 36
input_size = 36
output_activation = "linear"
lr = 0.005

[[agent.critic.hidden_layers]]
size = 36
activation = "tanh"

[training]
episodes = 50000
checkpoint_interval = 5000
log_interval = 500
seed = 42

[curriculum]
advance_threshold = 0.95
window_size = 1000

[continuous]
max_episodes = 50000
random_side = false            # true = random, false = alternating (even=P1, odd=P2)

[logger]
level = "info"
max_backups = 3
max_size = 10485760
"#;

/// Runs the seed-test subcommand.
///
/// Trains the same config across N random seeds to test statistical stability.
///
/// # Errors
///
/// Returns an error on config/IO/training failures.
pub fn run_seed_test(args: SeedTestArgs) -> Result<(), Box<dyn std::error::Error>> {
    use crate::training::experiment;

    let config = AppConfig::load(Path::new(&args.config))?;
    config.validate()?;

    let file = std::fs::File::create("experiment.txt")?;
    let stdout = io::stdout();
    let mut writer = MultiWriter {
        a: io::BufWriter::new(file),
        b: stdout.lock(),
    };

    let results = if args.continuous {
        experiment::run_seed_test_continuous(&config, args.n, &mut writer)?
    } else {
        experiment::run_seed_test(&config, args.n, &mut writer)?
    };

    let summary = format!(
        "\n=== SEED TEST ({} runs, lambda={:.8}) ===\n{:<6} {:<24} {:<10} {:<10} {:<10} {:<10}\n{}\n",
        results.len(),
        results.first().map(|r| r.lambda).unwrap_or(0.0),
        "run",
        "seed",
        "max_depth",
        "win%",
        "loss%",
        "draw%",
        results
            .iter()
            .map(|r| format!(
                "{:<6} {:<24} {:<10} {:<10.1} {:<10.1} {:<10.1}",
                r.run_number,
                r.seed,
                r.max_depth,
                r.win_rate * 100.0,
                r.loss_rate * 100.0,
                r.draw_rate * 100.0,
            ))
            .collect::<Vec<_>>()
            .join("\n"),
    );

    write!(writer, "{summary}")?;
    writer.flush()?;

    println!("\nResults saved to experiment.txt");
    Ok(())
}

/// Writer that duplicates output to two writers (file + stdout).
struct MultiWriter<A: io::Write, B: io::Write> {
    a: A,
    b: B,
}

impl<A: io::Write, B: io::Write> io::Write for MultiWriter<A, B> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.a.write_all(buf)?;
        self.b.write_all(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.a.flush()?;
        self.b.flush()
    }
}

/// Runs the experiment subcommand.
///
/// Loads config, runs N repetitions with lambda sweep [0.95..1.00],
/// writes results to experiment.txt and stdout.
///
/// # Arguments
///
/// * `args` - Experiment arguments from CLI.
///
/// # Errors
///
/// Returns an error on config/IO/training failures.
pub fn run_experiment(args: ExperimentArgs) -> Result<(), Box<dyn std::error::Error>> {
    use crate::training::experiment::{self, SweepParam};

    let config = AppConfig::load(Path::new(&args.config))?;
    config.validate()?;

    let sweep = match args.sweep.to_lowercase().as_str() {
        "lambda" => SweepParam::Lambda,
        other => {
            return Err(format!("Unknown sweep parameter '{other}'; expected 'lambda'").into())
        }
    };

    let file = std::fs::File::create("experiment.txt")?;
    let stdout = io::stdout();
    let mut writer = MultiWriter {
        a: io::BufWriter::new(file),
        b: stdout.lock(),
    };

    let results = experiment::run_experiment_sweep(&config, args.n, sweep, &mut writer)?;

    // Summary table
    let sweep_col = sweep.name();
    let summary = format!(
        "\n=== SUMMARY ({} runs, sweep={}) ===\n{:<6} {:<8} {:<8} {:<10} {:<10} {:<10} {:<10}\n{}\n",
        results.len(),
        sweep_col,
        "run",
        "seed",
        "lambda",
        "max_depth",
        "win%",
        "loss%",
        "draw%",
        results
            .iter()
            .map(|r| format!(
                "{:<6} {:<8} {:<8.2} {:<10} {:<10.1} {:<10.1} {:<10.1}",
                r.run_number,
                r.seed,
                r.lambda,
                r.max_depth,
                r.win_rate * 100.0,
                r.loss_rate * 100.0,
                r.draw_rate * 100.0,
            ))
            .collect::<Vec<_>>()
            .join("\n"),
    );

    write!(writer, "{summary}")?;
    writer.flush()?;

    println!("\nResults saved to experiment.txt");
    Ok(())
}

/// Runs the find-champion subcommand.
///
/// Loads config, runs N independent training sessions, and reports the
/// best champion found (highest fitness score).
///
/// # Arguments
///
/// * `args` - Find-champion arguments from CLI.
///
/// # Errors
///
/// Returns an error if config loading, training, or I/O fails.
pub fn run_find_champion(args: FindChampionArgs) -> Result<(), Box<dyn std::error::Error>> {
    use crate::training::champion::ChampionFinder;

    let mut config = AppConfig::load(Path::new(&args.config))?;

    if let Some(n) = args.iterations {
        config.champion.n_iterations = n;
    }

    config.validate()?;

    let stop_flag = Arc::new(AtomicBool::new(false));
    let flag = stop_flag.clone();
    let _ = ctrlc::set_handler(move || {
        flag.store(true, Ordering::SeqCst);
    });

    let mut finder = ChampionFinder::new(config, stop_flag);
    let result = finder.find()?;

    println!();
    println!("=== Champion Found ===");
    if result.champion_iteration == 0 {
        println!("No champion found (no iterations completed).");
    } else {
        println!("Fitness:   {:.4}", result.champion_fitness);
        println!("Depth:     {}", result.champion_depth);
        println!(
            "Iteration: {}/{}",
            result.champion_iteration,
            result.iterations.len()
        );
        println!("Seed:      {}", result.champion_seed);
    }
    println!("Iterations run: {}", result.iterations.len());

    Ok(())
}

/// Runs the stress-test subcommand.
///
/// Loads a champion model and subjects it to continuous training against
/// random-depth opponents, reporting fitness drift statistics.
///
/// # Arguments
///
/// * `args` - Stress-test arguments from CLI.
///
/// # Errors
///
/// Returns an error if config loading, champion loading, training, or I/O fails.
pub fn run_stress_test(args: StressTestArgs) -> Result<(), Box<dyn std::error::Error>> {
    use crate::training::stress_test::StressTester;

    let mut config = AppConfig::load(Path::new(&args.config))?;

    if let Some(path) = args.champion {
        config.stress_test.champion_path = path;
    }
    if let Some(m) = args.max_episodes {
        config.stress_test.max_episodes = m;
    }

    config.validate()?;

    let stop_flag = Arc::new(AtomicBool::new(false));
    let flag = stop_flag.clone();
    let _ = ctrlc::set_handler(move || {
        flag.store(true, Ordering::SeqCst);
    });

    let stress_cfg = config.stress_test.clone();
    let mut tester = StressTester::new(config, stress_cfg, stop_flag)?;
    let result = tester.run()?;

    println!();
    println!("=== Stress Test Summary ===");
    println!("Episodes run:     {}", result.total_episodes);
    println!("Baseline fitness: {:.4}", result.baseline_fitness);
    println!(
        "Final fitness:    {:.4} ({:+.4})",
        result.final_fitness,
        result.final_fitness - result.baseline_fitness
    );
    println!("Max fitness:      {:.4}", result.max_fitness);
    println!("Min fitness:      {:.4}", result.min_fitness);
    println!("Improvements:     {}", result.improvements);
    println!("Stable:           {}", result.stable);
    println!("Degradations:     {}", result.degradations);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cli_help_parses() {
        // Verify CLI struct can be constructed — clap derive is valid
        use clap::CommandFactory;
        let cmd = Cli::command();
        assert!(cmd.get_name() == "pc_tictactoe");
    }

    #[test]
    fn test_all_subcommands_have_help() {
        use clap::CommandFactory;
        let cmd = Cli::command();
        let subs: Vec<&str> = cmd.get_subcommands().map(|s| s.get_name()).collect();
        assert!(subs.contains(&"train"));
        assert!(subs.contains(&"play"));
        assert!(subs.contains(&"evaluate"));
        assert!(subs.contains(&"benchmark"));
        assert!(subs.contains(&"init"));
    }

    #[test]
    fn test_find_champion_subcommand_parses() {
        use clap::CommandFactory;
        let cmd = Cli::command();
        let subs: Vec<&str> = cmd.get_subcommands().map(|s| s.get_name()).collect();
        assert!(
            subs.contains(&"find-champion"),
            "Expected find-champion subcommand, got: {subs:?}"
        );
    }

    #[test]
    fn test_stress_test_subcommand_parses() {
        use clap::CommandFactory;
        let cmd = Cli::command();
        let subs: Vec<&str> = cmd.get_subcommands().map(|s| s.get_name()).collect();
        assert!(
            subs.contains(&"stress-test"),
            "Expected stress-test subcommand, got: {subs:?}"
        );
    }

    #[test]
    fn test_default_config_toml_is_valid() {
        let config: crate::utils::config::AppConfig = toml::from_str(DEFAULT_CONFIG_TOML).unwrap();
        assert!(config.validate().is_ok());
        let agent_config = config.to_agent_config();
        assert!(agent_config.is_ok());
    }

    #[test]
    fn test_default_config_toml_has_optimal_values() {
        let config: crate::utils::config::AppConfig = toml::from_str(DEFAULT_CONFIG_TOML).unwrap();
        assert_eq!(config.agent.actor.output_activation, "linear");
        assert!((config.agent.actor.alpha - 0.03).abs() < 1e-12);
        assert!((config.agent.actor.lr_weights - 0.005).abs() < 1e-12);
        assert!((config.agent.actor.local_lambda - 0.99).abs() < 1e-12);
        assert!(!config.agent.actor.residual);
        assert!((config.agent.actor.rezero_init - 0.001).abs() < 1e-12);
        assert_eq!(config.agent.actor.hidden_layers.len(), 1);
        assert_eq!(config.agent.actor.hidden_layers[0].size, 27);
        assert_eq!(config.agent.critic.input_size, 36);
    }

    #[test]
    fn test_default_config_toml_has_cl_fields() {
        let config: crate::utils::config::AppConfig = toml::from_str(DEFAULT_CONFIG_TOML).unwrap();
        assert!((config.agent.scale_floor - 0.0).abs() < 1e-12);
        assert!((config.agent.scale_ceil - 2.0).abs() < 1e-12);
        assert!(!config.agent.actor_hysteresis);
        assert!(!config.agent.critic_hysteresis);
        assert!((config.agent.ewc_lambda - 0.0).abs() < 1e-12);
        assert!((config.agent.consolidation_decay - 1.0).abs() < 1e-12);
    }
}
