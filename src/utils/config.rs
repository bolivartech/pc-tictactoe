// Author: Julian Bolivar
// Version: 1.0.0
// Date: 2026-03-25

//! TOML-based application configuration.
//!
//! Parses a TOML config file into [`AppConfig`], validates topology
//! consistency (critic input size must match actor input + latent size),
//! and converts to pc_core agent configuration types.

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

use pc_rl_core::activation::Activation;
use pc_rl_core::layer::LayerDef;
use pc_rl_core::mlp_critic::MlpCriticConfig;
use pc_rl_core::pc_actor::PcActorConfig;
use pc_rl_core::pc_actor_critic::PcActorCriticConfig;
use serde::Deserialize;

/// Top-level application configuration parsed from TOML.
///
/// Contains all settings for the agent, training loop, curriculum,
/// continuous learning, and logging.
///
/// # Examples
///
/// ```
/// use pc_tictactoe::utils::config::AppConfig;
///
/// let config = AppConfig::default();
/// assert!(config.validate().is_ok());
/// ```
#[derive(Debug, Clone, Default, Deserialize)]
pub struct AppConfig {
    /// Agent architecture configuration.
    #[serde(default)]
    pub agent: AgentSection,
    /// Training loop configuration.
    #[serde(default)]
    pub training: TrainingSection,
    /// Curriculum schedule configuration.
    #[serde(default)]
    pub curriculum: CurriculumSection,
    /// Continuous learning configuration.
    #[serde(default)]
    pub continuous: ContinuousSection,
    /// Logger configuration.
    #[serde(default)]
    pub logger: LoggerSection,
}

/// Agent architecture: actor, critic, and shared hyperparameters.
#[derive(Debug, Clone, Deserialize)]
pub struct AgentSection {
    /// Actor sub-configuration.
    #[serde(default)]
    pub actor: ActorSection,
    /// Critic sub-configuration.
    #[serde(default)]
    pub critic: CriticSection,
    /// Discount factor for returns.
    #[serde(default = "default_gamma")]
    pub gamma: f64,
    /// Surprise threshold below which learning rate is reduced.
    #[serde(default = "default_surprise_low")]
    pub surprise_low: f64,
    /// Surprise threshold above which learning rate is increased.
    #[serde(default = "default_surprise_high")]
    pub surprise_high: f64,
    /// Whether to adaptively recalibrate surprise thresholds. Default: true.
    #[serde(default = "default_adaptive_surprise")]
    pub adaptive_surprise: bool,
    /// Size of the circular buffer for adaptive surprise (default: 400).
    #[serde(default = "default_surprise_buffer_size")]
    pub surprise_buffer_size: usize,
    /// Entropy regularization coefficient.
    #[serde(default = "default_entropy_coeff")]
    pub entropy_coeff: f64,
    // ─── TD(n) ─────────────────────────────────────────────────────────────
    /// Number of steps for TD(n) return computation.
    /// 0 = standard TD(0), n > 0 = accumulate n steps before bootstrapping.
    #[serde(default)]
    pub td_steps: usize,
    // ─── Continuous Learning (M1–M4) ───────────────────────────────────────
    /// Learning rate floor when surprise is low. 0.0 = true weight freeze.
    #[serde(default = "default_scale_floor")]
    pub scale_floor: f64,
    /// Learning rate ceiling when surprise is high.
    #[serde(default = "default_scale_ceil")]
    pub scale_ceil: f64,
    /// Enable actor FROZEN/PLASTIC hysteresis state machine.
    #[serde(default)]
    pub actor_hysteresis: bool,
    /// Fast EWMA window for actor hysteresis.
    #[serde(default = "default_fast_window")]
    pub actor_fast_window: usize,
    /// Slow EWMA window for actor hysteresis.
    #[serde(default = "default_slow_window")]
    pub actor_slow_window: usize,
    /// Fraction above slow EWMA to wake actor (FROZEN -> PLASTIC).
    #[serde(default = "default_wake_fraction")]
    pub actor_wake_fraction: f64,
    /// Fraction below slow EWMA to sleep actor (PLASTIC -> FROZEN).
    #[serde(default = "default_sleep_fraction")]
    pub actor_sleep_fraction: f64,
    /// Enable critic FROZEN/PLASTIC hysteresis state machine.
    #[serde(default)]
    pub critic_hysteresis: bool,
    /// Fast EWMA window for critic hysteresis.
    #[serde(default = "default_fast_window")]
    pub critic_fast_window: usize,
    /// Slow EWMA window for critic hysteresis.
    #[serde(default = "default_slow_window")]
    pub critic_slow_window: usize,
    /// Fraction above slow EWMA to wake critic.
    #[serde(default = "default_wake_fraction")]
    pub critic_wake_fraction: f64,
    /// Fraction below slow EWMA to sleep critic.
    #[serde(default = "default_sleep_fraction")]
    pub critic_sleep_fraction: f64,
    /// When actor wakes, force critic to wake if frozen long enough.
    #[serde(default)]
    pub actor_wakes_critic: bool,
    /// Minimum critic frozen steps before actor-wakes-critic coupling triggers.
    #[serde(default = "default_actor_wakes_critic_threshold")]
    pub actor_wakes_critic_threshold: u64,
    /// When critic wakes (high |TD error|), force actor to wake if frozen long enough.
    #[serde(default = "default_true")]
    pub critic_wakes_actor: bool,
    /// Minimum actor frozen steps before critic-wakes-actor coupling triggers.
    #[serde(default = "default_critic_wakes_actor_threshold")]
    pub critic_wakes_actor_threshold: u64,
    /// Actor consolidation decay base. Layer i gets decay^(n_hidden-1-i).
    #[serde(default = "default_consolidation_decay")]
    pub consolidation_decay: f64,
    /// Critic consolidation decay base.
    #[serde(default = "default_consolidation_decay")]
    pub critic_consolidation_decay: f64,
    /// Enable adaptive sigmoid-driven consolidation decay for actor.
    #[serde(default)]
    pub adaptive_consolidation: bool,
    /// EMA smoothing for per-layer prediction error (M3b).
    #[serde(default = "default_consolidation_ema_beta")]
    pub consolidation_ema_beta: f64,
    /// Sigmoid steepness for adaptive consolidation.
    #[serde(default = "default_consolidation_sigmoid_k")]
    pub consolidation_sigmoid_k: f64,
    /// Sigmoid midpoint for adaptive consolidation.
    #[serde(default = "default_consolidation_error_threshold")]
    pub consolidation_error_threshold: f64,
    /// EWC regularization strength. 0.0 = disabled.
    #[serde(default)]
    pub ewc_lambda: f64,
    /// Fisher decay on FROZEN -> PLASTIC transitions.
    #[serde(default = "default_fisher_decay")]
    pub fisher_decay: f64,
    /// Fisher EMA smoothing.
    #[serde(default = "default_fisher_ema_beta")]
    pub fisher_ema_beta: f64,
    /// Use reversed logits for actor Fisher estimation.
    #[serde(default)]
    pub logits_reversal: bool,
}

/// Actor network configuration section.
#[derive(Debug, Clone, Deserialize)]
pub struct ActorSection {
    /// Number of input features.
    #[serde(default = "default_input_size")]
    pub input_size: usize,
    /// Hidden layer definitions.
    #[serde(default = "default_actor_hidden")]
    pub hidden_layers: Vec<HiddenLayerDef>,
    /// Number of output actions.
    #[serde(default = "default_output_size")]
    pub output_size: usize,
    /// Activation for the output layer.
    #[serde(default = "default_activation_tanh")]
    pub output_activation: String,
    /// PC inference update rate.
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    /// Convergence tolerance.
    #[serde(default = "default_tol")]
    pub tol: f64,
    /// Minimum inference steps.
    #[serde(default = "default_min_steps")]
    pub min_steps: usize,
    /// Maximum inference steps.
    #[serde(default = "default_max_steps")]
    pub max_steps: usize,
    /// Weight learning rate.
    #[serde(default = "default_lr_weights")]
    pub lr_weights: f64,
    /// Whether to use synchronous PC updates.
    #[serde(default = "default_true")]
    pub synchronous: bool,
    /// Softmax temperature for action selection.
    #[serde(default = "default_temperature")]
    pub temperature: f64,
    /// Blend factor: 1.0 = pure backprop, 0.0 = pure local PC, intermediate = hybrid.
    #[serde(default = "default_local_lambda")]
    pub local_lambda: f64,
    /// Enable residual skip connections between same-dimension hidden layers.
    #[serde(default)]
    pub residual: bool,
    /// Initial value for ReZero scaling factors on residual connections.
    #[serde(default = "default_rezero_init")]
    pub rezero_init: f64,
}

/// Default rezero_init: 0.001.
fn default_rezero_init() -> f64 {
    0.001
}

/// Default local_lambda: 1.0 (pure backprop).
fn default_local_lambda() -> f64 {
    1.0
}

/// Critic network configuration section.
#[derive(Debug, Clone, Deserialize)]
pub struct CriticSection {
    /// Dimensionality of critic input (board state + actor latent).
    #[serde(default = "default_critic_input_size")]
    pub input_size: usize,
    /// Hidden layer definitions.
    #[serde(default = "default_critic_hidden")]
    pub hidden_layers: Vec<HiddenLayerDef>,
    /// Activation for the output layer.
    #[serde(default = "default_activation_linear")]
    pub output_activation: String,
    /// Learning rate.
    #[serde(default = "default_critic_lr")]
    pub lr: f64,
}

/// Definition of a single hidden layer in TOML format.
#[derive(Debug, Clone, Deserialize)]
pub struct HiddenLayerDef {
    /// Number of neurons.
    pub size: usize,
    /// Activation function name (case-insensitive).
    pub activation: String,
}

/// Training loop configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct TrainingSection {
    /// Number of episodes to train.
    #[serde(default = "default_episodes")]
    pub episodes: usize,
    /// Checkpoint interval (episodes).
    #[serde(default = "default_checkpoint_interval")]
    pub checkpoint_interval: usize,
    /// How often to print progress stats (every N episodes).
    #[serde(default = "default_log_interval")]
    pub log_interval: usize,
    /// Random seed.
    #[serde(default = "default_seed")]
    pub seed: u64,
}

/// Curriculum schedule configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct CurriculumSection {
    /// Win rate threshold to advance difficulty.
    #[serde(default = "default_advance_threshold")]
    pub advance_threshold: f64,
    /// Window size for measuring win rate.
    #[serde(default = "default_window_size")]
    pub window_size: usize,
}

/// Continuous learning configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct ContinuousSection {
    /// Maximum episodes for continuous mode.
    #[serde(default = "default_max_episodes")]
    pub max_episodes: usize,
    /// Use random side assignment instead of alternating.
    #[serde(default)]
    pub random_side: bool,
}

/// Logger configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct LoggerSection {
    /// Minimum log level: "debug", "info", "training", "warning", "error".
    #[serde(default = "default_log_level")]
    pub level: String,
    /// Path to log file (optional).
    #[serde(default)]
    pub file: Option<String>,
    /// Path to CSV metrics file (optional).
    #[serde(default)]
    pub csv_file: Option<String>,
    /// Maximum number of log file backups for rotation.
    #[serde(default = "default_max_backups")]
    pub max_backups: usize,
    /// Maximum log file size in bytes before rotation.
    #[serde(default = "default_max_size")]
    pub max_size: u64,
}

/// Configuration validation error.
#[derive(Debug)]
pub struct ConfigError {
    /// Human-readable description of the validation failure.
    pub message: String,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "config validation error: {}", self.message)
    }
}

impl Error for ConfigError {}

// ─── Default value functions ────────────────────────────────────────────────

fn default_input_size() -> usize {
    9
}
fn default_output_size() -> usize {
    9
}
fn default_critic_input_size() -> usize {
    27
}
fn default_gamma() -> f64 {
    0.95
}
fn default_surprise_low() -> f64 {
    0.02
}
fn default_surprise_high() -> f64 {
    0.15
}
fn default_adaptive_surprise() -> bool {
    true
}
fn default_entropy_coeff() -> f64 {
    0.01
}
fn default_surprise_buffer_size() -> usize {
    400
}
fn default_alpha() -> f64 {
    0.1
}
fn default_tol() -> f64 {
    0.01
}
fn default_min_steps() -> usize {
    1
}
fn default_max_steps() -> usize {
    20
}
fn default_lr_weights() -> f64 {
    0.01
}
fn default_true() -> bool {
    true
}
fn default_temperature() -> f64 {
    1.0
}
fn default_critic_lr() -> f64 {
    0.005
}
fn default_episodes() -> usize {
    10000
}
fn default_checkpoint_interval() -> usize {
    1000
}
fn default_log_interval() -> usize {
    500
}
fn default_seed() -> u64 {
    42
}
fn default_advance_threshold() -> f64 {
    0.6
}
fn default_window_size() -> usize {
    100
}
fn default_max_episodes() -> usize {
    50000
}
fn default_log_level() -> String {
    "info".to_string()
}
fn default_max_backups() -> usize {
    3
}
fn default_max_size() -> u64 {
    10_485_760
}
fn default_activation_tanh() -> String {
    "tanh".to_string()
}
fn default_activation_linear() -> String {
    "linear".to_string()
}
fn default_scale_floor() -> f64 {
    0.0
}
fn default_scale_ceil() -> f64 {
    2.0
}
fn default_fast_window() -> usize {
    20
}
fn default_slow_window() -> usize {
    100
}
fn default_wake_fraction() -> f64 {
    0.5
}
fn default_sleep_fraction() -> f64 {
    0.3
}
fn default_actor_wakes_critic_threshold() -> u64 {
    1000
}
fn default_critic_wakes_actor_threshold() -> u64 {
    1000
}
fn default_consolidation_decay() -> f64 {
    1.0
}
fn default_consolidation_ema_beta() -> f64 {
    0.99
}
fn default_consolidation_sigmoid_k() -> f64 {
    10.0
}
fn default_consolidation_error_threshold() -> f64 {
    0.05
}
fn default_fisher_decay() -> f64 {
    0.9
}
fn default_fisher_ema_beta() -> f64 {
    0.99
}

fn default_actor_hidden() -> Vec<HiddenLayerDef> {
    vec![HiddenLayerDef {
        size: 18,
        activation: "tanh".to_string(),
    }]
}

fn default_critic_hidden() -> Vec<HiddenLayerDef> {
    vec![HiddenLayerDef {
        size: 36,
        activation: "tanh".to_string(),
    }]
}

// ─── Default trait impls ────────────────────────────────────────────────────

impl Default for AgentSection {
    fn default() -> Self {
        Self {
            actor: ActorSection::default(),
            critic: CriticSection::default(),
            gamma: default_gamma(),
            surprise_low: default_surprise_low(),
            surprise_high: default_surprise_high(),
            adaptive_surprise: default_adaptive_surprise(),
            surprise_buffer_size: default_surprise_buffer_size(),
            entropy_coeff: default_entropy_coeff(),
            td_steps: 0,
            scale_floor: default_scale_floor(),
            scale_ceil: default_scale_ceil(),
            actor_hysteresis: false,
            actor_fast_window: default_fast_window(),
            actor_slow_window: default_slow_window(),
            actor_wake_fraction: default_wake_fraction(),
            actor_sleep_fraction: default_sleep_fraction(),
            critic_hysteresis: false,
            critic_fast_window: default_fast_window(),
            critic_slow_window: default_slow_window(),
            critic_wake_fraction: default_wake_fraction(),
            critic_sleep_fraction: default_sleep_fraction(),
            actor_wakes_critic: true,
            actor_wakes_critic_threshold: default_actor_wakes_critic_threshold(),
            critic_wakes_actor: true,
            critic_wakes_actor_threshold: default_critic_wakes_actor_threshold(),
            consolidation_decay: default_consolidation_decay(),
            critic_consolidation_decay: default_consolidation_decay(),
            adaptive_consolidation: false,
            consolidation_ema_beta: default_consolidation_ema_beta(),
            consolidation_sigmoid_k: default_consolidation_sigmoid_k(),
            consolidation_error_threshold: default_consolidation_error_threshold(),
            ewc_lambda: 0.0,
            fisher_decay: default_fisher_decay(),
            fisher_ema_beta: default_fisher_ema_beta(),
            logits_reversal: false,
        }
    }
}

impl Default for ActorSection {
    fn default() -> Self {
        Self {
            input_size: default_input_size(),
            hidden_layers: default_actor_hidden(),
            output_size: default_output_size(),
            output_activation: default_activation_tanh(),
            alpha: default_alpha(),
            tol: default_tol(),
            min_steps: default_min_steps(),
            max_steps: default_max_steps(),
            lr_weights: default_lr_weights(),
            synchronous: default_true(),
            temperature: default_temperature(),
            local_lambda: default_local_lambda(),
            residual: false,
            rezero_init: default_rezero_init(),
        }
    }
}

impl Default for CriticSection {
    fn default() -> Self {
        Self {
            input_size: default_critic_input_size(),
            hidden_layers: default_critic_hidden(),
            output_activation: default_activation_linear(),
            lr: default_critic_lr(),
        }
    }
}

impl Default for TrainingSection {
    fn default() -> Self {
        Self {
            episodes: default_episodes(),
            checkpoint_interval: default_checkpoint_interval(),
            log_interval: default_log_interval(),
            seed: default_seed(),
        }
    }
}

impl Default for CurriculumSection {
    fn default() -> Self {
        Self {
            advance_threshold: default_advance_threshold(),
            window_size: default_window_size(),
        }
    }
}

impl Default for ContinuousSection {
    fn default() -> Self {
        Self {
            max_episodes: default_max_episodes(),
            random_side: false,
        }
    }
}

impl Default for LoggerSection {
    fn default() -> Self {
        Self {
            level: default_log_level(),
            file: None,
            csv_file: None,
            max_backups: default_max_backups(),
            max_size: default_max_size(),
        }
    }
}

// ─── Parsing helpers ────────────────────────────────────────────────────────

/// Parses an activation name string (case-insensitive) into an [`Activation`].
///
/// # Errors
///
/// Returns a [`ConfigError`] if the string does not match a known activation.
///
/// # Parameters
///
/// * `s` - Activation name (e.g. "tanh", "relu", "sigmoid", "elu", "linear").
fn parse_activation(s: &str) -> Result<Activation, ConfigError> {
    match s.to_lowercase().as_str() {
        "tanh" => Ok(Activation::Tanh),
        "relu" => Ok(Activation::Relu),
        "sigmoid" => Ok(Activation::Sigmoid),
        "elu" => Ok(Activation::Elu),
        "softsign" => Ok(Activation::Softsign),
        "linear" => Ok(Activation::Linear),
        other => Err(ConfigError {
            message: format!(
                "unknown activation '{other}'; expected tanh, relu, sigmoid, elu, softsign, or linear"
            ),
        }),
    }
}

/// Converts a slice of [`HiddenLayerDef`] to pc_core [`LayerDef`] values.
///
/// # Errors
///
/// Returns a [`ConfigError`] if any activation string is invalid.
fn convert_hidden(defs: &[HiddenLayerDef]) -> Result<Vec<LayerDef>, ConfigError> {
    defs.iter()
        .map(|h| {
            Ok(LayerDef {
                size: h.size,
                activation: parse_activation(&h.activation)?,
            })
        })
        .collect()
}

// ─── AppConfig methods ──────────────────────────────────────────────────────

impl AppConfig {
    /// Loads configuration from a TOML file, falling back to defaults if the
    /// file does not exist.
    ///
    /// # Parameters
    ///
    /// * `path` - Path to the TOML file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be parsed.
    pub fn load(path: &Path) -> Result<Self, Box<dyn Error>> {
        if !path.exists() {
            return Ok(Self::default());
        }
        let text = fs::read_to_string(path)?;
        let config: AppConfig = toml::from_str(&text)?;
        Ok(config)
    }

    /// Validates topology consistency.
    ///
    /// Checks that `critic.input_size == actor.input_size + sum(actor.hidden_layers[i].size)`.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] describing the mismatch.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let latent_sum: usize = self.agent.actor.hidden_layers.iter().map(|h| h.size).sum();
        let expected = self.agent.actor.input_size + latent_sum;
        let actual = self.agent.critic.input_size;
        if actual != expected {
            return Err(ConfigError {
                message: format!(
                    "critic.input_size ({actual}) != actor.input_size ({}) + sum(hidden sizes) ({latent_sum}) = {expected}",
                    self.agent.actor.input_size
                ),
            });
        }
        self.validate_cl()?;
        Ok(())
    }

    /// Validates continuous learning field combinations.
    ///
    /// Checks that CL fields form valid combinations: scale range is
    /// non-inverted, hysteresis windows are ordered, decay factors are
    /// in bounds, and EWC parameters are non-negative.
    ///
    /// # Errors
    ///
    /// Returns a [`ConfigError`] describing the invalid combination.
    fn validate_cl(&self) -> Result<(), ConfigError> {
        let a = &self.agent;

        // M1: scale range
        if a.scale_floor > a.scale_ceil {
            return Err(ConfigError {
                message: format!(
                    "scale_floor ({}) > scale_ceil ({})",
                    a.scale_floor, a.scale_ceil
                ),
            });
        }
        if a.scale_floor < 0.0 {
            return Err(ConfigError {
                message: format!("scale_floor ({}) must be >= 0.0", a.scale_floor),
            });
        }

        // M2: hysteresis windows and fractions
        if a.actor_hysteresis {
            if a.actor_fast_window >= a.actor_slow_window {
                return Err(ConfigError {
                    message: format!(
                        "actor_fast_window ({}) must be < actor_slow_window ({})",
                        a.actor_fast_window, a.actor_slow_window
                    ),
                });
            }
            if a.actor_wake_fraction <= 0.0 {
                return Err(ConfigError {
                    message: format!(
                        "actor_wake_fraction ({}) must be > 0.0",
                        a.actor_wake_fraction
                    ),
                });
            }
            if a.actor_sleep_fraction <= 0.0 || a.actor_sleep_fraction >= 1.0 {
                return Err(ConfigError {
                    message: format!(
                        "actor_sleep_fraction ({}) must be in (0.0, 1.0)",
                        a.actor_sleep_fraction
                    ),
                });
            }
        }
        if a.critic_hysteresis {
            if a.critic_fast_window >= a.critic_slow_window {
                return Err(ConfigError {
                    message: format!(
                        "critic_fast_window ({}) must be < critic_slow_window ({})",
                        a.critic_fast_window, a.critic_slow_window
                    ),
                });
            }
            if a.critic_wake_fraction <= 0.0 {
                return Err(ConfigError {
                    message: format!(
                        "critic_wake_fraction ({}) must be > 0.0",
                        a.critic_wake_fraction
                    ),
                });
            }
            if a.critic_sleep_fraction <= 0.0 || a.critic_sleep_fraction >= 1.0 {
                return Err(ConfigError {
                    message: format!(
                        "critic_sleep_fraction ({}) must be in (0.0, 1.0)",
                        a.critic_sleep_fraction
                    ),
                });
            }
        }

        // M3: consolidation decay
        if a.consolidation_decay <= 0.0 || a.consolidation_decay > 1.0 {
            return Err(ConfigError {
                message: format!(
                    "consolidation_decay ({}) must be in (0.0, 1.0]",
                    a.consolidation_decay
                ),
            });
        }
        if a.critic_consolidation_decay <= 0.0 || a.critic_consolidation_decay > 1.0 {
            return Err(ConfigError {
                message: format!(
                    "critic_consolidation_decay ({}) must be in (0.0, 1.0]",
                    a.critic_consolidation_decay
                ),
            });
        }

        // M4: EWC
        if a.ewc_lambda < 0.0 {
            return Err(ConfigError {
                message: format!("ewc_lambda ({}) must be >= 0.0", a.ewc_lambda),
            });
        }
        if a.fisher_decay < 0.0 || a.fisher_decay > 1.0 {
            return Err(ConfigError {
                message: format!("fisher_decay ({}) must be in [0.0, 1.0]", a.fisher_decay),
            });
        }
        if a.fisher_ema_beta <= 0.0 || a.fisher_ema_beta >= 1.0 {
            return Err(ConfigError {
                message: format!(
                    "fisher_ema_beta ({}) must be in (0.0, 1.0)",
                    a.fisher_ema_beta
                ),
            });
        }

        Ok(())
    }

    /// Converts the TOML config to a pc_core [`PcActorCriticConfig`].
    ///
    /// Calls [`validate`](AppConfig::validate) first, then parses activation
    /// strings into enum variants.
    ///
    /// # Errors
    ///
    /// Returns an error if validation fails or an activation string is unknown.
    pub fn to_agent_config(&self) -> Result<PcActorCriticConfig, Box<dyn Error>> {
        self.validate()?;
        let actor_hidden = convert_hidden(&self.agent.actor.hidden_layers)?;
        let critic_hidden = convert_hidden(&self.agent.critic.hidden_layers)?;
        let output_act = parse_activation(&self.agent.actor.output_activation)?;
        let critic_output_act = parse_activation(&self.agent.critic.output_activation)?;

        Ok(PcActorCriticConfig {
            actor: PcActorConfig {
                input_size: self.agent.actor.input_size,
                hidden_layers: actor_hidden,
                output_size: self.agent.actor.output_size,
                output_activation: output_act,
                alpha: self.agent.actor.alpha,
                tol: self.agent.actor.tol,
                min_steps: self.agent.actor.min_steps,
                max_steps: self.agent.actor.max_steps,
                lr_weights: self.agent.actor.lr_weights,
                synchronous: self.agent.actor.synchronous,
                temperature: self.agent.actor.temperature,
                local_lambda: self.agent.actor.local_lambda,
                residual: self.agent.actor.residual,
                rezero_init: self.agent.actor.rezero_init,
            },
            critic: MlpCriticConfig {
                input_size: self.agent.critic.input_size,
                hidden_layers: critic_hidden,
                output_activation: critic_output_act,
                lr: self.agent.critic.lr,
            },
            gamma: self.agent.gamma,
            surprise_low: self.agent.surprise_low,
            surprise_high: self.agent.surprise_high,
            adaptive_surprise: self.agent.adaptive_surprise,
            surprise_buffer_size: self.agent.surprise_buffer_size,
            entropy_coeff: self.agent.entropy_coeff,
            td_steps: self.agent.td_steps,
            scale_floor: self.agent.scale_floor,
            scale_ceil: self.agent.scale_ceil,
            actor_hysteresis: self.agent.actor_hysteresis,
            actor_fast_window: self.agent.actor_fast_window,
            actor_slow_window: self.agent.actor_slow_window,
            actor_wake_fraction: self.agent.actor_wake_fraction,
            actor_sleep_fraction: self.agent.actor_sleep_fraction,
            critic_hysteresis: self.agent.critic_hysteresis,
            critic_fast_window: self.agent.critic_fast_window,
            critic_slow_window: self.agent.critic_slow_window,
            critic_wake_fraction: self.agent.critic_wake_fraction,
            critic_sleep_fraction: self.agent.critic_sleep_fraction,
            actor_wakes_critic: self.agent.actor_wakes_critic,
            actor_wakes_critic_threshold: self.agent.actor_wakes_critic_threshold,
            critic_wakes_actor: self.agent.critic_wakes_actor,
            critic_wakes_actor_threshold: self.agent.critic_wakes_actor_threshold,
            consolidation_decay: self.agent.consolidation_decay,
            critic_consolidation_decay: self.agent.critic_consolidation_decay,
            adaptive_consolidation: self.agent.adaptive_consolidation,
            consolidation_ema_beta: self.agent.consolidation_ema_beta,
            consolidation_sigmoid_k: self.agent.consolidation_sigmoid_k,
            consolidation_error_threshold: self.agent.consolidation_error_threshold,
            ewc_lambda: self.agent.ewc_lambda,
            fisher_decay: self.agent.fisher_decay,
            fisher_ema_beta: self.agent.fisher_ema_beta,
            logits_reversal: self.agent.logits_reversal,
        })
    }

    /// Applies CLI overrides to the configuration.
    ///
    /// Only overrides fields for which a `Some` value is provided.
    ///
    /// # Parameters
    ///
    /// * `episodes` - Override for `training.episodes`.
    /// * `seed` - Override for `training.seed`.
    pub fn apply_cli_overrides(&mut self, episodes: Option<usize>, seed: Option<u64>) {
        if let Some(ep) = episodes {
            self.training.episodes = ep;
        }
        if let Some(s) = seed {
            self.training.seed = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_passes_validation() {
        let config = AppConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_inconsistent_critic_input_fails_validation() {
        let mut config = AppConfig::default();
        config.agent.critic.input_size = 999;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_to_agent_config_succeeds_for_valid_config() {
        let config = AppConfig::default();
        let agent_config = config.to_agent_config();
        assert!(agent_config.is_ok());
        let ac = agent_config.unwrap();
        assert_eq!(ac.actor.input_size, 9);
        assert_eq!(ac.actor.output_size, 9);
        assert_eq!(ac.critic.input_size, 27);
    }

    #[test]
    fn test_to_agent_config_fails_for_invalid_config() {
        let mut config = AppConfig::default();
        config.agent.critic.input_size = 10;
        assert!(config.to_agent_config().is_err());
    }

    #[test]
    fn test_toml_deserialization_default_config_succeeds() {
        let toml_str = r#"
[agent]
gamma = 0.99

[agent.actor]
input_size = 9
output_size = 9
output_activation = "tanh"

[[agent.actor.hidden_layers]]
size = 18
activation = "tanh"

[agent.critic]
input_size = 27

[[agent.critic.hidden_layers]]
size = 36
activation = "tanh"

[training]
episodes = 5000
"#;
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.agent.gamma, 0.99);
        assert_eq!(config.training.episodes, 5000);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_two_hidden_layers_validates_correctly() {
        let mut config = AppConfig::default();
        config.agent.actor.hidden_layers = vec![
            HiddenLayerDef {
                size: 18,
                activation: "tanh".to_string(),
            },
            HiddenLayerDef {
                size: 12,
                activation: "relu".to_string(),
            },
        ];
        // critic input must be input_size + 18 + 12 = 39
        config.agent.critic.input_size = 39;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_unknown_activation_string_returns_error() {
        let result = parse_activation("swish");
        assert!(result.is_err());
    }

    #[test]
    fn test_load_nonexistent_file_returns_default() {
        let config = AppConfig::load(Path::new("nonexistent_config_file.toml")).unwrap();
        assert!(config.validate().is_ok());
        assert_eq!(config.agent.actor.input_size, 9);
    }

    #[test]
    fn test_cli_override_episodes_wins_over_toml() {
        let mut config = AppConfig::default();
        assert_eq!(config.training.episodes, 10000);
        config.apply_cli_overrides(Some(500), None);
        assert_eq!(config.training.episodes, 500);
    }

    #[test]
    fn test_cli_override_none_does_not_change_value() {
        let mut config = AppConfig::default();
        let original = config.training.episodes;
        config.apply_cli_overrides(None, None);
        assert_eq!(config.training.episodes, original);
    }

    #[test]
    fn test_default_actor_section_residual_false() {
        let config = AppConfig::default();
        assert!(!config.agent.actor.residual);
        assert!((config.agent.actor.rezero_init - 0.001).abs() < 1e-12);
    }

    #[test]
    fn test_to_agent_config_passes_residual_fields() {
        let mut config = AppConfig::default();
        config.agent.actor.residual = true;
        config.agent.actor.rezero_init = 0.01;
        config.agent.actor.hidden_layers = vec![
            HiddenLayerDef {
                size: 27,
                activation: "tanh".to_string(),
            },
            HiddenLayerDef {
                size: 27,
                activation: "tanh".to_string(),
            },
        ];
        config.agent.critic.input_size = 63; // 9 + 27 + 27
        let ac = config.to_agent_config().unwrap();
        assert!(ac.actor.residual);
        assert!((ac.actor.rezero_init - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_cl_fields_parse_from_toml() {
        let toml_str = r#"
[agent]
gamma = 0.99
scale_floor = 0.0
scale_ceil = 2.0
actor_hysteresis = true
actor_fast_window = 20
actor_slow_window = 100
actor_wake_fraction = 0.5
actor_sleep_fraction = 0.3
critic_hysteresis = false
actor_wakes_critic = true
actor_wakes_critic_threshold = 500
consolidation_decay = 0.95
critic_consolidation_decay = 0.9
adaptive_consolidation = true
consolidation_ema_beta = 0.99
consolidation_sigmoid_k = 10.0
consolidation_error_threshold = 0.05
ewc_lambda = 0.1
fisher_decay = 0.9
fisher_ema_beta = 0.99
logits_reversal = false

[agent.actor]
input_size = 9
output_size = 9
output_activation = "tanh"

[[agent.actor.hidden_layers]]
size = 18
activation = "tanh"

[agent.critic]
input_size = 27

[[agent.critic.hidden_layers]]
size = 36
activation = "tanh"

[training]
episodes = 5000
"#;
        let config: AppConfig = toml::from_str(toml_str).unwrap();
        assert!((config.agent.scale_floor - 0.0).abs() < 1e-12);
        assert!((config.agent.scale_ceil - 2.0).abs() < 1e-12);
        assert!(config.agent.actor_hysteresis);
        assert_eq!(config.agent.actor_fast_window, 20);
        assert_eq!(config.agent.actor_slow_window, 100);
        assert!((config.agent.actor_wake_fraction - 0.5).abs() < 1e-12);
        assert!((config.agent.actor_sleep_fraction - 0.3).abs() < 1e-12);
        assert!(!config.agent.critic_hysteresis);
        assert!(config.agent.actor_wakes_critic);
        assert_eq!(config.agent.actor_wakes_critic_threshold, 500);
        assert!((config.agent.consolidation_decay - 0.95).abs() < 1e-12);
        assert!((config.agent.critic_consolidation_decay - 0.9).abs() < 1e-12);
        assert!(config.agent.adaptive_consolidation);
        assert!((config.agent.consolidation_ema_beta - 0.99).abs() < 1e-12);
        assert!((config.agent.consolidation_sigmoid_k - 10.0).abs() < 1e-12);
        assert!((config.agent.consolidation_error_threshold - 0.05).abs() < 1e-12);
        assert!((config.agent.ewc_lambda - 0.1).abs() < 1e-12);
        assert!((config.agent.fisher_decay - 0.9).abs() < 1e-12);
        assert!((config.agent.fisher_ema_beta - 0.99).abs() < 1e-12);
        assert!(!config.agent.logits_reversal);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cl_fields_default_to_disabled() {
        let config = AppConfig::default();
        assert!((config.agent.scale_floor - 0.0).abs() < 1e-12);
        assert!((config.agent.scale_ceil - 2.0).abs() < 1e-12);
        assert!(!config.agent.actor_hysteresis);
        assert!(!config.agent.critic_hysteresis);
        assert!(config.agent.actor_wakes_critic);
        assert!(config.agent.critic_wakes_actor);
        assert!((config.agent.consolidation_decay - 1.0).abs() < 1e-12);
        assert!((config.agent.critic_consolidation_decay - 1.0).abs() < 1e-12);
        assert!(!config.agent.adaptive_consolidation);
        assert!((config.agent.ewc_lambda - 0.0).abs() < 1e-12);
        assert!(!config.agent.logits_reversal);
    }

    #[test]
    fn test_to_agent_config_passes_cl_fields() {
        let mut config = AppConfig::default();
        config.agent.scale_floor = 0.1;
        config.agent.scale_ceil = 3.0;
        config.agent.actor_hysteresis = true;
        config.agent.actor_fast_window = 30;
        config.agent.ewc_lambda = 0.5;
        config.agent.consolidation_decay = 0.8;
        let ac = config.to_agent_config().unwrap();
        assert!((ac.scale_floor - 0.1).abs() < 1e-12);
        assert!((ac.scale_ceil - 3.0).abs() < 1e-12);
        assert!(ac.actor_hysteresis);
        assert_eq!(ac.actor_fast_window, 30);
        assert!((ac.ewc_lambda - 0.5).abs() < 1e-12);
        assert!((ac.consolidation_decay - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_cl_validation_scale_floor_gt_ceil_fails() {
        let mut config = AppConfig::default();
        config.agent.scale_floor = 3.0;
        config.agent.scale_ceil = 2.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cl_validation_negative_scale_floor_fails() {
        let mut config = AppConfig::default();
        config.agent.scale_floor = -0.1;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cl_validation_fast_window_gte_slow_fails() {
        let mut config = AppConfig::default();
        config.agent.actor_hysteresis = true;
        config.agent.actor_fast_window = 100;
        config.agent.actor_slow_window = 50;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cl_validation_zero_wake_fraction_fails() {
        let mut config = AppConfig::default();
        config.agent.actor_hysteresis = true;
        config.agent.actor_wake_fraction = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cl_validation_consolidation_decay_out_of_range_fails() {
        let mut config = AppConfig::default();
        config.agent.consolidation_decay = 1.5;
        assert!(config.validate().is_err());

        config.agent.consolidation_decay = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cl_validation_negative_ewc_lambda_fails() {
        let mut config = AppConfig::default();
        config.agent.ewc_lambda = -0.01;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cl_validation_fisher_ema_beta_out_of_range_fails() {
        let mut config = AppConfig::default();
        config.agent.fisher_ema_beta = 1.0;
        assert!(config.validate().is_err());

        config.agent.fisher_ema_beta = 0.0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_cl_validation_valid_cl_config_passes() {
        let mut config = AppConfig::default();
        config.agent.actor_hysteresis = true;
        config.agent.critic_hysteresis = true;
        config.agent.scale_floor = 0.0;
        config.agent.scale_ceil = 2.0;
        config.agent.consolidation_decay = 0.95;
        config.agent.ewc_lambda = 0.1;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cl_validation_disabled_hysteresis_skips_window_check() {
        let mut config = AppConfig::default();
        // Invalid windows, but hysteresis is disabled so should pass
        config.agent.actor_hysteresis = false;
        config.agent.actor_fast_window = 200;
        config.agent.actor_slow_window = 10;
        assert!(config.validate().is_ok());
    }
}
