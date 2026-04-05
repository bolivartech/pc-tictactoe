# PC-TicTacToe

[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange?logo=rust)](https://www.rust-lang.org/)
[![CI](https://github.com/BolivarTech/PC-TicTacToe/actions/workflows/ci.yml/badge.svg)](https://github.com/BolivarTech/PC-TicTacToe/actions)
[![pc-rl-core](https://img.shields.io/crates/v/pc-rl-core?label=pc-rl-core&color=blue)](https://crates.io/crates/pc-rl-core)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-green.svg)](LICENSE-MIT)

A **Deliberative Predictive Coding (DPC)** reinforcement learning agent that learns to play Tic-Tac-Toe from scratch, using the [pc-rl-core](https://crates.io/crates/pc-rl-core) framework.

The agent trains via REINFORCE with baseline against minimax opponents using curriculum learning. It achieves near-perfect play (minimax depth 9) with only ~550-1,000 parameters — **4-330x smaller** than typical published architectures for the same task.

## Results

The agent reaches **minimax depth 9** (near-perfect play) in **23% of seeds** with a 3-layer `[27,27,18]` architecture, ultra-low PC error (`local_lambda=0.9999`), and adaptive surprise scheduling.

At depth 9, the agent achieves **~99% draws** against a near-perfect minimax opponent — essentially optimal play for Tic-Tac-Toe.

### Statistical Validation (N=35 seeds, 20 phases, ~3,800 runs)

| Topology | Lambda | Activation | Residual | Surprise | Episodes | Mean Depth | D=9 (functional) |
|----------|--------|------------|----------|----------|----------|:----------:|:-----------------:|
| **[27,27,18]** | **0.9999** | **softsign** | **yes** | **adaptive buf=400** | **200k** | **7.63** | **23%** |
| [27,27,18] | 0.9999 | softsign | yes | fixed | 200k | 7.69 | 14% |
| 1x27 | 0.99 | tanh | no | fixed | 50k | 7.94 | 37% |
| 1x27 | 0.99 | softsign | no | fixed | 50k | 7.89 | 31% |

> **Phase 20 discovery**: The baseline 40% D=9 rate was misleading — 64% of those models collapsed immediately after advancing (100% loss rate). Adaptive surprise with `buffer=400` eliminates most collapses and increases functional D=9 from 14% to 23%.

## Architecture

```
Input (9) ──> [H1 27, Softsign] ──> [H2 27, Softsign] ──> [H3 18, Softsign] ──> [Output 9, Linear] ──> Softmax ──> Action
                   ^    |     skip        skip+proj
                   |    v
               PC Inference Loop (top-down / bottom-up)
                   |
                   v
             Latent Concat (27+27+18 = 72)
                   |
          [Board State (9)] ++ [Latent (72)] = Critic Input (81)
                   |
                   v
          [Critic Hidden 36, Softsign] ──> V(s)
```

**Curriculum Learning**: The agent starts against a weak opponent (minimax depth 1) and advances when it achieves >95% non-loss rate over a 1,000-game window. Metrics reset on each advancement to prevent cascading depth jumps.

## Quick Start

### Prerequisites

- [Rust](https://rustup.rs/) 1.70 or later

### Build and Run

```bash
# Build
cargo build --release

# Train with config file
cargo run --release -- train -c config.toml

# Play against the trained agent
cargo run --release -- play --model model.json

# Play as first player
cargo run --release -- play --model model.json --first

# Evaluate against minimax
cargo run --release -- evaluate --model model.json --games 100 --depth 9
```

### Experiments

```bash
# Run seed-test (multiple seeds, same config)
cargo run --release -- seed-test -c config.toml --seeds 35

# Run parameter sweep experiment
cargo run --release -- experiment -c config.toml

# Initialize default config
cargo run --release -- init
```

## Configuration

All hyperparameters are configured via TOML. See [`config.toml`](config.toml) for the full configuration with optimal parameters.

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| `output_activation` | `linear` | Unbounded logits for softmax (tanh prevents learning) |
| `alpha` | `0.03` | PC inference loop update rate |
| `lr_weights` | `0.005` | Actor learning rate |
| `hidden_layers` | `[27,27,18]` | 3-layer softsign with dimensionality reduction |
| `residual` | `true` | Skip connections with ReZero + projection |
| `local_lambda` | `0.9999` | Ultra-low PC error for deep networks (`0.99` for 1-layer) |
| `adaptive_surprise` | `true` | Dynamic surprise thresholds from recent history |
| `surprise_buffer_size` | `400` | Circular buffer size (~0.4x curriculum window) |
| `gamma` | `0.99` | Discount factor |
| `entropy_coeff` | `0.0` | No entropy regularization |

## Key Findings

- **Adaptive surprise eliminates D=9 collapse** — 64% of fixed-threshold D=9 models were collapsed; adaptive `buffer=400` raises functional D=9 from 14% to 23%.
- **Optimal buffer ratio: 0.3-0.4x curriculum window** — `buffer=400` with `window=1000` is the sweet spot.
- **Depth-Lambda Scaling Law** — `lambda = 1 - 10^(-(L+1))` where `L` = number of hidden layers.
- **Deliberation is the primary advantage** — PC inference loop adds +2-3 depth levels over pure MLP.
- **Bounded activations required** — ReLU dies, ELU explodes; tanh and softsign work with the PC loop.
- **Parameter efficiency** — ~550 actor parameters matching networks 4-330x larger through iterative inference.

> Validated through 20 experimental phases and ~3,800 training runs. Full analysis available in the parent project's [experiment_analysis.md](https://github.com/BolivarTech/PC-RL-Core/blob/main/docs/experiment_analysis.md).

## Project Structure

```
pc_tictactoe/
├── Cargo.toml              # Depends on pc-rl-core from crates.io
├── config.toml             # Training configuration (optimal parameters)
└── src/
    ├── main.rs             # Entry point
    ├── env/
    │   ├── tictactoe.rs    # Game rules, board state, valid actions
    │   └── minimax.rs      # Minimax opponent (depth 1-9)
    ├── training/
    │   ├── trainer.rs      # Episode-based training loop
    │   ├── continuous.rs   # Continuous learning with surprise scheduling
    │   └── experiment.rs   # Parameter sweep and seed-test runners
    ├── ui/
    │   └── cli.rs          # CLI interface (clap)
    └── utils/
        ├── config.rs       # TOML configuration parsing
        ├── logger.rs       # Training logger
        └── metrics.rs      # Performance metrics tracking
```

## Dependencies

| Crate | Purpose |
|-------|---------|
| [pc-rl-core](https://crates.io/crates/pc-rl-core) | DPC reinforcement learning framework |
| [clap](https://crates.io/crates/clap) | CLI argument parsing |
| [serde](https://crates.io/crates/serde) / [serde_json](https://crates.io/crates/serde_json) | Serialization |
| [toml](https://crates.io/crates/toml) | Configuration parsing |
| [rand](https://crates.io/crates/rand) | Random number generation |
| [chrono](https://crates.io/crates/chrono) | Timestamps |
| [ctrlc](https://crates.io/crates/ctrlc) | Graceful shutdown |

## Related Projects

- **[PC-RL-Core](https://github.com/BolivarTech/PC-RL-Core)** — The library crate that provides the Deliberative Predictive Coding Actor-Critic framework.

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
