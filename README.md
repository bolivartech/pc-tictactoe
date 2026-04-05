# PC-TicTacToe

A **Deliberative Predictive Coding (DPC)** reinforcement learning agent that learns to play Tic-Tac-Toe from scratch, using the [pc-rl-core](https://crates.io/crates/pc-rl-core) framework.

The agent trains via REINFORCE with baseline against minimax opponents with curriculum learning. It achieves near-perfect play (minimax depth 9) with only ~550-1,000 parameters -- 4-330x smaller than typical published architectures for the same task.

## Results

The agent reaches **minimax depth 9** (near-perfect play) in **23% of seeds** (functional models) with a 3-layer [27,27,18] architecture, ultra-low PC error (`local_lambda=0.9999`), and adaptive surprise scheduling.

At depth 9, the agent achieves **~99% draws** against a near-perfect minimax opponent -- essentially optimal play for Tic-Tac-Toe.

### Statistical Validation (N=35 seeds, 20 phases, ~3,800 runs)

| Topology | Lambda | Activation | Residual | Surprise | Episodes | Mean | D=9 (functional) |
|----------|--------|------------|----------|----------|----------|------|-------------------|
| **[27,27,18]** | **0.9999** | **softsign** | **yes (proj)** | **adaptive buf=400** | **200k** | **7.63** | **23%** |
| [27,27,18] | 0.9999 | softsign | yes (proj) | fixed | 200k | 7.69 | 14% |
| 1x27 | 0.99 | tanh | no | fixed | 50k | 7.94 | 37% |
| 1x27 | 0.99 | softsign | no | fixed | 50k | 7.89 | 31% |

**Phase 20 discovery**: The baseline 40% D=9 rate was misleading -- 64% of those models had 100% loss rate (collapsed immediately after advancing). Adaptive surprise with buffer=400 eliminates most collapses and increases functional D=9 from 14% to 23%, producing 3 perfect-play models (including one with theoretically optimal 50W/0L/50D).

## Architecture

```
Input (9) --> [H1 27, Softsign] --> [H2 27, Softsign] --> [H3 18, Softsign] --> [Output 9, Linear] --> Softmax --> Action
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
         [Critic Hidden 36, Softsign] --> V(s)
```

**Curriculum Learning**: The agent starts against a weak opponent (minimax depth 1) and advances when it achieves >95% non-loss rate over a 1000-game window. Metrics reset on each advancement to prevent cascading.

## Quick Start

```bash
# Build
cargo build --release

# Train (uses config.toml)
cargo run --release -- train -c config.toml

# Play against the trained agent
cargo run --release -- play --model model.json

# Play as first player
cargo run --release -- play --model model.json --first

# Evaluate against minimax
cargo run --release -- evaluate --model model.json --games 100 --depth 9

# Run seed-test (multiple seeds, same config)
cargo run --release -- seed-test -c config.toml --seeds 35

# Initialize default config
cargo run --release -- init
```

## Configuration

All hyperparameters are configured via TOML. See `config.toml` for the full configuration with optimal parameters.

Key parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `output_activation` | `linear` | Unbounded logits for softmax (tanh prevents learning) |
| `alpha` | `0.03` | PC inference loop update rate |
| `lr_weights` | `0.005` | Actor learning rate |
| `hidden_layers` | `[27,27,18] softsign` | 3-layer with dimensionality reduction |
| `residual` | `true` | Skip connections with ReZero + projection |
| `rezero_init` | `0.1` | ReZero initial scaling factor |
| `gamma` | `0.99` | Discount factor |
| `entropy_coeff` | `0.0` | No entropy regularization |
| `local_lambda` | `0.9999` | Ultra-low PC error for deep networks (0.99 for 1-layer) |
| `adaptive_surprise` | `true` | Dynamic surprise thresholds from recent history |
| `surprise_buffer_size` | `400` | Circular buffer size (~0.4x curriculum window) |

## Key Findings

- **Adaptive surprise eliminates D=9 collapse** -- 64% of fixed-threshold D=9 models were collapsed (100% loss); adaptive buffer=400 raises functional D=9 from 14% to 23%
- **Buffer-mediated transition damping** -- circular buffer of recent surprise scores creates a decaying LR envelope during curriculum transitions, protecting learned representations
- **Optimal buffer ratio: 0.3-0.4 x curriculum window** -- buffer=400 with window=1000 is the sweet spot; too small (100) resonates, too large (500) over-damps
- **3-layer [27,27,18] with lambda=0.9999** -- best deep configuration with adaptive surprise
- **Depth-Lambda Scaling Law: `lambda = 1 - 10^(-(L+1))`** -- PC error must decrease exponentially with network depth
- **Lambda and training budget interact** -- lambda=0.9999 needs 200k episodes (6% D=9 at 50k, 40% nominal at 200k)
- **Deliberation is the primary advantage** -- PC inference loop adds +2-3 depth levels over MLP
- **Output activation must be Linear** -- Tanh bounds logits to [-1,1], preventing policy learning
- **Bounded activations required for PC** -- ReLU dies, ELU explodes; tanh and softsign work
- **Parameter efficiency** -- ~550 actor parameters matching networks 4-330x larger through iterative inference

Validated through 20 experimental phases, ~3,800 training runs across multiple architectural configurations.

## Dependencies

```toml
[dependencies]
pc-rl-core = "1.2.1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rand = "0.8"
chrono = { version = "0.4", features = ["serde"] }
toml = "0.8"
clap = { version = "4", features = ["derive"] }
ctrlc = "3"
```

## License

Licensed under either of

- Apache License, Version 2.0
- MIT License

at your option.
