# Continuous Learning Experiment Results

**Date**: 2026-04-11
**Branch**: `continuos_learning`
**Config**: `config_cl_experiment.toml`
**Command**: `cargo run --release -- seed-test -c config_cl_experiment.toml -n 35 --continuous`

## Experiment Setup

### Network Architecture

| Parameter | Value |
|-----------|-------|
| Hidden layers | 1x27 tanh |
| Output activation | linear |
| Alpha (PC inference) | 0.03 |
| lr_weights | 0.005 |
| local_lambda | 0.99 |
| Critic input | 36 (9 board + 27 latent) |
| Critic hidden | 1x36 tanh |

### Training Parameters

| Parameter | Value |
|-----------|-------|
| Training mode | Continuous (`step_masked()` TD(0)) |
| Episodes | 50,000 per seed |
| Seeds | 35 random |
| Curriculum threshold | 0.95 non-loss rate |
| Curriculum window | 1,000 games |
| Gamma | 0.99 |

### CL Features (all enabled)

| Module | Feature | Value |
|--------|---------|-------|
| M1 | scale_floor / scale_ceil | 0.0 / 2.0 |
| M2 | actor_hysteresis | true (fast=20, slow=100, wake=0.5, sleep=0.3) |
| M2 | critic_hysteresis | true (fast=20, slow=100, wake=0.5, sleep=0.3) |
| M2 | actor_wakes_critic | true (threshold=1000) |
| M3a | consolidation_decay | 0.95 (actor and critic) |
| M3b | adaptive_consolidation | true (beta=0.99, k=10.0, threshold=0.05) |
| M4 | ewc_lambda | 0.1 (fisher_decay=0.9, fisher_ema_beta=0.99) |

## Results Summary

### CL Continuous vs Episodic Baseline

| Metric | CL Continuous | Episodic Baseline | Delta |
|--------|---------------|-------------------|-------|
| **Mean depth** | **6.29** | **7.57** | **-1.28** |
| StdDev | 1.23 | 0.81 | +0.42 |
| Min | 3 | 7 | -4 |
| Max | 8 | 9 | -1 |
| D>=7 | 62.9% | 100% | -37.1% |
| D>=8 | 5.7% | 37% | -31.3% |
| D=9 | 0% | 20% | -20% |

*Episodic baseline: N=35, lambda=0.99, 50k episodes, 1x27 tanh, REINFORCE with trajectory-based `learn()`.*

### Depth Distribution

| Depth | Count | Percentage |
|-------|-------|------------|
| D=3 | 1 | 2.9% |
| D=4 | 4 | 11.4% |
| D=5 | 3 | 8.6% |
| D=6 | 5 | 14.3% |
| D=7 | 20 | 57.1% |
| D=8 | 2 | 5.7% |
| D=9 | 0 | 0.0% |

### Health Metrics

| Metric | Value |
|--------|-------|
| Collapsed (>80% loss at final depth) | 1 / 35 (2.9%) |
| Functional (<=55% loss at final depth) | 27 / 35 (77.1%) |
| Stalled (D<=5) | 8 / 35 (22.9%) |

## Per-Seed Results

| Seed | Max Depth | Win% | Loss% | Draw% | Notes |
|------|-----------|------|-------|-------|-------|
| 6645124718816394247 | 7 | 0.1 | 59.6 | 40.3 | |
| 11219180285506193958 | 7 | 0.1 | 58.5 | 41.4 | |
| 23927817162983146 | 4 | 26.1 | 27.6 | 46.3 | Stalled: offensive bias |
| 4897142006134589999 | 5 | 31.4 | 54.9 | 13.7 | Stalled: offensive bias |
| 4554404362814281 | 7 | 0.9 | 55.1 | 44.0 | |
| 16859009791432043429 | 7 | 0.0 | 51.7 | 48.3 | |
| 17971049575214323992 | 6 | 5.5 | 44.8 | 49.7 | |
| 13797571969144865060 | 7 | 0.1 | 52.2 | 47.7 | |
| 1695373398328840348 | 6 | 0.0 | 51.0 | 49.0 | |
| 805221039274623130 | 7 | 0.0 | 50.9 | 49.1 | |
| 2318973924464539610 | 8 | 0.0 | 52.3 | 47.7 | Best |
| 14525894584004716136 | 5 | 46.3 | 53.3 | 0.4 | Stalled: offensive bias |
| 3068745675708347934 | 7 | 0.1 | 53.0 | 46.9 | |
| 4008320363505152839 | 7 | 48.7 | 51.0 | 0.3 | High win at D=7 |
| 16210669821402741639 | 7 | 0.0 | 51.3 | 48.7 | |
| 9553558552089701331 | 5 | 70.8 | 27.0 | 2.2 | Stalled: extreme offensive |
| 15183479633034911737 | 7 | 42.9 | 53.5 | 3.6 | High win at D=7 |
| 480887532581385318 | 6 | 0.0 | 81.0 | 19.0 | Collapsed |
| 18390290873261938391 | 4 | 47.7 | 51.9 | 0.4 | Stalled: offensive bias |
| 239801581935950227 | 4 | 49.7 | 14.4 | 35.9 | Stalled: offensive bias |
| 12483189733405065676 | 7 | 0.0 | 54.6 | 45.4 | |
| 4100217076796200422 | 6 | 0.0 | 58.4 | 41.6 | |
| 14067414058479696177 | 7 | 40.5 | 54.9 | 4.6 | High win at D=7 |
| 2099609757702465012 | 7 | 0.1 | 53.9 | 46.0 | |
| 6418458053078907113 | 7 | 0.1 | 52.3 | 47.6 | |
| 7117657137829777681 | 7 | 0.1 | 52.2 | 47.7 | |
| 3662609802039661085 | 3 | 48.9 | 6.2 | 44.9 | Worst: stuck at D=2 for 36k ep |
| 13456868398758840634 | 7 | 0.0 | 52.2 | 47.8 | |
| 12793444700205814014 | 7 | 0.0 | 52.5 | 47.5 | |
| 478404055364062305 | 4 | 86.9 | 12.9 | 0.2 | Stalled: extreme offensive |
| 5931819216632671145 | 7 | 0.0 | 52.0 | 48.0 | |
| 17871589518245824121 | 7 | 11.8 | 58.5 | 29.7 | |
| 13545193994923750391 | 8 | 0.2 | 58.0 | 41.8 | Best tied |
| 10750144139721905048 | 6 | 40.5 | 56.7 | 2.8 | High win at D=6 |
| 16477540924195691722 | 7 | 0.0 | 52.2 | 47.8 | |

## Failure Mode Analysis

### Pattern 1: Offensive Bias (8/35 runs stalled at D<=5)

Stalled runs show high win rate (40-87%) with persistent loss rate that never drops below the 5% threshold needed for advancement. The agent learns to **win** but not to **defend**.

**Example** — Seed `3662609802039661085` (worst, D=3):
- Stuck at depth 2 for 36,000 episodes
- Win rate oscillated 40-47%, loss rate never below 6%
- At depth 2, minimax is beatable by offense alone, so the agent never receives gradient signal to learn defensive play
- Hysteresis may freeze offensive weights before defensive play is learned
- EWC anchors to the offensive solution, penalizing defensive weight changes

**Example** — Seed `478404055364062305` (D=4, 86.9% win):
- Extreme offensive specialization: nearly 87% win rate at final depth
- 12.9% loss rate prevents curriculum advancement
- 0.2% draw rate means the agent almost never plays defensively

### Pattern 2: D=7 Ceiling (20/35 runs)

Most successful runs reach D=7 with ~0% win / ~52% loss / ~48% draw. This is the same ceiling pattern seen in episodic training, but the episodic baseline sometimes breaks through (20% reach D=9). The CL features may be over-protecting weights learned at earlier depths, preventing the fine adjustments needed for deeper play.

### Pattern 3: Slow Convergence

Runs that do advance show slower curriculum progression:
- Depth 1→2: ~3-4k episodes (vs ~2.5k episodic)
- Depth 2→3: ~8-11k episodes (vs ~4-5k episodic)

TD(0) per-step learning produces noisier gradient signal than full-trajectory REINFORCE, especially in short games (5-9 steps).

## Diagnosis

The CL features with these hyperparameters **degrade** performance compared to the episodic baseline. Contributing factors ranked by suspected impact:

### 1. EWC Over-Protection (ewc_lambda=0.1)

EWC anchors weights to solutions learned at earlier curriculum depths. When the agent advances from depth N to depth N+1, it needs to significantly adjust its policy — but EWC penalizes deviations from the depth-N solution. This is the opposite of what CL is designed for: CL protects against catastrophic forgetting across **different tasks**, but curriculum levels in TicTacToe are the **same task** at increasing difficulty, not different tasks.

### 2. Consolidation Decay Inappropriate for 1-Layer Network

`consolidation_decay=0.95` applies per-layer learning rate modulation where deeper layers learn faster. With a single hidden layer there is no depth hierarchy to exploit — the decay just reduces the learning rate by 5% for no benefit.

### 3. Hysteresis Premature Freezing

When the agent reaches a local optimum at a given depth (surprise drops), hysteresis transitions to FROZEN state. But curriculum advancement introduces a harder opponent, requiring significant policy changes. The FROZEN→PLASTIC wake transition depends on surprise exceeding the slow EWMA — but if the agent's offense still works against the new depth, surprise may not spike enough to wake learning.

### 4. TD(0) vs REINFORCE for Short Episodes

TicTacToe episodes are 5-9 steps. TD(0) learns from individual transitions (state→action→reward→next_state), while REINFORCE uses the complete trajectory return. For short episodes, the full return carries richer signal. TD(0)'s advantage (lower variance, faster per-step updates) may not offset the loss of trajectory-level credit assignment.

## Recommendations for Follow-Up Experiments

### Experiment 2: TD(0) Baseline (no CL features)

Isolate whether the performance gap comes from TD(0) vs REINFORCE or from the CL features:

```toml
# All CL disabled, just continuous TD(0) training
scale_floor = 0.0
scale_ceil = 2.0
actor_hysteresis = false
critic_hysteresis = false
consolidation_decay = 1.0
adaptive_consolidation = false
ewc_lambda = 0.0
```

```bash
cargo run --release -- seed-test -c config_td0_baseline.toml -n 35 --continuous
```

### Experiment 3: M1+M2 Only (no consolidation, no EWC)

Test whether hysteresis alone helps or hurts:

```toml
scale_floor = 0.0
scale_ceil = 2.0
actor_hysteresis = true
critic_hysteresis = true
consolidation_decay = 1.0
adaptive_consolidation = false
ewc_lambda = 0.0
```

### Experiment 4: Lower EWC (if EWC is the bottleneck)

```toml
ewc_lambda = 0.01             # 10x lower
# or
ewc_lambda = 0.001            # 100x lower
```

### Experiment 5: Multi-Layer Network with CL

The CL features (especially M3 consolidation and M4 EWC) are designed for deeper networks. Test with the 3-layer [27,27,18] softsign architecture where per-layer differentiation matters:

```toml
local_lambda = 0.9999
residual = true
rezero_init = 0.1
hidden_layers = [
    { size = 27, activation = "softsign" },
    { size = 27, activation = "softsign" },
    { size = 18, activation = "softsign" },
]
# With CL at reduced strength
ewc_lambda = 0.01
consolidation_decay = 0.95
adaptive_consolidation = true
```

Episodes should be increased to 200k to match the episodic 3-layer baseline.

## Conclusion

The first CL experiment establishes that enabling all CL features (M1-M4) simultaneously on a single-layer network with default hyperparameters produces **worse results** than the episodic baseline (mean depth 6.29 vs 7.57). The primary failure modes are offensive bias from EWC over-protection and slow convergence from TD(0) per-step learning.

The next step is to isolate variables: first test TD(0) without CL features (experiment 2), then add features incrementally (experiments 3-4), and finally test on the deeper architecture where CL features are architecturally motivated (experiment 5).
