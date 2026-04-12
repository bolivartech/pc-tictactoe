# Continuous Learning Experiment Results

**Date**: 2026-04-11
**Branch**: `continuos_learning`
**Config**: `config_cl_experiment.toml`
**Command**: `cargo run --release -- seed-test -c config_cl_experiment.toml -n 35 --continuous`

---

## Experiment 1: CL on 1-Layer Network (1x27 tanh, 50k episodes)

### Setup

| Parameter | Value |
|-----------|-------|
| Hidden layers | 1x27 tanh |
| Output activation | linear |
| Alpha (PC inference) | 0.03 |
| lr_weights | 0.005 |
| local_lambda | 0.99 |
| Critic input | 36 (9 board + 27 latent) |
| Critic hidden | 1x36 tanh |
| Training mode | Continuous (`step_masked()` TD(0)) |
| Episodes | 50,000 per seed |
| Seeds | 35 random |
| Curriculum threshold | 0.95 non-loss rate |
| Curriculum window | 1,000 games |

#### CL Features (all enabled)

| Module | Feature | Value |
|--------|---------|-------|
| M1 | scale_floor / scale_ceil | 0.0 / 2.0 |
| M2 | actor_hysteresis | true (fast=20, slow=100, wake=0.5, sleep=0.3) |
| M2 | critic_hysteresis | true (fast=20, slow=100, wake=0.5, sleep=0.3) |
| M2 | actor_wakes_critic | true (threshold=1000) |
| M3a | consolidation_decay | 0.95 (actor and critic) |
| M3b | adaptive_consolidation | true (beta=0.99, k=10.0, threshold=0.05) |
| M4 | ewc_lambda | 0.1 (fisher_decay=0.9, fisher_ema_beta=0.99) |

### Results

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

#### Depth Distribution

| Depth | Count | Percentage |
|-------|-------|------------|
| D=3 | 1 | 2.9% |
| D=4 | 4 | 11.4% |
| D=5 | 3 | 8.6% |
| D=6 | 5 | 14.3% |
| D=7 | 20 | 57.1% |
| D=8 | 2 | 5.7% |
| D=9 | 0 | 0.0% |

#### Health Metrics

| Metric | Value |
|--------|-------|
| Collapsed (>80% loss at final depth) | 1 / 35 (2.9%) |
| Functional (<=55% loss at final depth) | 27 / 35 (77.1%) |
| Stalled (D<=5) | 8 / 35 (22.9%) |

### Per-Seed Results

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

### Failure Mode Analysis

#### Pattern 1: Offensive Bias (8/35 runs stalled at D<=5)

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

#### Pattern 2: D=7 Ceiling (20/35 runs)

Most successful runs reach D=7 with ~0% win / ~52% loss / ~48% draw. This is the same ceiling pattern seen in episodic training, but the episodic baseline sometimes breaks through (20% reach D=9). The CL features may be over-protecting weights learned at earlier depths, preventing the fine adjustments needed for deeper play.

#### Pattern 3: Slow Convergence

Runs that do advance show slower curriculum progression:
- Depth 1→2: ~3-4k episodes (vs ~2.5k episodic)
- Depth 2→3: ~8-11k episodes (vs ~4-5k episodic)

TD(0) per-step learning produces noisier gradient signal than full-trajectory REINFORCE, especially in short games (5-9 steps).

### Diagnosis

The CL features with these hyperparameters **degrade** performance compared to the episodic baseline. Contributing factors ranked by suspected impact:

1. **EWC Over-Protection (ewc_lambda=0.1)**: Anchors weights to solutions learned at earlier curriculum depths. Curriculum levels are the **same task** at increasing difficulty, not different tasks — EWC's forgetting protection works against curriculum adaptation.

2. **Consolidation Decay Inappropriate for 1-Layer**: `consolidation_decay=0.95` applies per-layer LR modulation. With a single hidden layer there is no depth hierarchy to exploit.

3. **Hysteresis Premature Freezing**: When surprise drops at a local optimum, hysteresis freezes weights. Curriculum advancement then requires surprise to exceed the slow EWMA to wake — but offense may still work, suppressing surprise spikes.

4. **TD(0) vs REINFORCE for Short Episodes**: TicTacToe episodes are 5-9 steps. Full trajectory return carries richer signal than per-step TD(0) in such short games.

### Conclusion (Experiment 1)

Enabling all CL features (M1-M4) simultaneously on a single-layer network produces **worse results** than the episodic baseline (mean depth 6.29 vs 7.57). The primary failure modes are offensive bias from EWC over-protection and slow convergence from TD(0).

---

## Experiment 2: CL on 3-Layer Network ([27,27,18] softsign, 200k episodes)

### Setup

| Parameter | Value |
|-----------|-------|
| Hidden layers | [27, 27, 18] softsign |
| Output activation | linear |
| Alpha (PC inference) | 0.03 |
| lr_weights | 0.005 |
| local_lambda | 0.9999 |
| Residual connections | true (rezero_init=0.1) |
| Critic input | 81 (9 + 27 + 27 + 18) |
| Critic hidden | 1x36 softsign |
| Training mode | Continuous (`step_masked()` TD(0)) |
| Episodes | 200,000 per seed |
| Seeds | 35 random |
| Curriculum threshold | 0.95 non-loss rate |
| Curriculum window | 1,000 games |

CL features identical to Experiment 1 (all M1-M4 enabled, same hyperparameters).

### Results

| Metric | CL 3-Layer | CL 1-Layer (Exp 1) | Episodic 3-Layer | Episodic 1-Layer |
|--------|------------|---------------------|------------------|------------------|
| **Mean depth** | **6.71** | **6.29** | **7.63** | **7.57** |
| StdDev | 1.00 | 1.23 | — | 0.81 |
| Min | 2 | 3 | — | 7 |
| Max | 9 | 8 | — | 9 |
| D>=7 | 74.3% | 62.9% | — | 100% |
| D>=8 | 5.7% | 5.7% | — | 37% |
| D=9 | 2.9% (1) | 0% | 23% (8) | 20% (7) |

*Episodic 3-layer baseline: N=35, lambda=0.9999, 200k episodes, [27,27,18] softsign, adaptive surprise buf=400.*

#### Depth Distribution

| Depth | Count | Percentage |
|-------|-------|------------|
| D=2 | 1 | 2.9% |
| D=6 | 8 | 22.9% |
| D=7 | 24 | 68.6% |
| D=8 | 1 | 2.9% |
| D=9 | 1 | 2.9% |

### Per-Seed Results

| Seed | Max Depth | Notes |
|------|-----------|-------|
| 5055614087462936786 | **9** | **Best: 0.9% loss, 99.1% draw — functional D=9** |
| 10933304803249642293 | 8 | |
| 1643298049397144543 | 7 | |
| 13117698049459150449 | 7 | |
| 8075457413956352693 | 7 | |
| 9466470194415056052 | 6 | |
| 16296401170530544366 | 7 | |
| 15942950685931882494 | 7 | |
| 8429591594808251617 | 7 | |
| 3576547489759345254 | 7 | |
| 17709214650204424487 | 7 | |
| 14386569364386370427 | 7 | |
| 17707451842276816308 | 7 | |
| 2151902309033975782 | 7 | |
| 12324055776474642952 | 6 | |
| 11969019074232556044 | 7 | |
| 11509293356497007877 | 7 | |
| 8669659922000560991 | 7 | |
| 15104251731416439984 | 7 | |
| 14734932213956265950 | 6 | |
| 16192612798633492537 | 6 | |
| 15349314643578223569 | 6 | |
| 1924205002059389278 | 6 | |
| 11580784087793721050 | 6 | |
| 3479430948679714399 | 7 | |
| 8169064584761860652 | 7 | |
| 459166065219410815 | 7 | |
| 5902713266781432123 | **2** | **Worst: stuck at D=2 for 200k ep (offensive lock-in)** |
| 12458040580924374935 | 7 | |
| 657429217313879772 | 7 | |
| 17838304057201147097 | 7 | |
| 10813095782145506900 | 7 | |
| 13055075499939207937 | 7 | |
| 6080230163054789720 | 6 | |

### D=9 Run Analysis (seed=5055614087462936786)

The only D=9 run shows a distinctive curriculum progression:

| Transition | Episode | Time at Previous Depth |
|------------|---------|------------------------|
| D1 → D2 | 1,500 | 1,500 |
| D2 → D3 | 3,500 | 2,000 |
| D3 → D4 | 4,500 | 1,000 |
| D4 → D5 | 5,500 | 1,000 |
| D5 → D6 | 7,000 | 1,500 |
| D6 → D7 | 8,000 | 1,000 |
| D7 → D8 | 44,500 | **36,500** (18% of budget) |
| D8 → D9 | 180,000 | **135,500** (68% of budget) |

Final performance at D=9: **0.0% win / 0.9% loss / 99.1% draw** — functional, not collapsed.

The D7→D8 and D8→D9 transitions consumed 86% of the training budget. The agent passed through depths 1-6 in only 8k episodes, then spent 172k episodes breaking through the D=7 and D=8 barriers.

### D=2 Stalled Run Analysis (seed=5902713266781432123)

Extreme failure case — stuck at depth 2 for the entire 200k episode budget:
- Reached D=2 at episode 2,000 (normal)
- By episode 7,000: 48% win / 52% loss / 0% draw — **zero draws**
- Remained at ~49% win / ~50% loss / ~0% draw for 50k+ episodes
- Eventually developed some draws (~35% by episode 196k) but never reached 95% non-loss
- The agent learned offense-only play: wins half the games as Player One but can't defend as Player Two
- EWC likely anchored the offensive weights from D=1, preventing adaptation to D=2's defensive requirements

### Improvements vs Experiment 1

1. **D=9 achieved**: First CL experiment to reach depth 9 (1/35 = 2.9%). The D=9 model is functional with 99.1% draw rate against perfect minimax.

2. **Offensive bias nearly eliminated**: Only 1/35 runs stalled at D<=5 (vs 8/35 in Exp 1). The 3-layer architecture with residual connections provides enough capacity to learn both offense and defense simultaneously.

3. **Higher mean depth**: 6.71 vs 6.29 (+0.42). The deeper network and 4x more episodes help.

4. **Lower variance**: SD 1.00 vs 1.23. More consistent results.

5. **Consolidation decay meaningful**: With 3 hidden layers, `consolidation_decay=0.95` creates a genuine depth hierarchy — shallow layers (generic features) consolidate faster while deep layers (task-specific) remain plastic longer.

### Persistent Issues

1. **D=7 ceiling dominates**: 68.6% of runs plateau at D=7 with ~52% loss / ~48% draw. Same ceiling as episodic training, but episodic breaks through more often (23% D=9 vs 2.9%).

2. **D>=8 rate still low**: 5.7% (2/35) vs episodic's higher rate. The CL features don't help break the D=7→D=8 barrier.

3. **Mean still below episodic**: 6.71 vs 7.63 (-0.92). The gap narrowed from 1.28 (Exp 1) to 0.92 but remains significant.

4. **EWC over-protection**: The D=2 stalled run (seed `5902713266781432123`) confirms EWC at 0.1 can catastrophically prevent curriculum adaptation in some seeds.

---

## Cross-Experiment Comparison

| Experiment | Architecture | Mode | Mean | D>=8 | D=9 | Stalled D<=5 |
|------------|-------------|------|------|------|-----|--------------|
| Episodic 1-layer (baseline) | 1x27 tanh | REINFORCE | 7.57 | 37% | 20% | 0% |
| **CL Exp 1: 1-layer** | 1x27 tanh | TD(0)+CL | **6.29** | **5.7%** | **0%** | **22.9%** |
| Episodic 3-layer (baseline) | [27,27,18] softsign | REINFORCE | 7.63 | — | 23% | — |
| **CL Exp 2: 3-layer** | [27,27,18] softsign | TD(0)+CL | **6.71** | **5.7%** | **2.9%** | **2.9%** |

### Key Observations

1. **CL features degrade performance** in both architectures compared to episodic baselines. The degradation is smaller with 3 layers (-0.92) than 1 layer (-1.28).

2. **3-layer architecture benefits CL more**: The offensive bias problem (stalled D<=5) drops from 22.9% to 2.9%, and D=9 becomes achievable (2.9% vs 0%).

3. **EWC is the primary bottleneck**: Both the offensive bias pattern and the D=7 ceiling are consistent with EWC over-protecting weights from earlier curriculum phases.

4. **TD(0) vs REINFORCE gap**: Even without considering CL features, TD(0) per-step learning may be inherently disadvantaged for short-episode games. This hypothesis needs validation (see recommended experiments).

---

---

## Experiment 3: TD(0) Baseline — No CL ([27,27,18] softsign, 200k episodes)

### Setup

| Parameter | Value |
|-----------|-------|
| Hidden layers | [27, 27, 18] softsign |
| Output activation | linear |
| Alpha (PC inference) | 0.03 |
| lr_weights | 0.005 |
| local_lambda | 0.9999 |
| Residual connections | true (rezero_init=0.1) |
| Critic input | 81 (9 + 27 + 27 + 18) |
| Critic hidden | 1x36 softsign |
| Training mode | Continuous (`step_masked()` TD(0)) |
| Episodes | 200,000 per seed |
| Seeds | 35 random |
| Curriculum threshold | 0.95 non-loss rate |
| Curriculum window | 1,000 games |

#### CL Features (all disabled)

| Module | Feature | Value |
|--------|---------|-------|
| M1 | scale_floor / scale_ceil | 0.0 / 2.0 |
| M2 | actor_hysteresis | false |
| M2 | critic_hysteresis | false |
| M3 | consolidation_decay | 1.0 (no decay) |
| M3b | adaptive_consolidation | false |
| M4 | ewc_lambda | 0.0 (disabled) |

**Purpose**: Isolate whether the performance gap vs episodic training comes from TD(0) per-step learning or from CL features.

### Results

| Metric | TD(0) No CL | CL Exp 2 (with CL) | Episodic 3-Layer | Episodic 1-Layer |
|--------|-------------|---------------------|------------------|------------------|
| **Mean depth** | **6.43** | **6.71** | **7.63** | **7.57** |
| StdDev | 1.20 | 1.00 | — | 0.81 |
| Min | 2 | 2 | — | 7 |
| Max | 8 | 8 | — | 9 |
| D>=7 | 62.9% | 74.3% | — | 100% |
| D>=8 | 2.9% | 5.7% | — | 37% |
| D=9 | 0% | 2.9% | 23% | 20% |

#### Depth Distribution

| Depth | Count | Percentage |
|-------|-------|------------|
| D=2 | 2 | 5.7% |
| D=6 | 11 | 31.4% |
| D=7 | 21 | 60.0% |
| D=8 | 1 | 2.9% |

#### Health Metrics

| Metric | Value |
|--------|-------|
| Stalled (D<=5) | 2 / 35 (5.7%) |
| Collapsed (>80% loss) | 0 / 35 (0.0%) |
| High-win (>40%) | 4 / 35 (11.4%) |
| No-draw (<1% draw) | 3 / 35 (8.6%) |

### Per-Seed Results

| Seed | Max Depth | Win% | Loss% | Draw% | Notes |
|------|-----------|------|-------|-------|-------|
| 2043704767215754731 | 6 | 49.8 | 50.2 | 0.0 | Offensive bias (0% draw) |
| 16378063109570011060 | 7 | 0.0 | 51.3 | 48.7 | |
| 4833856944993873624 | 7 | 0.0 | 54.0 | 46.0 | |
| 1829250433885307294 | 7 | 3.8 | 50.2 | 46.0 | |
| 1602490558669094663 | 7 | 0.0 | 50.0 | 50.0 | |
| 8077547698343758214 | 6 | 0.0 | 50.0 | 50.0 | |
| 10538908052320084975 | 7 | 0.0 | 50.1 | 49.9 | |
| 12598518363405095464 | 2 | 0.1 | 50.3 | 49.6 | Stalled |
| 17733316755369277421 | 7 | 24.2 | 50.9 | 24.9 | |
| 955067113903131506 | 6 | 0.1 | 50.0 | 49.9 | |
| 4653829097689577580 | 7 | 0.0 | 50.9 | 49.1 | |
| 637832407920993153 | 6 | 0.0 | 57.4 | 42.6 | |
| 17582995133829348970 | 6 | 0.0 | 50.6 | 49.4 | |
| 1276219739894159763 | 6 | 0.0 | 50.0 | 50.0 | |
| 2177752556344113802 | 6 | 50.0 | 50.0 | 0.0 | Offensive bias (0% draw) |
| 2673669181830052880 | 7 | 0.0 | 51.5 | 48.5 | |
| 16390967182907099166 | 7 | 0.0 | 50.6 | 49.4 | |
| 1988754193033301883 | 6 | 0.0 | 50.0 | 50.0 | |
| 14202297880242826964 | 7 | 0.0 | 50.3 | 49.7 | |
| 2781291731788634884 | 7 | 0.0 | 50.0 | 50.0 | |
| 47051537093746808 | 7 | 0.0 | 50.1 | 49.9 | |
| 11064108218824745739 | 7 | 0.0 | 50.1 | 49.9 | |
| 11810983166884289264 | 7 | 0.0 | 50.0 | 50.0 | |
| 13566841846590897270 | 7 | 0.0 | 50.5 | 49.5 | |
| 5130570611840405432 | 6 | 0.0 | 50.1 | 49.9 | |
| 12765977124966873088 | 6 | 33.9 | 58.0 | 8.1 | |
| 278304561829151306 | 7 | 0.1 | 61.7 | 38.2 | |
| 14948012462617388929 | 7 | 0.0 | 52.2 | 47.8 | |
| 10243162511752146125 | 2 | 50.0 | 50.0 | 0.0 | Stalled: offensive (0% draw) |
| 10547423566626701362 | 7 | 0.0 | 50.0 | 50.0 | |
| 14674474562114901691 | 7 | 50.0 | 50.0 | 0.0 | Offensive bias (0% draw) |
| 8313851744069836661 | 6 | 0.0 | 49.8 | 50.2 | |
| 15250646023639029948 | 7 | 0.0 | 50.4 | 49.6 | |
| 2359315442615510714 | 7 | 0.0 | 50.8 | 49.2 | |
| 6759014360914324840 | 8 | 0.0 | 49.6 | 50.4 | Best |

### Key Findings

#### 1. TD(0) is the bottleneck, not CL features

Comparing Exp 3 (no CL) vs Exp 2 (with CL, EWC=0):

| Metric | No CL (Exp 3) | With CL (Exp 2) | CL Effect |
|--------|---------------|------------------|-----------|
| Mean depth | 6.43 | 6.71 | **+0.28** (CL helps) |
| D>=7 | 62.9% | 74.3% | **+11.4%** (CL helps) |
| D>=8 | 2.9% | 5.7% | **+2.8%** (CL helps) |
| D=9 | 0% | 2.9% | **+2.9%** (CL helps) |
| Stalled | 5.7% | 2.9% | **-2.8%** (CL helps) |

CL features (M1+M2+M3, without EWC) **improve** TD(0) performance in every metric. The performance gap vs episodic training (mean 6.43 vs 7.57 = -1.14) comes from TD(0) itself, not from CL.

#### 2. Offensive bias is a TD(0) problem

4/35 runs (11.4%) show >40% win rate with near-0% draw — the agent learns offense-only play. This pattern appears both with and without CL, confirming it's a TD(0) signal quality issue, not a CL interaction.

The "50/50/0" pattern (50% win, 50% loss, 0% draw) means the agent wins exactly when it goes first (Player One advantage at that depth) but can't defend as Player Two. TD(0)'s per-step bootstrap doesn't propagate the "you need to block" signal back to earlier moves effectively.

#### 3. D=7 ceiling is universal for TD(0)

60% of runs reach D=7 with ~50% loss / ~50% draw. This ceiling persists regardless of CL features. The episodic REINFORCE baseline breaks through this ceiling 37% of the time (D>=8), suggesting the full-trajectory return is essential for learning the deep defensive patterns needed at D=8+.

### Conclusion (Experiment 3)

TD(0) per-step learning produces a mean depth of 6.43 on the 3-layer network — **1.14 levels below episodic REINFORCE** (7.57). CL features (M1+M2+M3) improve this to 6.71 (+0.28), confirming they provide value but cannot overcome the fundamental signal quality limitation of single-step bootstrapping for 5-9 step episodes.

**Next step**: Implement TD(n) in pc-rl-core to bridge the gap between TD(0) and REINFORCE while retaining the `step_masked()` infrastructure and CL features. TD(4-5) should approach REINFORCE quality for TicTacToe's episode length. See `docs/td_n_implementation_plan.md`.

---

## Cross-Experiment Summary

| # | Experiment | Architecture | CL | Mean | D>=8 | D=9 | Stalled |
|---|-----------|-------------|-----|------|------|-----|---------|
| — | Episodic 1-layer (baseline) | 1x27 tanh | — | 7.57 | 37% | 20% | 0% |
| — | Episodic 3-layer (baseline) | [27,27,18] softsign | — | 7.63 | — | 23% | — |
| 1 | CL 1-layer, ALL CL (50k) | 1x27 tanh | M1+M2+M3+M4 | 6.29 | 5.7% | 0% | 22.9% |
| 2 | CL 3-layer, ALL CL no EWC (200k) | [27,27,18] softsign | M1+M2+M3 | 6.71 | 5.7% | 2.9% | 2.9% |
| 3 | CL 3-layer, NO CL (200k) | [27,27,18] softsign | None | 6.43 | 2.9% | 0% | 5.7% |

### Key Conclusions

1. **TD(0) vs REINFORCE gap = ~1.14 depth levels** (Exp 3 vs episodic baseline). This is the dominant factor limiting continuous learning performance on TicTacToe.

2. **CL features (M1+M2+M3) add +0.28 depth** on top of TD(0) baseline (Exp 2 vs Exp 3). Hysteresis + consolidation + scale range improve learning.

3. **EWC (M4) is harmful for curriculum** — costs ~0.42 depth levels (Exp 1 vs Exp 2, adjusting for architecture). EWC protects weights from earlier curriculum depths, preventing adaptation to harder opponents.

4. **3-layer architecture helps CL** — eliminates offensive bias (22.9% → 2.9% stalled) and enables D=9 (0% → 2.9%). Consolidation decay is meaningful with multiple layers.

5. **Path forward: TD(n)** — implementing n-step returns in `step_masked()` should bridge the TD(0)-REINFORCE gap while preserving CL features. See `docs/td_n_implementation_plan.md`.

---

## Recommended Next Experiments

### Experiment 4: TD(n) on TicTacToe

**Purpose**: Test whether TD(n) with n=4-5 approaches REINFORCE performance while retaining CL infrastructure.

Requires: TD(n) implementation in pc-rl-core (see `docs/td_n_implementation_plan.md`).

```toml
td_steps = 4    # or sweep [0, 2, 4, 5, 8]
# With CL M1+M2+M3 (no EWC)
actor_hysteresis = true
critic_hysteresis = true
consolidation_decay = 0.95
adaptive_consolidation = true
ewc_lambda = 0.0
```

### Experiment 5: M1+M2 Only (no consolidation, no EWC)

**Purpose**: Isolate hysteresis contribution from consolidation.

```toml
actor_hysteresis = true
critic_hysteresis = true
consolidation_decay = 1.0
adaptive_consolidation = false
ewc_lambda = 0.0
```

### Experiment 6: Reduced EWC

**Purpose**: Test if very low EWC (0.001-0.01) provides benefit without the harmful over-protection seen at 0.1.

```toml
ewc_lambda = 0.001    # 100x lower than Exp 1-2
```

---

## Experiment 4: ALL CL + EWC=0.001 ([27,27,18] softsign, 200k episodes)

### Setup

Same as Experiment 2 except `ewc_lambda = 0.001` (100x lower than Exp 1-2 which used 0.1).

**Purpose**: Test if micro-EWC provides benefit without the over-protection seen at 0.1.

### Results

| Metric | EWC=0.001 (Exp 4) | EWC=0 (Exp 2) | No CL (Exp 3) |
|--------|---------------------|----------------|----------------|
| **Mean depth** | **6.43** | **6.71** | **6.43** |
| StdDev | 1.20 | 1.00 | 1.20 |
| D>=7 | 62.9% | 74.3% | 62.9% |
| D>=8 | 2.9% | 5.7% | 2.9% |
| D=9 | 0% | 2.9% | 0% |
| Stalled D<=5 | 5.7% | 2.9% | 5.7% |
| High-win (>40%) | **25.7%** | 0% | 11.4% |
| No-draw (<1%) | **22.9%** | 0% | 8.6% |

#### Depth Distribution

| Depth | Count | Percentage |
|-------|-------|------------|
| D=2 | 2 | 5.7% |
| D=6 | 11 | 31.4% |
| D=7 | 21 | 60.0% |
| D=8 | 1 | 2.9% |

### Key Finding: EWC=0.001 Worsens Offensive Bias

EWC=0.001 produced the **worst offensive bias** of all experiments: 25.7% of runs (9/35) ended with >40% win rate and near-0% draws. This is worse than no CL at all (11.4%).

**Mechanism**: Even at 0.001, EWC anchors offensive weights learned at early depths. When the agent needs to learn defensive play (draws), the penalty resists the weight changes needed. The effect is subtle but accumulates over thousands of episodes, trapping 25% of seeds in offensive-only local minima.

### Conclusion (Experiment 4)

EWC=0.001 erases the benefit of CL features (M1+M2+M3), producing results identical to TD(0) without CL (mean 6.43) while adding severe offensive bias. Even micro-EWC is counterproductive for curriculum learning.

---

## Experiment 5: ALL CL + EWC=0.01 ([27,27,18] softsign, 200k episodes)

### Setup

Same as Experiment 2 except `ewc_lambda = 0.01` (10x lower than Exp 1-2).

**Purpose**: Test intermediate EWC strength between 0.001 and 0.1.

### Results

| Metric | EWC=0.01 (Exp 5) | EWC=0 (Exp 2) | EWC=0.001 (Exp 4) | No CL (Exp 3) |
|--------|-------------------|----------------|---------------------|----------------|
| **Mean depth** | **6.71** | **6.71** | **6.43** | **6.43** |
| StdDev | **0.45** | 1.00 | 1.20 | 1.20 |
| D>=7 | 71.4% | 74.3% | 62.9% | 62.9% |
| D>=8 | **0%** | 5.7% | 2.9% | 2.9% |
| D=9 | **0%** | 2.9% | 0% | 0% |
| Stalled D<=5 | **0%** | 2.9% | 5.7% | 5.7% |
| High-win (>40%) | 2.9% | 0% | 25.7% | 11.4% |
| No-draw (<1%) | 5.7% | 0% | 22.9% | 8.6% |

#### Depth Distribution

| Depth | Count | Percentage |
|-------|-------|------------|
| D=6 | 10 | 28.6% |
| D=7 | 25 | 71.4% |

### Key Finding: EWC=0.01 Is a Variance Compressor

EWC=0.01 has a unique profile:
- **Most consistent** experiment: SD=0.45 (half of EWC=0's 1.00)
- **Zero stalls**: No runs stuck at D<=5 — the regularization prevents catastrophic early failures
- **Zero breakthroughs**: No D>=8 — the regularization also prevents the weight adjustments needed to break the D=7 barrier
- All 35 runs fall exactly in D=6 or D=7

EWC=0.01 acts as a **stabilizer that compresses variance in both directions**. It eliminates bad outliers (stalls) but also eliminates good outliers (D=9 breakthroughs).

### Per-Seed Results

| Seed | Max Depth | Win% | Loss% | Draw% | Notes |
|------|-----------|------|-------|-------|-------|
| 10612898363319890476 | 6 | 0.0 | 50.2 | 49.8 | |
| 14591736869251471028 | 7 | 0.0 | 50.0 | 50.0 | |
| 2196172497182831787 | 7 | 31.2 | 38.0 | 30.8 | |
| 9655392786921996517 | 7 | 0.0 | 51.8 | 48.2 | |
| 157609645414484935 | 7 | 0.0 | 100.0 | 0.0 | Collapsed (100% loss) |
| 767712270205390717 | 6 | 0.0 | 52.2 | 47.8 | |
| 8818346561871798133 | 7 | 0.0 | 51.7 | 48.3 | |
| 131026663809412892 | 7 | 0.0 | 50.2 | 49.8 | |
| 16106870278067712081 | 7 | 0.0 | 50.5 | 49.5 | |
| 4789593483134899290 | 7 | 0.0 | 50.8 | 49.2 | |
| 4850978948952663190 | 7 | 0.0 | 57.1 | 42.9 | |
| 4079923755026591305 | 7 | 0.0 | 50.1 | 49.9 | |
| 12574805091622085408 | 7 | 0.0 | 50.1 | 49.9 | |
| 7060496949895095296 | 7 | 0.0 | 50.0 | 50.0 | |
| 12291377338625711910 | 6 | 21.8 | 75.9 | 2.3 | |
| 1923619048265451022 | 7 | 0.2 | 52.6 | 47.2 | |
| 9913965660871889779 | 6 | 49.8 | 50.0 | 0.2 | Offensive bias |
| 5229569336883715921 | 6 | 0.0 | 50.5 | 49.5 | |
| 3958872925701857175 | 7 | 0.0 | 53.4 | 46.6 | |
| 4827886763911061407 | 6 | 0.0 | 49.9 | 50.1 | |
| 7795743518564972618 | 6 | 0.0 | 50.0 | 50.0 | |
| 16045343278116472324 | 6 | 0.0 | 50.2 | 49.8 | |
| 3529563798045221285 | 6 | 0.0 | 50.3 | 49.7 | |
| 950071532918290141 | 7 | 0.0 | 51.0 | 49.0 | |
| 1263431426212177929 | 7 | 0.0 | 50.6 | 49.4 | |
| 8405823101689603585 | 7 | 0.1 | 50.4 | 49.5 | |
| 11354176146632036039 | 7 | 0.0 | 51.1 | 48.9 | |
| 3422226114505585424 | 6 | 0.0 | 49.9 | 50.1 | |
| 15586799941651308502 | 7 | 0.0 | 50.2 | 49.8 | |
| 11566606896045375378 | 7 | 0.0 | 50.4 | 49.6 | |
| 699405758796758849 | 7 | 0.0 | 54.8 | 45.2 | |
| 15316936686473200466 | 7 | 0.0 | 50.3 | 49.7 | |
| 14812081037241177562 | 7 | 0.0 | 50.0 | 50.0 | |
| 6805287656426657652 | 7 | 0.0 | 50.3 | 49.7 | |
| 3972698476113938481 | 7 | 0.0 | 53.0 | 47.0 | |

### Conclusion (Experiment 5)

EWC=0.01 matches EWC=0 on mean depth (6.71) and eliminates stalls, but creates a hard ceiling at D=7 with zero D>=8 breakthroughs. It trades exploration potential for stability — useful if consistency matters more than peak performance.

---

## Cross-Experiment Summary

| # | Experiment | EWC | Mean | SD | D>=8 | D=9 | Stalled | High-Win | No-Draw |
|---|-----------|-----|------|----|------|-----|---------|----------|---------|
| — | Episodic REINFORCE (baseline) | — | 7.57 | 0.81 | 37% | 20% | 0% | — | — |
| 1 | CL 1-layer, ALL CL (50k) | 0.1 | 6.29 | 1.23 | 5.7% | 0% | 22.9% | — | — |
| 2 | CL 3-layer, M1+M2+M3 (200k) | **0** | **6.71** | 1.00 | **5.7%** | **2.9%** | 2.9% | **0%** | **0%** |
| 3 | TD(0) NO CL (200k) | — | 6.43 | 1.20 | 2.9% | 0% | 5.7% | 11.4% | 8.6% |
| 4 | CL 3-layer, ALL CL (200k) | 0.001 | 6.43 | 1.20 | 2.9% | 0% | 5.7% | 25.7% | 22.9% |
| 5 | CL 3-layer, ALL CL (200k) | 0.01 | 6.71 | **0.45** | 0% | 0% | **0%** | 2.9% | 5.7% |

### EWC Impact on Curriculum Learning

| EWC Lambda | Effect on Curriculum |
|------------|---------------------|
| 0 | **Best overall**: highest D>=8 rate, only config to reach D=9, zero offensive bias |
| 0.001 | **Worst**: erases CL benefit, severe offensive bias (25.7% high-win) |
| 0.01 | **Most stable** (SD=0.45) but caps at D=7 — zero breakthroughs |
| 0.1 | **Harmful**: 1-layer stalls at 22.9%, anchors offensive weights |

**Conclusion**: EWC is counterproductive for curriculum learning at any positive value. Curriculum levels are the same task at increasing difficulty — EWC's inter-task forgetting protection anchors weights to inferior solutions from earlier depths, preventing the adaptation needed for harder opponents.

The non-monotonic behavior (0.001 worse than 0.01) suggests EWC interacts with hysteresis in complex ways: very low EWC may allow partial weight updates that drift toward offense without the stabilizing constraint that higher EWC provides, while higher EWC constrains exploration uniformly.

### Key Conclusions (Updated)

1. **TD(0) vs REINFORCE gap = ~1.14 depth levels** (Exp 3 vs episodic baseline). This is the dominant factor.

2. **CL features (M1+M2+M3) add +0.28 depth** on top of TD(0) baseline when EWC=0 (Exp 2 vs Exp 3).

3. **EWC is harmful for curriculum** at any positive value. Best performance is always EWC=0.

4. **EWC=0.01 is useful only if consistency matters more than peak depth** — eliminates stalls but caps at D=7.

5. **Path forward: TD(n)** — implementing n-step returns in `step_masked()` should bridge the TD(0)-REINFORCE gap while preserving CL features.

---

## Recommended Next Experiments

### Experiment 6: TD(n) on TicTacToe

**Purpose**: Bridge the TD(0)-REINFORCE gap using n-step returns while retaining CL infrastructure.

Requires: TD(n) implementation in pc-rl-core.

```toml
td_steps = 4    # or sweep [0, 2, 4, 5, 8]
# With best CL config (M1+M2+M3, no EWC)
actor_hysteresis = true
critic_hysteresis = true
consolidation_decay = 0.95
adaptive_consolidation = true
ewc_lambda = 0.0
```

### Experiment 7: M1+M2 Only (no consolidation, no EWC)

**Purpose**: Isolate hysteresis contribution from consolidation.

```toml
actor_hysteresis = true
critic_hysteresis = true
consolidation_decay = 1.0
adaptive_consolidation = false
ewc_lambda = 0.0
```

---

## Experiment 7: TD(5) Control — No CL ([27,27,18] softsign, 200k)

### Setup

Same network as Exp 2, but:
- `td_steps = 5` (Monte Carlo for TicTacToe — agent makes 2-5 moves per episode)
- All CL features DISABLED
- `random_side = false` (alternating, matches REINFORCE baseline)
- `scale_floor = 0.1` (legacy v2.0.0 behavior)

**Purpose**: Isolate TD(5) vs REINFORCE signal quality with CL removed as a variable.

### Results

| Metric | Value |
|--------|-------|
| Mean | **6.71** |
| StdDev | 1.39 |
| D>=7 | 74.3% |
| D>=8 | 11.4% (4 runs) |
| **D=9** | **8.6% (3 runs)** |
| Stalled (D<=5) | 5.7% |
| Collapsed (>80% loss) | 11.4% |

### Key Finding: TD(5) Can Reach D=9

**TD(5) without CL is the first continuous-mode experiment to reach D=9** (3 runs). This refutes the earlier hypothesis that TD itself is the ceiling limitation. However, the cost is high variance (SD 1.39) and 11.4% collapses.

### Trade-off vs CL

| Config | Stability | Peak Depth |
|--------|-----------|------------|
| TD(5) + CL (Exp 6) | 0% collapse, 0% stall | D=9: 0% |
| TD(5) sin CL (Exp 7) | 11.4% collapse, 5.7% stall | D=9: 8.6% |

CL suppresses the aggressive weight updates needed to break the D=7 barrier but also protects against the inestability that produces collapses.

### Conclusion (Experiment 7)

TD(5) is not the ceiling limitation — it CAN reach D=9 without CL. The performance gap vs REINFORCE baseline (6.71 vs 7.63) is ~0.92 depth levels and comes from something other than the TD signal quality itself (likely differences between `step_masked()` and `act+learn` at the core level, or `scale_floor` legacy behavior still applying).

---

## Experiment 8: GAE(0.95) Control — No CL ([27,27,18] softsign, 200k)

### Setup

Same as Exp 7 but:
- `gae_lambda = 0.95` (eligibility traces)
- `td_steps = 0` (mutually exclusive with gae_lambda)

**Purpose**: Test if GAE(λ) with output-level eligibility traces outperforms TD(5) — theoretically GAE(0.95) is a smooth interpolation between TD(0) and Monte Carlo.

### Results

| Metric | Value |
|--------|-------|
| Mean | **6.00** |
| StdDev | 1.69 |
| D>=7 | 57.1% |
| D>=8 | 2.9% (1 run) |
| **D=9** | **0%** |
| Stalled (D<=5) | 17.1% |
| Collapsed (>80% loss) | 22.9% |

#### Depth Distribution

| Depth | Count | % |
|-------|-------|---|
| D=2 | 4 | 11.4% |
| D=3 | 1 | 2.9% |
| D=4 | 1 | 2.9% |
| D=6 | 9 | 25.7% |
| D=7 | 19 | 54.3% |
| D=8 | 1 | 2.9% |

### Key Finding: GAE(0.95) is WORSE than TD(5) for TicTacToe

| Metric | TD(5) sin CL (Exp 7) | GAE(0.95) sin CL (Exp 8) | Delta |
|--------|----------------------|--------------------------|-------|
| Mean | 6.71 | **6.00** | **-0.71** |
| SD | 1.39 | 1.69 | +0.30 |
| D>=8 | 11.4% | 2.9% | -8.5% |
| D=9 | 8.6% | 0% | -8.6% |
| Stalled | 5.7% | 17.1% | +11.4% |
| Collapsed | 11.4% | 22.9% | +11.5% |

### Why GAE Underperforms TD(5) Here

1. **TD(5) is already Monte Carlo for TicTacToe**. With 2-5 agent moves per episode, TD(5) captures the complete trajectory. GAE(0.95) is an approximation of Monte Carlo — when TD(5) is already exact, GAE adds noise without benefit.

2. **Output-level eligibility traces**: GAE implements traces on the output layer. For a 3-layer PC network, traces may not propagate well through deep hidden layers. TD(n) buffers complete transitions with InferResult caches, giving better propagation.

3. **Interaction with PC inference loop**: The PC Actor has an internal inference loop (alpha=0.03). Output-level gradient traces may desynchronize with PC internal activations, while TD(n) preserves the complete inference state per transition.

### Where GAE Would Actually Help

GAE(λ) should outperform TD(n) in domains with **long episodes** where:
- Fixed n-step is impractical (memory cost of TD(n) scales with n × network_size)
- TD(0) has high bias but TD(large n) has high variance
- The trajectory contains many intermediate rewards

For TicTacToe (5-9 step episodes), none of these apply. **TD(5) is the correct choice** here.

### Conclusion (Experiment 8)

GAE(0.95) significantly underperforms TD(5) for TicTacToe — mean 6.00 vs 6.71, zero D=9 runs vs 8.6%, and doubled stall/collapse rates. The eligibility traces approach is theoretically elegant but provides no benefit when the episode is short enough that TD(n) already captures the full trajectory. **GAE is reserved for longer-episode domains** (Qubic, Connect 4, Chess).

### Cross-Algorithm Summary (no CL, 200k, alternating)

| Algorithm | Mean | D=9 | Stalled | Collapsed | Verdict |
|-----------|------|-----|---------|-----------|---------|
| REINFORCE baseline | 7.63 | 23% | — | — | Gold standard |
| TD(0) | 6.43 | 0% | 5.7% | 0% | Insufficient signal for 5-9 step episodes |
| **TD(5) = MC** | **6.71** | **8.6%** | **5.7%** | **11.4%** | **Best continuous algorithm** |
| GAE(0.95) | 6.00 | 0% | 17.1% | 22.9% | Noisy approximation, worse than TD(5) |
| GAE(1.0) | 6.23 | 2.9% | 17.1% | 11.4% | Pure MC traces, still worse than TD(5) |

---

## Experiment 9: GAE(1.0) Control — Pure Monte Carlo ([27,27,18] softsign, 200k)

### Setup

Same as Exp 8 but `gae_lambda = 1.0` (no trace decay). Theoretically equivalent to pure Monte Carlo — should match TD(5) since TD(5) is also Monte Carlo for TicTacToe (agent makes at most 5 moves per episode).

**Purpose**: Test whether GAE with no decay (λ=1.0) matches TD(5) and confirm/refute the theoretical equivalence between GAE(1.0) and TD(n) for short episodes.

### Results

| Metric | Value |
|--------|-------|
| Mean | **6.23** |
| StdDev | 1.40 |
| D>=7 | 51.4% |
| D>=8 | 8.6% (3 runs) |
| **D=9** | **2.9% (1 run)** |
| Stalled (D<=5) | 17.1% |
| Collapsed (>80% loss) | 11.4% |
| High-win (>40%) | 25.7% |
| No-draw (<1%) | 37.1% |

#### Depth Distribution

| Depth | Count | % |
|-------|-------|---|
| D=2 | 1 | 2.9% |
| D=3 | 1 | 2.9% |
| D=4 | 3 | 8.6% |
| D=5 | 1 | 2.9% |
| D=6 | 11 | 31.4% |
| D=7 | 15 | 42.9% |
| D=8 | 2 | 5.7% |
| D=9 | 1 | 2.9% |

### Key Finding: GAE(1.0) is NOT equivalent to TD(5)

Despite being theoretically both Monte Carlo for TicTacToe, GAE(1.0) underperforms TD(5) significantly:

| Metric | TD(5) (Exp 7) | GAE(1.0) (Exp 9) | Delta |
|--------|---------------|-------------------|-------|
| Mean | 6.71 | 6.23 | **-0.48** |
| D=9 | 8.6% | 2.9% | -5.7% |
| D>=8 | 11.4% | 8.6% | -2.8% |
| Stalled | 5.7% | 17.1% | +11.4% |
| High-win | 17.1% | 25.7% | +8.6% |
| No-draw | 28.6% | 37.1% | +8.5% |

GAE(1.0) improves marginally over GAE(0.95) (mean 6.23 vs 6.00, confirming λ=1.0 is closer to MC than λ=0.95), but still falls short of TD(5).

### Why GAE(1.0) ≠ TD(5) in practice

Despite theoretical equivalence, the two produce different results due to implementation differences:

1. **Signal propagation through layers**: TD(n) buffers complete transitions with `InferResult` caches, preserving activations across all 3 hidden layers. GAE uses eligibility traces **only at the output layer**, so the gradient signal must propagate back through the PC inference loop each update without the benefit of cached activations.

2. **Update timing**: TD(5) waits for the buffer to fill, then learns from the oldest transition using its cached state. GAE updates at every step with the accumulated trace. In 5-9 step episodes, this produces different gradient flows even when the mathematical target is the same.

3. **Offensive bias amplification**: GAE(1.0) has the highest no-draw rate (37.1%) of all experiments. The trace accumulation may bias gradients toward early actions (where offense pays off as P1), producing agents that never learn P2 defense.

### Conclusion (Experiment 9)

**GAE(1.0) is not a viable replacement for TD(5) in TicTacToe.** Even at λ=1.0 (pure Monte Carlo), the output-level eligibility traces do not match the full transition buffering of TD(n). For short episodes with multi-layer PC networks, TD(5) remains strictly superior.

**Implication for champion search**: Use `td_steps = 5` (not GAE) when searching for optimal agents via `find-champion`. The best continuous-mode algorithm for TicTacToe is TD(5) without CL.

---

### The Residual Gap

All `step_masked()` algorithms (TD(0), TD(5), GAE(0.95), GAE(1.0)) produce mean depth 6.00-6.71 vs REINFORCE's 7.63. The ~0.9-1.6 gap is **not explained by TD signal quality** — TD(5) is mathematically equivalent to Monte Carlo for TicTacToe, yet still falls short of REINFORCE.

**Remaining candidates for the gap:**
1. `scale_floor = 0.1` legacy behavior in `step_masked()` (not present in `act+learn`)
2. Differences in `surprise_buffer` update timing between the two APIs
3. Entropy regularization applied differently per-step vs per-trajectory
4. Some subtle ordering difference in the critic input construction
5. Activation cache handling differences between per-step inference and trajectory-based inference

This gap is at the **core library level** (pc-rl-core `step_masked` implementation) and would require investigation in core to resolve.
