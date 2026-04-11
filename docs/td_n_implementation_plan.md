# TD(n) Generic Implementation Plan for pc-rl-core

**Date**: 2026-04-11
**Target**: `pc-rl-core` crate, `continuos_learning` branch
**Author**: Julian Bolivar

## Motivation

TD(0) (`step_masked()`) learns from single-step transitions: `target = r + γV(s')`. For short-episode games like TicTacToe (5-9 steps), this produces noisy gradient signal. REINFORCE (trajectory-based `learn()`) uses the full episode return, giving better signal but higher variance for long episodes.

TD(n) is the middle ground: accumulate n steps of real rewards before bootstrapping with V(s'):

```
TD(0): target = r_t + γ × V(s_{t+1})
TD(n): target = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^(n-1)r_{t+n-1} + γⁿ × V(s_{t+n})
TD(∞): target = r_t + γr_{t+1} + ... + γ^T r_T    (= REINFORCE, no bootstrap)
```

Experimental evidence from PC-TicTacToe:
- Episodic REINFORCE: mean depth 7.57 (N=35)
- TD(0) with CL: mean depth 6.71
- TD(0) without CL: mean depth 6.43
- Gap is ~1.14 depth levels, attributable to TD(0) signal quality

TD(n) with n=4-5 should approach REINFORCE quality while retaining `step_masked()` infrastructure and CL features (hysteresis, consolidation, EWC).

## Design Principles

1. **Zero overhead when disabled**: `td_steps = 0` (default) must produce identical behavior to current TD(0). No buffer allocation, no extra computation.
2. **Backward compatible**: All existing configs, save files, and API contracts remain valid.
3. **Single config parameter**: `td_steps: usize` controls n. The agent handles everything internally.
4. **CL features unchanged**: Hysteresis, consolidation, EWC apply to the learning step exactly as before — only the target computation changes.
5. **Terminal flush**: When an episode ends, all buffered transitions must be flushed with progressively shorter n-step returns.

## Architecture

### Current TD(0) Flow (step_inner)

```
step_masked(state, valid, reward, terminal=false):
  1. If state_prev exists:
     learn_continuous(state_prev → state, reward)  // TD(0) update
  2. Infer on state, select action
  3. Store (state, action, infer, valid) as _prev
  4. Return action

step_masked(state, valid, reward, terminal=true):
  1. learn_continuous(state_prev → state, reward, terminal)  // V(s')=0
  2. Clear _prev state
  3. Return action (discarded)
```

### Proposed TD(n) Flow

```
step_masked(state, valid, reward, terminal=false):
  1. Push (state_prev, action_prev, infer_prev, valid_prev, reward) into buffer
  2. If buffer.len() == n:
     Pop oldest transition
     Compute n-step return: G = Σ(γ^i × r_i) + γⁿ × V(current_state)
     learn_continuous(oldest_state → current_state, G_as_target)
  3. Infer on state, select action
  4. Store (state, action, infer, valid) as _prev
  5. Return action

step_masked(state, valid, reward, terminal=true):
  1. Push final transition into buffer
  2. Flush buffer from oldest to newest:
     For each remaining transition at position k (0..buffer.len()):
       remaining_steps = buffer.len() - k
       G = Σ(γ^i × r_{k+i}, i=0..remaining_steps) + 0  // terminal, no bootstrap
       learn_continuous(state_k, G_as_target)
  3. Clear buffer and _prev state
  4. Return action (discarded)
```

### Key Insight: learn_continuous Reuse

`learn_continuous()` currently computes:
```rust
let target = reward + if terminal { 0.0 } else { self.config.gamma * v_next };
let td_error = target - v_s;
```

For TD(n), we **don't change learn_continuous**. Instead, we pre-compute the n-step return and pass it as a modified (reward, terminal) pair:

```rust
// TD(n) wraps learn_continuous by computing:
let n_step_return = sum(gamma^i * rewards[i]) + gamma^n * V(s_{t+n});
// Then calls learn_continuous with:
//   reward = n_step_return - gamma^n * V(s_{t+n})  // net reward
//   next_state = s_{t+n}
//   terminal = (is_episode_terminal && this_is_last_flush)
```

Actually, the cleaner approach is to call learn_continuous with the **bootstrapped state** being `s_{t+n}` (the state n steps ahead), and the **accumulated reward** being the discounted sum of intermediate rewards. This way learn_continuous sees:

```rust
learn_continuous(
    input: s_t,           // state where action was taken
    infer: infer_t,       // inference at s_t
    action: a_t,          // action taken
    valid_actions: v_t,   // valid actions at s_t
    reward: G_n,          // n-step discounted reward sum
    next_input: s_{t+n},  // state n steps later
    next_infer: infer_{t+n},
    terminal: is_truly_terminal,  // true only if episode ended within n steps
)
```

Where `G_n = r_t + γr_{t+1} + ... + γ^(n-1)r_{t+n-1}`.

Then learn_continuous computes:
```
target = G_n + γⁿ × V(s_{t+n})    // if not terminal
target = G_n                        // if terminal (V=0)
td_error = target - V(s_t)
```

This is mathematically equivalent to the n-step TD target and requires **zero changes to learn_continuous itself**.

## Data Structures

### Transition Buffer Entry

```rust
/// A single buffered transition for TD(n) computation.
struct TdTransition<L: LinAlg> {
    /// State observation.
    state: L::Vector,
    /// Inference result at this state.
    infer: InferResult<L>,
    /// Action taken.
    action: usize,
    /// Valid action mask at this state.
    valid_actions: Vec<usize>,
    /// Reward received after taking this action.
    reward: f64,
}
```

### Buffer in PcActorCritic

```rust
// New field in PcActorCritic struct
td_buffer: VecDeque<TdTransition<L>>,
```

### Config Addition

```rust
// In PcActorCriticConfig
/// Number of steps for TD(n) return computation.
/// 0 = standard TD(0) (default, zero overhead).
/// n > 0 = accumulate n steps before learning.
#[serde(default)]
pub td_steps: usize,
```

## Implementation Steps

### Step 1: Add config field

**File**: `src/pc_actor_critic/config.rs`

Add `td_steps: usize` with `#[serde(default)]` (defaults to 0). No validation needed beyond non-negative (which usize guarantees).

### Step 2: Add TdTransition type and buffer

**File**: `src/pc_actor_critic/mod.rs`

Add the `TdTransition<L>` struct (private, not serialized). Add `td_buffer: VecDeque<TdTransition<L>>` to `PcActorCritic` struct. Initialize as empty `VecDeque::new()` in constructors.

### Step 3: Implement n-step return computation

**File**: `src/pc_actor_critic/mod.rs`

Add a private method:

```rust
/// Computes the n-step discounted return from a slice of transitions.
///
/// Returns the accumulated reward: Σ(γ^i × r_i) for i in 0..transitions.len()
fn compute_n_step_reward(&self, transitions: &[TdTransition<L>]) -> f64 {
    let mut g = 0.0;
    let mut gamma_power = 1.0;
    for t in transitions {
        g += gamma_power * t.reward;
        gamma_power *= self.config.gamma;
    }
    g
}
```

### Step 4: Implement buffer flush for terminal episodes

**File**: `src/pc_actor_critic/mod.rs`

Add a private method:

```rust
/// Flushes the TD(n) buffer at episode end.
///
/// Each buffered transition gets an (n-k)-step return where k is its
/// position. The last transition gets a 1-step return (like TD(0)).
/// All returns use terminal=true (V(s')=0) since the episode ended.
fn flush_td_buffer(
    &mut self,
    terminal_state: &[f64],
    terminal_infer: &InferResult<L>,
) {
    let buffer: Vec<TdTransition<L>> = self.td_buffer.drain(..).collect();
    let len = buffer.len();
    for (k, transition) in buffer.iter().enumerate() {
        let remaining = &buffer[k..];
        let n_step_reward = self.compute_n_step_reward(remaining);
        // All flushes are terminal — no bootstrap
        self.learn_continuous(
            /* input */        transition.state.as_slice(),
            /* infer */        &transition.infer,
            /* action */       transition.action,
            /* valid_actions */ &transition.valid_actions,
            /* reward */       n_step_reward,
            /* next_input */   terminal_state,
            /* next_infer */   terminal_infer,
            /* terminal */     true,
        );
    }
}
```

Note: `transition.state.as_slice()` assumes `L::Vector` implements `AsRef<[f64]>` or similar. Adjust based on actual `LinAlg` trait API.

### Step 5: Modify step_inner() for TD(n)

**File**: `src/pc_actor_critic/mod.rs`

The key change is in `step_inner()`. Current flow:

```rust
// Current: learn from immediate previous
if let (Some(state_prev), Some(action_prev), Some(infer_prev)) = (...) {
    let learn_mask = ...;
    self.learn_continuous(state_prev, infer_prev, action_prev, learn_mask,
                          reward, state, &infer, terminal);
    // ... process hysteresis
}
```

**TD(n) flow** (when `td_steps > 0`):

```rust
if let (Some(sp), Some(ap), Some(ip)) = (state_prev, action_prev, infer_prev) {
    let vp = valid_actions_prev.unwrap_or_else(|| (0..self.config.actor.output_size).collect());

    if self.config.td_steps == 0 {
        // === TD(0): existing behavior, unchanged ===
        self.learn_continuous(&sp, &ip, ap, &vp, reward, state, &infer, terminal);
    } else if terminal {
        // === TD(n) terminal: push last transition, flush entire buffer ===
        self.td_buffer.push_back(TdTransition {
            state: sp, infer: ip, action: ap,
            valid_actions: vp, reward,
        });
        self.flush_td_buffer(state, &infer);
    } else {
        // === TD(n) non-terminal: buffer transition ===
        self.td_buffer.push_back(TdTransition {
            state: sp, infer: ip, action: ap,
            valid_actions: vp, reward,
        });

        // If buffer is full (n transitions), learn from oldest
        if self.td_buffer.len() >= self.config.td_steps {
            let n_step_reward = self.compute_n_step_reward(
                &self.td_buffer.iter().collect::<Vec<_>>()  // all n transitions
            );
            let oldest = self.td_buffer.pop_front().unwrap();
            // Bootstrap with V(current_state), gamma^n already in learn_continuous
            // Pass n_step_reward as "reward" and current state as "next"
            // learn_continuous will add gamma * V(s_current) internally,
            // but we need gamma^n, not gamma^1.
            // SEE: "Gamma correction" section below.
        }
    }
}
```

### Step 6: Gamma Correction for learn_continuous

**Problem**: `learn_continuous` computes `target = reward + gamma * V(next)`. But for TD(n), we need `target = G_n + gamma^n * V(s_{t+n})`. The gamma exponent is wrong.

**Solution A — Modify learn_continuous to accept gamma_power**:

Add an optional parameter or a new method:

```rust
/// TD(n)-aware learning with explicit discount power.
///
/// Like learn_continuous, but the bootstrap discount is gamma^n
/// instead of gamma^1.
fn learn_continuous_n(
    &mut self,
    /* same params as learn_continuous */
    gamma_power: f64,  // gamma^n for bootstrapping
) -> f64 {
    // ... same as learn_continuous but:
    let target = reward + if terminal { 0.0 } else { gamma_power * v_next };
    // ... rest identical
}
```

**Solution B — Pre-adjust the reward**:

Compute the n-step return and adjust so learn_continuous's gamma^1 gives the right answer:

```rust
// We want: target = G_n + γⁿ × V(s_{t+n})
// learn_continuous computes: target = reward_arg + γ × V(next)
// So set: reward_arg = G_n + (γⁿ - γ) × V(s_{t+n})
// This requires computing V(s_{t+n}) before the call, which we can get from critic.forward()
```

This is fragile and requires forward-passing the critic twice.

**Recommended: Solution A** — cleaner, explicit, no double computation. Add a `gamma_power` parameter to an internal `learn_continuous_inner()` method, and have `learn_continuous` call it with `gamma_power = gamma`.

### Step 7: Update reset_step() to clear buffer

**File**: `src/pc_actor_critic/mod.rs`

```rust
pub fn reset_step(&mut self) {
    self.state_prev = None;
    self.action_prev = None;
    self.infer_prev = None;
    self.valid_actions_prev = None;
    self.td_buffer.clear();  // NEW
}
```

### Step 8: Serialization (optional)

The TD buffer is **transient** — it holds mid-episode state that shouldn't persist across save/load. `reset_step()` clears it, and loading a model starts a fresh episode anyway.

- `td_steps` config field: automatically serialized with PcActorCriticConfig (serde default)
- `td_buffer`: NOT serialized (transient, like state_prev)
- `ClState`: no changes needed

### Step 9: Hysteresis interaction

Currently, hysteresis processes `last_td_error` after each `learn_continuous` call. With TD(n):

- **Buffer-full learning** (1 learn per n steps): Hysteresis updates once per n steps. The TD error is an n-step error (more accurate, less noisy). This is actually **better** for hysteresis — less noise in the EWMA signals.
- **Terminal flush** (multiple learns): Hysteresis updates for each flushed transition. The last few transitions have shorter returns (n-1, n-2, ... 1 step). This is fine — it's equivalent to rapid learning at episode end.

**No changes needed** to hysteresis logic. The existing `process_hysteresis()` call after learn_continuous works correctly.

## Configuration

### PcActorCriticConfig

```rust
/// Number of steps for TD(n) return computation.
/// 0 = standard TD(0) (default, zero overhead).
/// n > 0 = accumulate n real reward steps before bootstrapping with V(s_{t+n}).
/// Higher n reduces bootstrap bias but increases variance.
/// For reference: TD(∞) = Monte Carlo = REINFORCE (no bootstrap).
#[serde(default)]
pub td_steps: usize,
```

### TOML Usage

```toml
[agent]
td_steps = 0      # TD(0) — current default, single-step bootstrap
td_steps = 1      # TD(1) — 1 real step + bootstrap (similar to TD(0) but buffered)
td_steps = 4      # TD(4) — good for TicTacToe (5-9 step episodes)
td_steps = 8      # TD(8) — good for Qubic (10-30 step episodes)
td_steps = 100    # TD(100) — effectively REINFORCE for short episodes
```

### Validation

In config validation:
```rust
// No upper bound needed — if td_steps > episode length, it degrades
// gracefully to Monte Carlo (terminal flush handles all steps).
// td_steps = 0 means TD(0) (no buffer, current behavior).
```

## Interaction with CL Features

| CL Feature | TD(n) Interaction | Changes Needed |
|------------|-------------------|----------------|
| M1 Scale range | surprise_scale applied per learn_continuous call — works unchanged | None |
| M2 Hysteresis | Updates per learn_continuous call — fewer updates (1 per n steps vs 1 per step) but higher quality signal | None |
| M3 Consolidation | Per-layer decay applied per learn_continuous — works unchanged | None |
| M4 EWC | Fisher accumulation per learn_continuous — works unchanged | None |

## Edge Cases

### Episode shorter than n steps

If an episode has k < n steps, the buffer never fills. All k transitions are flushed at terminal with progressively shorter returns:

```
Step 0: 1 transition buffered
Step 1: 2 transitions buffered
Step 2 (terminal): flush all 3 with returns:
  - Transition 0: G = r0 + γr1 + γ²r2 (3-step, no bootstrap)
  - Transition 1: G = r1 + γr2 (2-step, no bootstrap)
  - Transition 2: G = r2 (1-step, no bootstrap)
```

This is correct — short episodes get full Monte Carlo returns.

### td_steps = 0

Buffer is never used. `VecDeque::new()` allocates nothing. The `if self.config.td_steps == 0` branch preserves exact current behavior.

### td_steps = 1

Functionally similar to TD(0) but with one step of buffering delay. The first transition is buffered, and learning happens one step later. At terminal, the single buffered transition is flushed. Slightly different timing but mathematically equivalent.

### reset_step() mid-episode

Clears the buffer, discarding unbacked transitions. This is correct — `reset_step()` is called between episodes, so no data is lost.

## Testing Strategy (TDD)

### Unit Tests in pc-rl-core

1. **test_td0_unchanged**: `td_steps=0` produces identical behavior to current code (regression guard)
2. **test_td_n_buffer_fills_at_n**: With `td_steps=3`, verify buffer has 3 entries before first learn
3. **test_td_n_terminal_flush**: With `td_steps=5` and 3-step episode, verify all 3 transitions are learned at terminal
4. **test_td_n_return_computation**: Verify n-step return math: `G = r0 + γr1 + γ²r2`
5. **test_td_n_gamma_power**: Verify bootstrap uses `γⁿ × V(s_{t+n})`, not `γ × V(s_{t+n})`
6. **test_td_n_reset_clears_buffer**: `reset_step()` empties the buffer
7. **test_td_n_short_episode**: Episode shorter than n → all transitions use Monte Carlo return
8. **test_td_n_serialization_config**: `td_steps` persists in save/load
9. **test_td_n_hysteresis_interaction**: Hysteresis still transitions correctly with buffered updates

### Integration Tests in PC-TicTacToe

1. **test_td_n_completes_game**: Agent with `td_steps=4` plays a complete game without panic
2. **test_td_n_trains_above_random**: Agent with `td_steps=4` achieves >25% win rate after 500 episodes (learning signal reaches early moves)

## Files to Modify in pc-rl-core

| File | Change |
|------|--------|
| `src/pc_actor_critic/config.rs` | Add `td_steps: usize` field |
| `src/pc_actor_critic/mod.rs` | Add `TdTransition`, `td_buffer`, buffer logic in `step_inner()`, `flush_td_buffer()`, `compute_n_step_reward()`, `learn_continuous_inner()` with `gamma_power` |
| `src/pc_actor_critic/mod.rs` | Update `reset_step()` to clear buffer |
| `src/pc_actor_critic/mod.rs` | Update constructors (`new`, `from_parts`) to init empty buffer |
| `src/lib.rs` | No changes (no new public types) |
| `src/serializer.rs` | No changes (buffer is transient) |

## Files to Modify in PC-TicTacToe

| File | Change |
|------|--------|
| `src/utils/config.rs` | Add `td_steps: usize` to `AgentSection`, wire in `to_agent_config()` |
| `config.toml` | Add `td_steps = 0` |
| `config_cl_experiment.toml` | Set `td_steps = 4` for experiments |
| `src/ui/cli.rs` | Add `td_steps` to `DEFAULT_CONFIG_TOML` |

## Estimated Complexity

- **pc-rl-core**: ~150-200 lines of new code + ~50 lines of tests
- **PC-TicTacToe**: ~10 lines (config field + TOML)
- **Risk**: Low — additive change, `td_steps=0` preserves exact current behavior
- **Breaking changes**: None — new serde-default field, existing saves load fine

## Experimental Plan

After implementation:

1. **Baseline verification**: Run seed-test with `td_steps=0` — results must match current TD(0) experiments
2. **TD(4) on TicTacToe 3x3**: 35 seeds, 200k episodes, 3-layer network — should approach episodic REINFORCE performance (mean ~7.5)
3. **TD(4) + CL**: Same as above with M1+M2+M3 enabled — test if CL helps TD(4)
4. **Sweep td_steps**: Test [0, 1, 2, 3, 4, 5, 8] on TicTacToe to find optimal n for 5-9 step episodes
5. **Qubic application**: When Qubic environment is ready, test TD(5-10) with CL on longer episodes
