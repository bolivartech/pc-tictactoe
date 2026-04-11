# Actor Hysteresis Signal — Hybrid Surprise + TD Error

**Date**: 2026-04-11
**Author**: Julian Bolivar
**Target**: `pc-rl-core` crate, `TD_n` branch
**Related**: PC-TicTacToe `continuos_learning` branch experiments

---

## Problem Statement

The actor hysteresis state machine uses **PC surprise (RMS prediction error)** as its signal to decide when to freeze (PLASTIC → FROZEN) and when to wake (FROZEN → PLASTIC). Experimental evidence from PC-TicTacToe shows this signal is insufficient — the actor **never freezes** when it should, and **never wakes** when it should.

### Evidence: Actor Never Freezes (Experiments 2-5)

In 5 experiments (N=35 seeds each, 200k-500k episodes), the actor remained 100% PLASTIC across entire runs. The diagnostic data shows why:

```
ep= 10000 d=7 actor=P fast=0.3912 slow=0.3883 threshold=0.2718 loss=50.5%
ep=100000 d=7 actor=P fast=0.4177 slow=0.4169 threshold=0.2918 loss=50.2%
ep=300000 d=7 actor=P fast=0.5041 slow=0.5051 threshold=0.3536 loss=50.3%
ep=500000 d=7 actor=P fast=0.4701 slow=0.4701 threshold=0.3291 loss=50.6%
```

The fast and slow EWMAs track each other closely (~0.40) and never diverge enough to trigger sleep (`fast < slow × 0.7`). The gap to threshold remains ~0.12 throughout.

**Root cause**: The agent alternates between Player One and Player Two every episode. Each side produces different board states, creating natural surprise oscillation that masks convergence. The PC surprise reflects "I'm seeing different states" (side alternation), not "I've stopped improving."

### Evidence: Actor Never Wakes After Manual Freeze

When `actor_sleep_fraction` was reduced to 0.05 (very easy to freeze), the actor froze immediately at ep ~500 and **never woke up** despite 74% loss rate for 500k episodes:

```
ep=   500 d=1 actor=F loss=77.0%
ep=500000 d=1 actor=F loss=74.2%  (FROZEN for 499,500 episodes)
```

**Root cause**: Once frozen, the actor's weights don't change, so its internal predictions become stable. PC surprise stays low and consistent even though the agent is performing terribly. The wake condition (`fast > slow × 1.5`) is never met because surprise doesn't reflect performance — it only reflects internal prediction consistency.

### Evidence: Intermediate Setting Still Fails

With `actor_sleep_fraction=0.10` and `actor_slow_window=500`, the actor froze at ep 374k (too late, already degrading) and never woke:

```
ep=374000 P->F fast=0.4138 slow=0.4455 loss=51.8%  (already degrading)
ep=400000      actor=F                   loss=81.1%  (collapsed, can't wake)
ep=500000      actor=F                   loss=80.5%  (stuck)
```

### Contrast: Critic Hysteresis Works

The critic uses **|TD error|** as its signal and correctly alternates between PLASTIC (28% of time) and FROZEN (72% of time). The TD error directly reflects prediction accuracy — when V(s) is wrong, |TD error| is high, and the critic wakes.

## Analysis: Why PC Surprise Fails as Actor Signal

### What PC Surprise Measures

PC surprise = RMS prediction error from the Predictive Coding inference loop. It measures how well the network's internal generative model predicts its own layer activations. High surprise means the network is seeing patterns its internal model hasn't adapted to.

### Why It Doesn't Reflect Performance

1. **Side alternation creates artificial surprise**: Player One and Player Two see fundamentally different board states. Even a perfectly trained agent will have non-zero surprise because the two sides activate different internal patterns. This baseline surprise oscillation prevents the hysteresis from detecting convergence.

2. **Consistent losing has low surprise**: An agent that always loses the same way develops consistent internal predictions. Its PC loop converges to predict its own (bad) activations accurately. Surprise is low even though performance is terrible.

3. **The signal is about internal consistency, not external performance**: Surprise answers "do I predict my own activations well?" not "am I winning or losing?" An agent can be internally consistent (low surprise) while being externally incompetent (high loss rate).

### Why |TD Error| Works Better

|TD error| = |target - V(s)| = |reward + γV(s') - V(s)|

This directly measures prediction accuracy of **outcomes**, not internal states:
- Agent expects to draw but loses → high TD error → wake
- Agent expects to lose and does lose → low TD error → stay frozen (correctly)
- Curriculum advances, agent faces harder opponent → V(s) estimates become wrong → high TD error → wake

## Proposed Solution: Hybrid Signal

### Core Idea

Use the **maximum** of PC surprise and |TD error| as the actor's hysteresis signal. This way either signal can trigger a wake:

```rust
let actor_signal = surprise_score.max(td_error_abs);
```

### Why Maximum (not average or weighted sum)

- **No calibration needed**: Surprise and |TD error| are on different scales. An average or weighted sum requires choosing α, which varies by domain. The max is scale-independent — whichever signal is "alarming" dominates.
- **OR semantics**: We want to wake if **either** the environment changed (surprise) **or** predictions are wrong (TD error). This is a logical OR, which `max()` implements naturally.
- **Preserves both signals**: Surprise still contributes for genuine environment changes (useful for multi-task scenarios). TD error adds the performance awareness that surprise lacks.
- **Conservative for sleep**: The max of two signals is always >= either signal alone, making it harder to freeze. This is correct — we only want to freeze when **both** signals agree it's safe.

### Signal Behavior Matrix

| Surprise | TD Error | Max Signal | Meaning | Action |
|----------|----------|------------|---------|--------|
| High | High | High | New environment, poor predictions | Stay PLASTIC |
| High | Low | High | New environment, adapting well | Stay PLASTIC |
| Low | High | High | Stable environment, losing unexpectedly | **WAKE** (the fix) |
| Low | Low | Low | Stable, performing as expected | FROZEN |

The critical case is row 3: "low surprise, high TD error" — this is exactly the scenario that currently fails. The max signal would correctly keep the actor awake or wake it up.

### Alternative Approaches Considered

**A. Use |TD error| only for actor (like critic)**
- Simpler but throws away PC surprise entirely
- Loses the ability to detect genuine environment changes that aren't reflected in rewards
- For curriculum learning this might work, but for future multi-task scenarios (Qubic after TicTacToe) the PC surprise would be valuable

**B. Weighted combination: α × surprise + (1-α) × |TD error|**
- Requires calibrating α per domain
- Surprise and |TD error| have different scales — α=0.5 may not balance them
- More parameters to tune

**C. Separate conditions: wake if surprise > threshold OR td_error > threshold**
- Requires two separate thresholds
- More complex hysteresis state machine
- Harder to implement cleanly with the existing EWMA-based dual-tracker

**D. Use win/loss rate as signal**
- Requires passing game metrics into the agent (breaks abstraction)
- Domain-specific, not generalizable
- The agent shouldn't need to know about "games"

**E. External hysteresis in the trainer**
- Breaks the self-contained CL design
- Requires duplicating freeze/wake logic outside pc-rl-core
- Not composable with other environments

### Recommendation: Start with max(), make it configurable

Implement `max(surprise, |TD error|)` as the default, with a config option to select the actor hysteresis signal source:

```rust
/// Signal source for actor hysteresis.
/// - "surprise": PC surprise only (original behavior)
/// - "td_error": |TD error| only (same as critic)
/// - "hybrid": max(surprise, |TD error|) (recommended)
pub actor_hysteresis_signal: ActorHysteresisSignal,
```

This allows experimentation without code changes and preserves backward compatibility.

## Implementation Details

### Where the Change Goes

**File**: `src/pc_actor_critic/mod.rs`

The signal is computed in `step_inner()` where `process_hysteresis()` is called:

```rust
// Current code (two call sites in step_inner):
self.process_hysteresis(surprise_score, self.last_td_error.abs());
//                      ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^
//                      actor signal    critic signal
```

The change is to compute the actor signal before passing it:

```rust
let actor_signal = match self.config.actor_hysteresis_signal {
    ActorHysteresisSignal::Surprise => surprise_score,
    ActorHysteresisSignal::TdError => self.last_td_error.abs(),
    ActorHysteresisSignal::Hybrid => surprise_score.max(self.last_td_error.abs()),
};
self.process_hysteresis(actor_signal, self.last_td_error.abs());
```

### Config Addition

**File**: `src/pc_actor_critic/config.rs`

```rust
/// Signal source for actor hysteresis state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActorHysteresisSignal {
    /// PC surprise only (RMS prediction error from PC inference loop).
    /// Original behavior. Good for detecting environment changes but
    /// does not reflect agent performance.
    Surprise,
    /// |TD error| only (same signal as critic hysteresis).
    /// Directly reflects prediction accuracy of outcomes.
    TdError,
    /// max(surprise, |TD error|) — recommended default.
    /// Wakes on environment changes OR performance degradation.
    Hybrid,
}

impl Default for ActorHysteresisSignal {
    fn default() -> Self {
        ActorHysteresisSignal::Hybrid
    }
}
```

Add to `PcActorCriticConfig`:

```rust
/// Signal source for actor hysteresis. Default: Hybrid (max of surprise and |TD error|).
#[serde(default)]
pub actor_hysteresis_signal: ActorHysteresisSignal,
```

### Call Sites to Update

There are two places in `step_inner()` where `process_hysteresis` is called:

1. **TD(0) path** (when `td_steps == 0`): After `learn_continuous_inner()` returns
2. **TD(n) buffer-full path**: After learning from the oldest buffered transition

Both need the same signal computation. Extract into a helper:

```rust
fn compute_actor_hysteresis_signal(&self, surprise: f64) -> f64 {
    match self.config.actor_hysteresis_signal {
        ActorHysteresisSignal::Surprise => surprise,
        ActorHysteresisSignal::TdError => self.last_td_error.abs(),
        ActorHysteresisSignal::Hybrid => surprise.max(self.last_td_error.abs()),
    }
}
```

### Serialization

`ActorHysteresisSignal` is a simple enum — serde handles it automatically. Existing save files without the field will default to `Hybrid` (new default). For backward compatibility with files that used the old behavior, consider defaulting to `Surprise` instead and requiring explicit opt-in to `Hybrid`. This is a design decision for the core team.

### Backward Compatibility

- Default `Hybrid` changes behavior for existing configs with `actor_hysteresis = true`
- If backward compat is required, default to `Surprise` and document `Hybrid` as recommended
- Either way, all existing configs without the field will work (serde default)

## Testing Strategy

### Unit Tests

1. **test_actor_signal_surprise_only**: With `Surprise` mode, verify actor signal equals surprise_score
2. **test_actor_signal_td_error_only**: With `TdError` mode, verify actor signal equals |td_error|
3. **test_actor_signal_hybrid_max**: With `Hybrid` mode, verify actor signal equals max(surprise, |td_error|)
4. **test_hybrid_wakes_on_high_td_error**: Agent with low surprise but high TD error should transition FROZEN → PLASTIC
5. **test_hybrid_sleeps_when_both_low**: Agent with both signals low should transition PLASTIC → FROZEN
6. **test_backward_compat_default**: New config without field defaults to expected signal mode
7. **test_serialization_round_trip**: Signal mode survives save/load

### Integration Tests in PC-TicTacToe

After implementing in core and integrating in TicTacToe:

1. **test_hybrid_signal_completes_training**: Agent with Hybrid signal trains without collapse
2. **Seed test N=35**: Compare Hybrid vs Surprise-only — Hybrid should show actor freezing/waking appropriately

## Expected Impact

### On Hysteresis Behavior

With Hybrid signal, the actor should:
1. **Stay PLASTIC during active learning** (both surprise and TD error contribute)
2. **Freeze when truly converged** (both surprise AND TD error are low)
3. **Wake when performance degrades** (TD error spikes even if surprise stays low)
4. **Wake on curriculum advancement** (surprise spikes from new opponent patterns)

### On TicTacToe Results

The Hybrid signal should prevent the two failure modes observed:
- **Never-freeze** (Exp 2 with sleep_fraction=0.3): TD error component will help trigger freeze when the agent plateaus at consistent 50% loss
- **Never-wake** (Exp with sleep_fraction=0.05): TD error component will spike when frozen agent's V(s) becomes inaccurate, triggering wake

### Predicted Mean Depth Improvement

CL with TD(4) and Hybrid signal should:
- Match or exceed CL with TD(0) and EWC=0 (mean 6.71)
- Potentially approach episodic REINFORCE (mean 7.57) due to TD(4) + functional hysteresis
- Eliminate collapse at high episode counts (500k+) by freezing during convergent phases

## TOML Configuration

```toml
[agent]
# Actor hysteresis with hybrid signal
actor_hysteresis = true
actor_hysteresis_signal = "hybrid"   # "surprise" | "td_error" | "hybrid"
actor_fast_window = 20
actor_slow_window = 500
actor_wake_fraction = 0.5
actor_sleep_fraction = 0.3          # Can return to 0.3 with hybrid signal
```

## Files to Modify in pc-rl-core

| File | Change |
|------|--------|
| `src/pc_actor_critic/config.rs` | Add `ActorHysteresisSignal` enum and config field |
| `src/pc_actor_critic/mod.rs` | Add `compute_actor_hysteresis_signal()` helper, update both `process_hysteresis` call sites |
| `src/lib.rs` | Re-export `ActorHysteresisSignal` if needed |

## Files to Modify in PC-TicTacToe (after core update)

| File | Change |
|------|--------|
| `src/utils/config.rs` | Add `actor_hysteresis_signal` field to `AgentSection`, wire in `to_agent_config()` |
| `src/ui/cli.rs` | Add to `DEFAULT_CONFIG_TOML` |
| `config.toml` | Add field |
| `config_cl_experiment.toml` | Set to `"hybrid"` |
