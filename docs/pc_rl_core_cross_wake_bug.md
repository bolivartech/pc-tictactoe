# pc-rl-core bug report — Cross-wake deadlock in `process_hysteresis`

**Status:** Open
**Discovered:** 2026-04-13
**Severity:** Medium (correctness issue affecting long-running stress tests)
**Affected file:** `src/pc_actor_critic/mod.rs`
**Affected functions:** `PcActorCritic::process_hysteresis` (around line 2293)
**Reporter:** Empirically observed in PC-TicTacToe `stress-test` CLI command running
the CL-balanced+EWC configuration for 500,000 episodes
(`config_stress_cl_balanced.toml`).

---

## Symptom

In a long-running training session with both `actor_hysteresis = true` and
`critic_hysteresis = true` (and both cross-wake couplings enabled —
`actor_wakes_critic = true`, `critic_wakes_actor = true`), the actor and critic
hysteresis state machines can reach a **deadlocked equilibrium** where:

1. **Actor is permanently FROZEN** with `actor_fast ≈ actor_slow` (both EWMAs
   converged to the same value).
2. **Critic is permanently PLASTIC** with `critic_fast ≈ critic_slow` (both
   EWMAs converged to a different value).
3. **Neither network ever transitions again.** `hysteresis_transitions` counter
   stops advancing for tens of thousands of episodes.
4. **The cross-wake couplings never fire** despite both being nominally enabled,
   because `critic_woke` (the variable that gates `critic_wakes_actor`) is a
   one-shot transition flag that requires the critic to FLIP from FROZEN to
   PLASTIC in the current step. Once the critic is in stable PLASTIC equilibrium
   (no more FROZEN→PLASTIC transitions), `critic_woke` is never `true` again.

## Concrete observation

From a CL-balanced+EWC stress test on the PC-TicTacToe binary
(`stress_test_cl_balanced.csv`), checkpoints from ep 423000 to ep 500000:

```text
[ep  423000] fit=0.4000 STABLE | A=F(f=0.5434 s=0.5436) C=P(f=0.0159 s=0.0171) | trans=7333
[ep  424000] fit=0.4000 STABLE | A=F(f=0.5435 s=0.5436) C=P(f=0.0185 s=0.0170) | trans=7333
[ep  425000] fit=0.4000 STABLE | A=F(f=0.5435 s=0.5437) C=P(f=0.0161 s=0.0168) | trans=7333
...
[ep  499000] fit=0.4000 STABLE | A=F(f=0.5425 s=0.5437) C=P(f=0.0106 s=0.0106) | trans=7333
[ep  500000] fit=0.4000 STABLE | A=F(f=0.5421 s=0.5435) C=P(f=0.0117 s=0.0115) | trans=7333
```

**77,000 consecutive episodes with zero state transitions** in either network.
Both networks are mathematically locked into their current state because:

- **Actor wake threshold:** `fast > slow × (1 + wake_fraction) = 0.5436 × 1.5 ≈ 0.8154`.
  Actor `fast ≈ 0.5430`, far below threshold → cannot wake spontaneously.
- **Critic sleep threshold:** `fast < slow × (1 − sleep_fraction) = 0.0170 × 0.7 ≈ 0.0119`.
  Critic `fast ≈ 0.0117 ± 0.003` — sometimes barely under, sometimes over → may
  occasionally cross the threshold but the EWMAs immediately re-equilibrate.

In practice, neither network undergoes a clean FROZEN↔PLASTIC transition during
this window, so neither cross-wake fires.

The configuration in use:

```toml
actor_hysteresis = true
actor_fast_window = 20
actor_slow_window = 500
actor_wake_fraction = 0.5
actor_sleep_fraction = 0.005

critic_hysteresis = true
critic_fast_window = 20
critic_slow_window = 200
critic_wake_fraction = 0.5
critic_sleep_fraction = 0.3

actor_wakes_critic = true
actor_wakes_critic_threshold = 1000
critic_wakes_actor = true
critic_wakes_actor_threshold = 1000

ewc_lambda = 0.1
```

## Root cause analysis

In `pc_actor_critic/mod.rs` `process_hysteresis()` (around line 2293):

```rust
pub(crate) fn process_hysteresis(&mut self, actor_signal: f64, critic_signal: f64) {
    let mut actor_woke = false;
    let mut actor_slept = false;
    let mut critic_woke = false;
    let mut critic_slept = false;

    // Update actor hysteresis ...
    if let Some(new_state) = hyst.update(actor_signal) {
        if new_state == PlasticityState::Plastic {
            actor_woke = true;        // ← only set on the transition step
            self.actor_plastic_step_counter = 0;
            self.actor_frozen_steps = 0;
        } else {
            actor_slept = true;
        }
    }

    // Update critic hysteresis ...
    if let Some(new_state) = hyst.update(critic_signal) {
        if new_state == PlasticityState::Plastic {
            critic_woke = true;       // ← only set on the transition step
            self.critic_plastic_step_counter = 0;
            self.critic_frozen_steps = 0;
        } else {
            critic_slept = true;
        }
    }

    // Actor wakes critic coupling
    if actor_woke && self.config.actor_wakes_critic {       // ← guarded by transition flag
        if let Some(ref mut critic_hyst) = self.critic_hysteresis {
            if critic_hyst.state == PlasticityState::Frozen
                && self.critic_frozen_steps >= self.config.actor_wakes_critic_threshold
            {
                critic_hyst.state = PlasticityState::Plastic;
                // ...
                critic_woke = true;
            }
        }
    }

    // Critic wakes actor coupling
    if critic_woke && self.config.critic_wakes_actor {      // ← guarded by transition flag
        if let Some(ref mut actor_hyst) = self.actor_hysteresis {
            if actor_hyst.state == PlasticityState::Frozen
                && self.actor_frozen_steps >= self.config.critic_wakes_actor_threshold
            {
                actor_hyst.state = PlasticityState::Plastic;
                // ...
                actor_woke = true;
            }
        }
    }
    // ...
}
```

The cross-wake guards `actor_woke` and `critic_woke` are **local one-shot flags
re-initialized to `false` at the top of every call** to `process_hysteresis`.
They are only set to `true` when `hyst.update(...)` returns `Some(Plastic)` —
i.e., **only on the exact step where the source network transitions
FROZEN→PLASTIC**.

Once a network reaches a sustained PLASTIC state with `fast ≈ slow` in
equilibrium, it stops transitioning. Subsequent calls to `process_hysteresis`
see `hyst.update(...)` returning `None` (no transition) → `critic_woke` stays
`false` → cross-wake guard fails → the other network's frozen state is never
challenged.

The symptom is asymmetric:
- The currently-PLASTIC network ages indefinitely — its
  `*_plastic_step_counter` keeps incrementing — but that data is **not used**
  by the cross-wake logic.
- The currently-FROZEN network has already failed its own spontaneous-wake
  check (otherwise it wouldn't be frozen), and the only escape was supposed
  to be the cross-wake from the other side, which now never fires.

The author's existing comment at line 2358-2365 documents the design intent
to avoid wake cascades:

> `Both couplings can coexist safely: no cascade is possible because the
> target guard checks "state == Frozen" — a network that just woke (naturally
> or via coupling) is Plastic, so the reverse coupling's guard fails.`

That reasoning correctly prevents wake-ping-pong cascades within a single step.
But the implementation overshoots: it gates the cross-wake on the **transition
event** instead of on the **sustained state**, which makes the coupling
**unable to fire when one network is in long-term equilibrium plasticity**.

## Existing test coverage

The unit tests at `mod.rs:6675` `critic_wakes_actor_coupling_default_threshold`
and similar **only verify the transition-moment behavior**: they put the critic
into a state about to transition, call one update, and check that the actor
wakes. None of them exercise the case where the critic has been continuously
plastic for many steps without further transitions.

This is why the bug escaped CI — the test surface is "what happens at the
transition step?" but the production failure mode is "what happens after the
transition stops happening?".

## Proposed fix

Two complementary changes:

### 1. Augment the cross-wake guards to also trigger on sustained state

Modify both cross-wake blocks so that the trigger fires on **either**
(a) the transition flag, **or** (b) the source network having been in PLASTIC
for at least the threshold number of steps. The required counter
(`*_plastic_step_counter`) already exists in the `PcActorCritic` struct — it's
incremented every step the network is plastic and reset to 0 on
FROZEN→PLASTIC transitions. Just check it.

```rust
// Actor wakes critic — fires on transition OR sustained plastic state
let actor_should_wake_critic = self.config.actor_wakes_critic && (
    actor_woke
    || (
        self.actor_hysteresis.as_ref()
            .map(|h| h.state == PlasticityState::Plastic)
            .unwrap_or(false)
        && self.actor_plastic_step_counter >= self.config.actor_wakes_critic_threshold
    )
);

if actor_should_wake_critic {
    if let Some(ref mut critic_hyst) = self.critic_hysteresis {
        if critic_hyst.state == PlasticityState::Frozen
            && self.critic_frozen_steps >= self.config.actor_wakes_critic_threshold
        {
            critic_hyst.state = PlasticityState::Plastic;
            critic_hyst.fast.k = 0;
            critic_hyst.slow.k = 0;
            self.critic_plastic_step_counter = 0;
            self.critic_frozen_steps = 0;
            critic_woke = true;
            // CRITICAL: reset the source counter too, so the sustained-plastic
            // path doesn't re-fire on every subsequent step. The cross-wake
            // should be a once-per-threshold event, not a per-step event.
            self.actor_plastic_step_counter = 0;
        }
    }
}

// Symmetric block for critic wakes actor
let critic_should_wake_actor = self.config.critic_wakes_actor && (
    critic_woke
    || (
        self.critic_hysteresis.as_ref()
            .map(|h| h.state == PlasticityState::Plastic)
            .unwrap_or(false)
        && self.critic_plastic_step_counter >= self.config.critic_wakes_actor_threshold
    )
);

if critic_should_wake_actor {
    if let Some(ref mut actor_hyst) = self.actor_hysteresis {
        if actor_hyst.state == PlasticityState::Frozen
            && self.actor_frozen_steps >= self.config.critic_wakes_actor_threshold
        {
            actor_hyst.state = PlasticityState::Plastic;
            actor_hyst.fast.k = 0;
            actor_hyst.slow.k = 0;
            self.actor_plastic_step_counter = 0;
            self.actor_frozen_steps = 0;
            actor_woke = true;
            // Same source counter reset to prevent per-step re-fire.
            self.critic_plastic_step_counter = 0;
        }
    }
}
```

**Critical detail:** the source network's `*_plastic_step_counter` MUST be
reset to 0 inside the cross-wake fire block (after the target's counters are
reset). Without this reset, the sustained-plastic condition would remain true
on every subsequent step, firing the cross-wake every step. With the reset,
the cross-wake fires at most once per `threshold` steps of sustained plasticity.

### 2. Verify cascade prevention is preserved

The author's stated invariant — "no wake-cascade is possible because the
target guard checks `state == Frozen`" — still holds with the new logic:

- After `critic_wakes_actor` fires, `actor_hyst.state = Plastic` and
  `critic_plastic_step_counter = 0`.
- On the next `process_hysteresis` call, `actor_woke` is `false` (set by
  `update()` only on transitions, and the actor was already plastic from the
  cross-wake — `update()` won't re-fire that state).
- Wait — actually, the actor's `update()` will see fresh data after the wake
  (because we reset `actor_hyst.fast.k = 0` and `actor_hyst.slow.k = 0`),
  and may trigger natural transitions later. That's intended.
- The new sustained-plasticity branch for `actor_should_wake_critic` requires
  `actor_plastic_step_counter >= actor_wakes_critic_threshold`. Since we just
  reset that counter to 0 above, it won't fire until at least `threshold`
  more steps have elapsed. **No cascade possible within the same step.**

The test at `mod.rs:6675` `critic_wakes_actor_coupling_default_threshold` and
related should still pass unchanged, since the transition path
(`actor_woke || critic_woke`) is preserved as the first branch of the OR.

### 3. New test cases to add

```rust
#[test]
fn critic_wakes_actor_after_sustained_plastic_state() {
    // Setup: critic in PLASTIC equilibrium for >= threshold steps,
    // never transitioning, never re-firing the one-shot critic_woke flag.
    // Actor in FROZEN with frozen_steps >= threshold.
    // Expected: cross-wake fires once when critic_plastic_step_counter
    // reaches threshold, and the actor wakes.

    let mut cfg = make_test_config();
    cfg.actor_hysteresis = true;
    cfg.critic_hysteresis = true;
    cfg.critic_wakes_actor = true;
    cfg.critic_wakes_actor_threshold = 50;
    let mut agent = PcActorCritic::new(CpuLinAlg::new(), cfg, 42).unwrap();

    // Force critic into stable plastic, actor into long-term frozen
    // (use direct field manipulation via a #[cfg(test)] helper, or run
    //  a controlled signal sequence to reach the desired state).
    // ...

    // Run process_hysteresis with signals that produce no critic transition
    // for 100 consecutive steps.
    for _ in 0..100 {
        agent.process_hysteresis(actor_steady_signal, critic_steady_signal);
    }

    // Assert: actor was woken via cross-wake at some point during those 100 steps
    assert_eq!(agent.actor_hysteresis.as_ref().unwrap().state, PlasticityState::Plastic);
}

#[test]
fn cross_wake_does_not_repeatedly_fire_on_every_step() {
    // After a sustained-plastic cross-wake fires, the source counter is
    // reset, so the next fire requires another `threshold` steps of
    // continuous plasticity.

    // Setup, run cross-wake fire, then assert that
    // *_plastic_step_counter == 0 after the fire.
}

#[test]
fn deadlock_recovery_is_observable() {
    // Reproduce the PC-TicTacToe stress test deadlock scenario: actor
    // permanently FROZEN with low signal variance, critic permanently
    // PLASTIC with low TD error. After the fix, verify that cross-wake
    // fires every threshold steps and actor cycles back to plastic.
}
```

## Validation steps after fix

1. Run pc-rl-core's existing test suite — should pass unchanged
   (`cargo test -p pc-rl-core`).
2. Add the three new test cases above.
3. In PC-TicTacToe (downstream), re-run a single CL-balanced+EWC stress test
   for at least 500k episodes:
   ```bash
   cargo run --release -- stress-test \
       -c config_stress_cl_balanced.toml \
       --max-episodes 500000
   ```
4. Inspect `stress_test_cl_balanced.csv`: the `hysteresis_transitions`
   counter should keep advancing past ep 200k-300k (it currently locks at
   ~7333 in the buggy build).
5. The actor `actor_state` column should show `P` periodically throughout
   the run, not get stuck in `F` indefinitely.
6. Whether the fitness trajectory changes (i.e. whether
   `fitness=0.40` is reached or avoided) is the scientific question this
   experiment was designed to answer; the cross-wake fix is a precondition
   for the answer to be valid.

## Why this matters for the downstream PC-TicTacToe experiment

The bug invalidates the "EWC prevents cascade to 0.40" conclusion drawn from
the CL-balanced+EWC 500k run. Without working cross-wake, the system enters
a deadlock where the actor cannot adapt and EWC's per-parameter protection
cannot be exercised in a meaningful way (no gradients are flowing through
the actor when it's frozen). The 0.40 final fitness in that run reflects
the deadlock state, not the failure of EWC to protect against catastrophic
forgetting. After the fix, the experiment should be re-run and the result
re-interpreted.
