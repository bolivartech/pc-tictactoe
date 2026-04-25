# Opcion B — Wire `replay_learn()` en ContinuousTrainer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrar `replay_learn()` de `pc-rl-core` en `ContinuousTrainer` con warmup gate por primer curriculum advancement, invocacion cada `replay_interval` episodios, y seal idempotente.

**Architecture:** Extension aditiva de `ContinuousTrainer` (`src/training/continuous.rs`) con 5 fields nuevos + 2 getters publicos. Nuevo campo `replay_interval` en `TrainingSection` del TOML con validacion. Cero cambios a `pc-rl-core`, cero cambios breaking a APIs publicas del trainer.

**Tech Stack:** Rust 2021, `cargo nextest`, TDD-Guard hooks, consumer de `pc-rl-core` (local path dep).

---

## Pre-flight — antes de cualquier tarea

- [ ] **P.1:** Verificar estado inicial limpio:
  ```bash
  cd /d/jbolivarg/RustProjects/PC-TicTacToe
  git status
  ```
  Expected: branch `self-recovery`, working tree limpio salvo por archivos ya documentados como ignorables. Los archivos nuevos del feature (`sbtdd/spec-behavior-base.md`, `sbtdd/spec-behavior.md`, `.gitkeep` en `sbtdd/` y `planning/`) pueden estar untracked — commitearlos ANTES de arrancar TDD con prefijo `docs:`.

- [ ] **P.2:** Verificar baseline pasa:
  ```bash
  cargo nextest run 2>&1 | tail -3
  cargo clippy --tests -- -D warnings 2>&1 | tail -3
  cargo fmt --check
  ```
  Expected: `162 tests run: 162 passed`; clippy limpio; fmt limpio.

- [ ] **P.3:** Inicializar `.claude/session-state.json` via `/subagent-driven-development` — el skill lo crea automaticamente segun §2.3 del `CLAUDE.local.md` al leer el plan (primera tarea + Red).

---

## Task 1: Campo `replay_interval` en `TrainingSection`

**Alcance:** Agregar el field nuevo al TOML config, helper de default, actualizar `impl Default`. Sin validacion todavia (Tarea 2).

**Dependencies:** ninguna.

**Mapea scenarios:** 4.7 (parcial — default accesible).

**Files:**
- Modify: `src/utils/config.rs`
  - Struct `TrainingSection` (L312-325)
  - `impl Default for TrainingSection` (L796-805)
  - Default helpers block (~L593 area)
  - Test module (bottom of file, `mod tests`)

### Red

- [ ] **1.1: Escribir tests de default + parsing**

Archivo: `src/utils/config.rs`, dentro del modulo `#[cfg(test)] mod tests { ... }` (antes de `}` de cierre).

```rust
#[test]
fn test_replay_interval_default_is_100() {
    let config = AppConfig::default();
    assert_eq!(config.training.replay_interval, 100);
}

#[test]
fn test_replay_interval_parses_from_toml() {
    let toml_str = r#"
[agent]
[agent.actor]
[[agent.actor.hidden_layers]]
size = 18
activation = "tanh"
[agent.critic]
input_size = 27
[[agent.critic.hidden_layers]]
size = 36
activation = "tanh"
[training]
replay_interval = 250
"#;
    let config: AppConfig = toml::from_str(toml_str).unwrap();
    assert_eq!(config.training.replay_interval, 250);
    assert!(config.validate().is_ok());
}
```

- [ ] **1.2: Run Red — tests deben fallar por compilacion**

```bash
cargo nextest run --no-fail-fast utils::config::tests::test_replay_interval 2>&1 | tail -10
```
Expected: compile error — `no field `replay_interval` on type `TrainingSection``.

- [ ] **1.3: Invocar `/verification-before-completion` (Red)**

Confirmar que los tests fallan por la razon correcta (field no existe en struct), no por error de sintaxis u otro bug en el test.

- [ ] **1.4: Commit Red**

```bash
git add src/utils/config.rs
git commit -m "test: add replay_interval default and parsing tests (Task 1 Red)"
```

### Green

- [ ] **1.5: Agregar default helper**

Archivo: `src/utils/config.rs`, insertar despues de `default_seed()` (~L598):

```rust
/// Default episodios entre invocaciones de `replay_learn` cuando el
/// replay buffer esta activo.
fn default_replay_interval() -> usize {
    100
}
```

- [ ] **1.6: Agregar field al struct**

Archivo: `src/utils/config.rs`, modificar `TrainingSection` (L312-325). Agregar despues del field `seed`:

```rust
    /// Episodios entre invocaciones de `replay_learn` cuando el replay buffer
    /// esta activo (`agent.replay_training_capacity > 0`). Default: 100.
    ///
    /// Trade-off: valores bajos -> mas overhead de replay por run; valores altos
    /// -> buffer mas sesgado por FIFO eviction en compartment B.
    /// Ignorado silenciosamente si `replay_training_capacity == 0`.
    #[serde(default = "default_replay_interval")]
    pub replay_interval: usize,
```

- [ ] **1.7: Actualizar `impl Default for TrainingSection`**

Archivo: `src/utils/config.rs`, L796-805:

```rust
impl Default for TrainingSection {
    fn default() -> Self {
        Self {
            episodes: default_episodes(),
            checkpoint_interval: default_checkpoint_interval(),
            log_interval: default_log_interval(),
            seed: default_seed(),
            replay_interval: default_replay_interval(),
        }
    }
}
```

- [ ] **1.8: Run Green — tests deben pasar**

```bash
cargo nextest run utils::config::tests::test_replay_interval 2>&1 | tail -5
```
Expected: `2 tests run: 2 passed`.

Luego full suite:
```bash
cargo nextest run 2>&1 | tail -3
```
Expected: `164 tests run: 164 passed` (162 + 2 nuevos).

- [ ] **1.9: Invocar `/verification-before-completion` (Green)**

Correr la suite de §0.1 completa:
```bash
cargo nextest run 2>&1 | tail -3
cargo clippy --tests -- -D warnings 2>&1 | tail -3
cargo fmt --check
cargo build --release 2>&1 | tail -3
```
Expected: todos limpios.

- [ ] **1.10: Commit Green**

```bash
git add src/utils/config.rs
git commit -m "feat: add replay_interval field to TrainingSection (Task 1 Green)"
```

### Refactor

- [ ] **1.11: Verificar rustdoc es completo**

Revisar que el rustdoc del field `replay_interval` cubre: proposito, default, trade-off, relacion con `replay_training_capacity`. Si falta algo, editar inline.

- [ ] **1.12: Run Refactor checks**

```bash
cargo doc --no-deps 2>&1 | tail -5
cargo nextest run 2>&1 | tail -3
cargo clippy --tests -- -D warnings 2>&1 | tail -3
cargo fmt --check
```
Expected: todos limpios.

- [ ] **1.13: Invocar `/verification-before-completion` (Refactor)**

- [ ] **1.14: Commit Refactor (si hubo cambios)**

Si no hubo cambios en Refactor, skip commit. Si los hubo:
```bash
git add src/utils/config.rs
git commit -m "refactor: polish rustdoc on replay_interval field (Task 1 Refactor)"
```

### Cierre de tarea

- [ ] **1.15: Marcar tarea completa en el plan**

Editar `planning/claude-plan-tdd.md` (el plan vivo) para marcar `Task 1` como `[x]`.

```bash
git add planning/claude-plan-tdd.md
git commit -m "chore: mark task 1 complete"
```

- [ ] **1.16: Actualizar `.claude/session-state.json`**

El skill actualiza automaticamente: `current_task_id: "2"`, `current_phase: "red"`, `phase_started_at_commit: <SHA del chore:>`.

---

## Task 2: Validacion TOML

**Alcance:** Agregar regla de validacion que rechaza `replay_training_capacity > 0 && replay_interval == 0`.

**Dependencies:** Task 1.

**Mapea scenarios:** 4.5, 4.6.

**Files:**
- Modify: `src/utils/config.rs`
  - Metodo `validate_cl()` — bloque Phase 2 (~L1063-1100)
  - Test module

### Red

- [ ] **2.1: Escribir tests de validacion**

Archivo: `src/utils/config.rs`, dentro de `mod tests`.

```rust
#[test]
fn test_validation_rejects_interval_zero_with_buffer_active() {
    let mut config = AppConfig::default();
    config.agent.replay_training_capacity = 1024;
    config.training.replay_interval = 0;
    let err = config.validate().unwrap_err();
    assert!(
        err.message.contains("replay_interval must be > 0"),
        "expected 'replay_interval must be > 0' in error message, got: {}",
        err.message
    );
}

#[test]
fn test_validation_allows_interval_zero_with_buffer_off() {
    let mut config = AppConfig::default();
    config.agent.replay_training_capacity = 0;
    config.training.replay_interval = 0;
    assert!(config.validate().is_ok());
}
```

- [ ] **2.2: Run Red — tests deben fallar**

```bash
cargo nextest run utils::config::tests::test_validation_rejects_interval_zero 2>&1 | tail -10
```
Expected: test `test_validation_rejects_interval_zero_with_buffer_active` falla porque validate retorna Ok() (la regla aun no existe). Test `test_validation_allows_interval_zero_with_buffer_off` pasa (es el caso permisivo que ya funciona).

- [ ] **2.3: Invocar `/verification-before-completion` (Red)**

Confirmar que el test fails por la razon correcta (validate no rechaza la config invalida).

- [ ] **2.4: Commit Red**

```bash
git add src/utils/config.rs
git commit -m "test: add replay_interval validation tests (Task 2 Red)"
```

### Green

- [ ] **2.5: Agregar la regla de validacion**

Archivo: `src/utils/config.rs`, metodo `validate_cl()`. Insertar al final del bloque `// Phase 2: distillation + replay` (despues de la regla de `replay_batch_size`, antes del `Ok(())` de cierre del bloque ~L1100).

Buscar el bloque existente que termina con:
```rust
        if (a.replay_training_capacity > 0 || a.replay_recent_capacity > 0)
            && a.replay_batch_size == 0
        {
            return Err(ConfigError {
                message: "replay_batch_size must be > 0 when replay buffer is enabled".to_string(),
            });
        }

        Ok(())
    }
```

Insertar antes del `Ok(())`:

```rust
        // Phase 2 orchestration (Opcion B — wire replay_learn):
        // replay_interval must be > 0 when buffer is active so the trainer
        // can fire replay_learn on a fixed cadence.
        if a.replay_training_capacity > 0 && self.training.replay_interval == 0 {
            return Err(ConfigError {
                message: "training.replay_interval must be > 0 when agent.replay_training_capacity > 0".to_string(),
            });
        }
```

- [ ] **2.6: Run Green — tests deben pasar**

```bash
cargo nextest run utils::config::tests::test_validation 2>&1 | tail -5
```
Expected: ambos tests pasan.

Full suite:
```bash
cargo nextest run 2>&1 | tail -3
```
Expected: `166 tests run: 166 passed` (162 + 2 Task 1 + 2 Task 2).

- [ ] **2.7: Invocar `/verification-before-completion` (Green)**

```bash
cargo nextest run 2>&1 | tail -3
cargo clippy --tests -- -D warnings 2>&1 | tail -3
cargo fmt --check
cargo build --release 2>&1 | tail -3
```

- [ ] **2.8: Commit Green**

```bash
git add src/utils/config.rs
git commit -m "feat: validate replay_interval > 0 when buffer active (Task 2 Green)"
```

### Refactor

- [ ] **2.9: Verificar mensaje de error es consistente**

Revisar que el mensaje de error (`"training.replay_interval must be > 0 when agent.replay_training_capacity > 0"`) sigue el patron de los otros errores del bloque. Si hay oportunidad de unificar, hacerlo. Si no, skip.

- [ ] **2.10: Run Refactor checks**

```bash
cargo nextest run 2>&1 | tail -3
cargo clippy --tests -- -D warnings 2>&1 | tail -3
cargo fmt --check
```

- [ ] **2.11: Invocar `/verification-before-completion` (Refactor)**

- [ ] **2.12: Commit Refactor (si hubo cambios)**

Si no hubo, skip.

### Cierre de tarea

- [ ] **2.13: Marcar tarea completa + actualizar state**

```bash
# Editar planning/claude-plan-tdd.md: marcar Task 2 como [x]
git add planning/claude-plan-tdd.md
git commit -m "chore: mark task 2 complete"
```

---

## Task 3: Fields + constructor + getters en `ContinuousTrainer`

**Alcance:** Agregar los 5 fields nuevos al struct, inicializarlos desde `AppConfig` en `new()`, exponer getters `replay_invocations()` y `training_memories_sealed()`. Sin logica de replay en `train()` todavia.

**Dependencies:** Task 1.

**Mapea scenarios:** base state para 4.1, 4.4, 4.7.

**Files:**
- Modify: `src/training/continuous.rs`
  - Struct `ContinuousTrainer` (L74-101)
  - `impl ContinuousTrainer::new()` (L111-133)
  - Getters section (~L279 area, cerca de `episode_count()`)
  - Test module (`#[cfg(test)] mod tests`)

### Red

- [ ] **3.1: Escribir test de estado inicial del trainer**

Archivo: `src/training/continuous.rs`, dentro de `mod tests`. Antes: identificar si ya existe un helper de construccion de test trainer; si no, crearlo.

Primero, helper `#[cfg(test)]` (al inicio del `mod tests`):

```rust
#[cfg(test)]
fn build_test_trainer(
    replay_training_capacity: usize,
    replay_interval: usize,
    advance_threshold: f64,
    window_size: usize,
    max_episodes: usize,
) -> ContinuousTrainer {
    use crate::utils::config::AppConfig;
    use pc_rl_core::linalg::cpu::CpuLinAlg;
    use pc_rl_core::pc_actor_critic::PcActorCritic;

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
```

Luego, tests de estado inicial:

```rust
#[test]
fn test_trainer_construction_phase2_off_initial_state() {
    let trainer = build_test_trainer(
        /* replay_training_capacity */ 0,
        /* replay_interval */ 100,
        /* advance_threshold */ 0.95,
        /* window_size */ 100,
        /* max_episodes */ 200,
    );
    assert_eq!(trainer.replay_invocations(), 0);
    assert!(!trainer.training_memories_sealed());
}

#[test]
fn test_trainer_construction_phase2_on_initial_state() {
    let trainer = build_test_trainer(
        /* replay_training_capacity */ 256,
        /* replay_interval */ 50,
        /* advance_threshold */ 0.30,
        /* window_size */ 20,
        /* max_episodes */ 100,
    );
    assert_eq!(trainer.replay_invocations(), 0);
    assert!(!trainer.training_memories_sealed());
}
```

- [ ] **3.2: Run Red — tests deben fallar por compilacion**

```bash
cargo nextest run training::continuous::tests::test_trainer_construction 2>&1 | tail -10
```
Expected: compile error — `method `replay_invocations` not found` y similar para `training_memories_sealed`.

- [ ] **3.3: Invocar `/verification-before-completion` (Red)**

Confirmar fails por API faltante, no por bug en helper/test.

- [ ] **3.4: Commit Red**

```bash
git add src/training/continuous.rs
git commit -m "test: add ContinuousTrainer replay fields initial state tests (Task 3 Red)"
```

### Green

- [ ] **3.5: Agregar los 5 fields al struct**

Archivo: `src/training/continuous.rs`, struct `ContinuousTrainer` (L74-101). Agregar despues del field `last_agent_side`:

```rust
    /// True iff `config.agent.replay_training_capacity > 0` at construction.
    /// Cached to avoid per-iteration config lookups and coupling to agent internals.
    replay_enabled: bool,
    /// Episodios entre invocaciones de `replay_learn` (de `config.training.replay_interval`).
    /// Ignorado si `replay_enabled == false`.
    replay_interval: usize,
    /// Batch size para cada `replay_learn` (de `config.agent.replay_batch_size`).
    replay_batch_size: usize,
    /// True despues del primer `seal_replay_training_memories()` exitoso.
    /// Gate del warmup: `replay_learn` no se invoca hasta que esto sea `true`.
    training_memories_sealed: bool,
    /// Counter de invocaciones exitosas (`Ok`) de `replay_learn`. Diagnostic.
    replay_invocations: usize,
```

- [ ] **3.6: Actualizar `ContinuousTrainer::new()`**

Archivo: `src/training/continuous.rs`, L111-133. Agregar despues de `last_agent_side: Player::One,`:

```rust
            replay_enabled: config.agent.replay_training_capacity > 0,
            replay_interval: config.training.replay_interval,
            replay_batch_size: config.agent.replay_batch_size,
            training_memories_sealed: false,
            replay_invocations: 0,
```

- [ ] **3.7: Agregar los getters**

Archivo: `src/training/continuous.rs`, en `impl ContinuousTrainer`, despues de `episode_count()` / `step_count()` / `current_depth()` (~L290). Localizar el bloque de getters publicos existente e insertar:

```rust
    /// Counter de invocaciones exitosas de `replay_learn` desde la construccion.
    ///
    /// Primarily for testing and diagnostic purposes.
    pub fn replay_invocations(&self) -> usize {
        self.replay_invocations
    }

    /// Estado del flag de seal: `true` despues del primer `seal_replay_training_memories()` exitoso.
    ///
    /// Primarily for testing and diagnostic purposes.
    pub fn training_memories_sealed(&self) -> bool {
        self.training_memories_sealed
    }
```

- [ ] **3.8: Run Green — tests deben pasar**

```bash
cargo nextest run training::continuous::tests::test_trainer_construction 2>&1 | tail -5
```
Expected: `2 tests run: 2 passed`.

Full suite:
```bash
cargo nextest run 2>&1 | tail -3
```
Expected: `168 tests run: 168 passed`.

- [ ] **3.9: Invocar `/verification-before-completion` (Green)**

```bash
cargo nextest run 2>&1 | tail -3
cargo clippy --tests -- -D warnings 2>&1 | tail -3
cargo fmt --check
cargo build --release 2>&1 | tail -3
```

- [ ] **3.10: Commit Green**

```bash
git add src/training/continuous.rs
git commit -m "feat: add replay fields and getters to ContinuousTrainer (Task 3 Green)"
```

### Refactor

- [ ] **3.11: Verificar rustdoc en getters y `replay_enabled`**

Revisar que los rustdoc son claros sobre el proposito diagnostico/test. Si el getter `replay_invocations()` necesita ampliar el rustdoc con "cuenta solo Ok de replay_learn, no invocaciones intentadas que fallaron", editar inline.

- [ ] **3.12: Run Refactor checks**

```bash
cargo doc --no-deps 2>&1 | tail -5
cargo nextest run 2>&1 | tail -3
cargo clippy --tests -- -D warnings 2>&1 | tail -3
cargo fmt --check
```

- [ ] **3.13: Invocar `/verification-before-completion` (Refactor)**

- [ ] **3.14: Commit Refactor (si hubo cambios)**

Si no hubo, skip.

### Cierre de tarea

- [ ] **3.15: Marcar tarea completa + actualizar state**

```bash
# Editar planning/claude-plan-tdd.md: Task 3 -> [x]
git add planning/claude-plan-tdd.md
git commit -m "chore: mark task 3 complete"
```

---

## Task 4: Seal al primer curriculum advance

**Alcance:** Invocar `agent.seal_replay_training_memories()` en el primer curriculum advance cuando `replay_enabled && !sealed`. Idempotencia via el flag `sealed`. Error handling log-warn-skip.

**Dependencies:** Task 3.

**Mapea scenarios:** 4.3, 4.4 (parcial — sealed=false antes del advance).

**Files:**
- Modify: `src/training/continuous.rs`
  - Loop en `train()` — bloque curriculum advance (L149-159)
  - Test module

### Red

- [ ] **4.1: Escribir tests de seal**

Archivo: `src/training/continuous.rs`, dentro de `mod tests`:

```rust
#[test]
fn test_scenario_4_3_seal_only_once_on_first_advance() {
    // Given: Phase 2 active, easy curriculum para forzar advance temprano
    let mut trainer = build_test_trainer(
        /* replay_training_capacity */ 256,
        /* replay_interval */ 100,
        /* advance_threshold */ 0.30,
        /* window_size */ 20,
        /* max_episodes */ 200,
    );
    // When: corre hasta completar max_episodes (multiples advances esperados)
    trainer.train();
    // Then: sealed == true, invariante post-primer-advance
    assert!(
        trainer.training_memories_sealed(),
        "sealed debe ser true despues del curriculum advance"
    );
    // El seal idempotente no se puede observar por repetidas invocaciones
    // directamente sin mock; aqui validamos que el flag quedo true (no false)
    // tras multiples advances. Log-based verification se cubre post-hoc.
}

#[test]
fn test_scenario_4_4_sealed_false_before_first_advance() {
    // Given: Phase 2 active, advance_threshold imposible para tests cortos
    let mut trainer = build_test_trainer(
        /* replay_training_capacity */ 256,
        /* replay_interval */ 5,
        /* advance_threshold */ 0.999,
        /* window_size */ 100,
        /* max_episodes */ 50,
    );
    // When: corre 50 episodios sin advance
    trainer.train();
    // Then: sealed permanece false
    assert!(
        !trainer.training_memories_sealed(),
        "sealed debe permanecer false sin advance"
    );
}
```

- [ ] **4.2: Run Red — tests deben fallar**

```bash
cargo nextest run training::continuous::tests::test_scenario_4_3_seal_only_once_on_first_advance training::continuous::tests::test_scenario_4_4_sealed_false_before_first_advance 2>&1 | tail -10
```
Expected: `test_scenario_4_3_...` falla — `sealed` queda `false` porque la logica de seal aun no existe. `test_scenario_4_4_...` pasa (estado inicial ya es false).

- [ ] **4.3: Invocar `/verification-before-completion` (Red)**

Confirmar que el fail de `test_scenario_4_3` es por "sealed should be true but was false", no otro bug.

- [ ] **4.4: Commit Red**

```bash
git add src/training/continuous.rs
git commit -m "test: add seal idempotency and pre-advance state tests (Task 4 Red)"
```

### Green

- [ ] **4.5: Agregar la logica de seal en `train()`**

Archivo: `src/training/continuous.rs`, L149-159. Modificar el bloque de curriculum advance.

Antes (existente):
```rust
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
            }
```

Despues (con [P2] seal trigger):
```rust
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

                // [P2] Seal al primer curriculum advance (warmup B gate).
                // Idempotencia via self.training_memories_sealed flag.
                if !self.training_memories_sealed && self.replay_enabled {
                    match self.agent.seal_replay_training_memories() {
                        Ok(()) => {
                            self.training_memories_sealed = true;
                            let line = format!(
                                "[ep {}] replay training memories sealed (curriculum advance {}→{})",
                                self.episode_count, prev_depth, self.current_depth,
                            );
                            eprintln!("{line}");
                            self.log_lines.push(line);
                        }
                        Err(e) => {
                            let line = format!(
                                "[ep {}] seal_replay_training_memories failed: {} (will retry next advance)",
                                self.episode_count, e,
                            );
                            eprintln!("{line}");
                            self.log_lines.push(line);
                        }
                    }
                }
            }
```

- [ ] **4.6: Run Green — tests deben pasar**

```bash
cargo nextest run training::continuous::tests::test_scenario_4_3_seal_only_once_on_first_advance training::continuous::tests::test_scenario_4_4_sealed_false_before_first_advance 2>&1 | tail -5
```
Expected: ambos pasan.

Full suite:
```bash
cargo nextest run 2>&1 | tail -3
```
Expected: `170 tests run: 170 passed`.

- [ ] **4.7: Invocar `/verification-before-completion` (Green)**

```bash
cargo nextest run 2>&1 | tail -3
cargo clippy --tests -- -D warnings 2>&1 | tail -3
cargo fmt --check
cargo build --release 2>&1 | tail -3
```

- [ ] **4.8: Commit Green**

```bash
git add src/training/continuous.rs
git commit -m "feat: seal replay training memories on first curriculum advance (Task 4 Green)"
```

### Refactor

- [ ] **4.9: Evaluar extraccion a helper privado**

Si el bloque de seal dentro del curriculum advance queda denso (>10 lineas), extraer a helper privado:

```rust
impl ContinuousTrainer {
    /// [P2] Attempts to seal replay training memories on first curriculum advance.
    /// No-op if already sealed or replay not enabled. Error handling: log warn + retry.
    fn try_seal_on_first_advance(&mut self, prev_depth: usize) {
        if self.training_memories_sealed || !self.replay_enabled {
            return;
        }
        match self.agent.seal_replay_training_memories() {
            Ok(()) => {
                self.training_memories_sealed = true;
                let line = format!(
                    "[ep {}] replay training memories sealed (curriculum advance {}→{})",
                    self.episode_count, prev_depth, self.current_depth,
                );
                eprintln!("{line}");
                self.log_lines.push(line);
            }
            Err(e) => {
                let line = format!(
                    "[ep {}] seal_replay_training_memories failed: {} (will retry next advance)",
                    self.episode_count, e,
                );
                eprintln!("{line}");
                self.log_lines.push(line);
            }
        }
    }
}
```

Y reemplazar la invocacion en `train()` con `self.try_seal_on_first_advance(prev_depth);`.

Si el bloque no queda denso (<10 lineas), skip la extraccion.

- [ ] **4.10: Run Refactor checks**

```bash
cargo nextest run 2>&1 | tail -3
cargo clippy --tests -- -D warnings 2>&1 | tail -3
cargo fmt --check
```

- [ ] **4.11: Invocar `/verification-before-completion` (Refactor)**

- [ ] **4.12: Commit Refactor (si hubo cambios)**

```bash
git add src/training/continuous.rs
git commit -m "refactor: extract try_seal_on_first_advance helper (Task 4 Refactor)"
```

### Cierre de tarea

- [ ] **4.13: Marcar tarea completa + actualizar state**

```bash
# Editar planning/claude-plan-tdd.md: Task 4 -> [x]
git add planning/claude-plan-tdd.md
git commit -m "chore: mark task 4 complete"
```

---

## Task 5: Replay trigger interval-based

**Alcance:** Invocar `agent.replay_learn(batch_size)` cada `replay_interval` episodios cuando `sealed == true && replay_interval > 0`. Incrementar `replay_invocations` on Ok. Error handling log-warn-skip.

**Dependencies:** Task 4.

**Mapea scenarios:** 4.1, 4.2, 4.4 (completo), 4.7.

**Files:**
- Modify: `src/training/continuous.rs`
  - Loop en `train()` — despues de `self.episode_count += 1;` (~L162)
  - Test module

### Red

- [ ] **5.1: Escribir tests de replay trigger**

Archivo: `src/training/continuous.rs`, dentro de `mod tests`:

```rust
#[test]
fn test_scenario_4_1_replay_inactive_when_phase_2_off() {
    // Given: Phase 2 off, replay_interval default
    let mut trainer = build_test_trainer(
        /* replay_training_capacity */ 0,
        /* replay_interval */ 100,
        /* advance_threshold */ 0.95,
        /* window_size */ 100,
        /* max_episodes */ 200,
    );
    // When: corre 200 episodios
    trainer.train();
    // Then: replay nunca se invoca, seal nunca ocurre
    assert_eq!(trainer.replay_invocations(), 0);
    assert!(!trainer.training_memories_sealed());
}

#[test]
fn test_scenario_4_2_replay_active_fires_at_intervals() {
    // Given: Phase 2 active, replay_interval = 10, curriculum facil
    let mut trainer = build_test_trainer(
        /* replay_training_capacity */ 256,
        /* replay_interval */ 10,
        /* advance_threshold */ 0.30,
        /* window_size */ 20,
        /* max_episodes */ 100,
    );
    // When: corre 100 episodios, esperamos advance + replays post-seal
    trainer.train();
    // Then: sealed == true, invocations > 0, invocations <= 100/10 = 10
    assert!(
        trainer.training_memories_sealed(),
        "sealed debe ser true"
    );
    assert!(
        trainer.replay_invocations() > 0,
        "replay debe haberse invocado al menos una vez post-seal"
    );
    assert!(
        trainer.replay_invocations() <= 10,
        "replay_invocations ({}) excede el maximo teorico (10)",
        trainer.replay_invocations()
    );
}

#[test]
fn test_scenario_4_4_replay_deferred_before_seal() {
    // Given: Phase 2 active, replay_interval = 5, pero advance_threshold imposible
    let mut trainer = build_test_trainer(
        /* replay_training_capacity */ 256,
        /* replay_interval */ 5,
        /* advance_threshold */ 0.999,
        /* window_size */ 100,
        /* max_episodes */ 50,
    );
    // When: corre 50 episodios sin advance
    trainer.train();
    // Then: warmup B respetado — sin seal, sin replay
    assert!(!trainer.training_memories_sealed());
    assert_eq!(
        trainer.replay_invocations(),
        0,
        "replay no debe invocarse sin seal previo"
    );
}
```

- [ ] **5.2: Run Red — tests deben fallar**

```bash
cargo nextest run training::continuous::tests::test_scenario_4_1_replay_inactive_when_phase_2_off training::continuous::tests::test_scenario_4_2_replay_active_fires_at_intervals training::continuous::tests::test_scenario_4_4_replay_deferred_before_seal 2>&1 | tail -10
```
Expected:
- `test_scenario_4_1`: pasa (replay_invocations == 0 sin logica, pero la logica no existe → no se invoca → 0 es correcto por accidente). Aceptable.
- `test_scenario_4_2`: falla — `replay_invocations == 0` aunque esperamos > 0.
- `test_scenario_4_4`: pasa (sin logica, invocations == 0 por default).

Nota: 4.1 y 4.4 passing en Red es esperable — verifican el estado correcto incluso con logica ausente (invariante por defecto). Son guardas contra regresion cuando agreguemos la logica en Green.

- [ ] **5.3: Invocar `/verification-before-completion` (Red)**

Confirmar que `test_scenario_4_2` falla especificamente por `replay_invocations == 0` (no por compile error u otro bug).

- [ ] **5.4: Commit Red**

```bash
git add src/training/continuous.rs
git commit -m "test: add replay trigger scenarios 4.1, 4.2, 4.4 (Task 5 Red)"
```

### Green

- [ ] **5.5: Agregar el trigger de replay en `train()`**

Archivo: `src/training/continuous.rs`. Insertar despues de `self.episode_count += 1;` (~L162), antes del bloque de logging (L163).

```rust
            // [P2] Replay trigger interval-based (post-warmup).
            // Gate: solo corre si sealed == true (warmup B per spec §3.5).
            if self.training_memories_sealed
                && self.replay_interval > 0
                && self.episode_count.is_multiple_of(self.replay_interval)
            {
                match self.agent.replay_learn(self.replay_batch_size) {
                    Ok(()) => {
                        self.replay_invocations += 1;
                        let line = format!(
                            "[ep {}] replay_learn batch={} (invocation #{})",
                            self.episode_count, self.replay_batch_size, self.replay_invocations,
                        );
                        eprintln!("{line}");
                        self.log_lines.push(line);
                    }
                    Err(e) => {
                        let line = format!(
                            "[ep {}] replay_learn failed: {} (skipped)",
                            self.episode_count, e,
                        );
                        eprintln!("{line}");
                        self.log_lines.push(line);
                    }
                }
            }
```

- [ ] **5.6: Run Green — tests deben pasar**

```bash
cargo nextest run training::continuous::tests::test_scenario_4_1 training::continuous::tests::test_scenario_4_2 training::continuous::tests::test_scenario_4_4 2>&1 | tail -5
```
Expected: los 3 pasan.

Full suite:
```bash
cargo nextest run 2>&1 | tail -3
```
Expected: `173 tests run: 173 passed` (162 + 2 Task 1 + 2 Task 2 + 2 Task 3 + 2 Task 4 + 3 Task 5 = 173).

- [ ] **5.7: Invocar `/verification-before-completion` (Green)**

```bash
cargo nextest run 2>&1 | tail -3
cargo clippy --tests -- -D warnings 2>&1 | tail -3
cargo fmt --check
cargo build --release 2>&1 | tail -3
```

- [ ] **5.8: Commit Green**

```bash
git add src/training/continuous.rs
git commit -m "feat: fire replay_learn at intervals post-seal (Task 5 Green)"
```

### Refactor

- [ ] **5.9: Evaluar extraccion a helper privado**

Si el bloque de replay queda denso (>10 lineas), extraer a helper:

```rust
impl ContinuousTrainer {
    /// [P2] Fires `replay_learn` if gate conditions are met.
    /// Gate: sealed && replay_interval > 0 && episode_count % replay_interval == 0.
    /// Error handling: log warn + skip, no aborta training.
    fn maybe_fire_replay(&mut self) {
        if !self.training_memories_sealed
            || self.replay_interval == 0
            || !self.episode_count.is_multiple_of(self.replay_interval)
        {
            return;
        }
        match self.agent.replay_learn(self.replay_batch_size) {
            Ok(()) => {
                self.replay_invocations += 1;
                let line = format!(
                    "[ep {}] replay_learn batch={} (invocation #{})",
                    self.episode_count, self.replay_batch_size, self.replay_invocations,
                );
                eprintln!("{line}");
                self.log_lines.push(line);
            }
            Err(e) => {
                let line = format!(
                    "[ep {}] replay_learn failed: {} (skipped)",
                    self.episode_count, e,
                );
                eprintln!("{line}");
                self.log_lines.push(line);
            }
        }
    }
}
```

Y reemplazar el bloque en `train()` con `self.maybe_fire_replay();`.

Si ya se extrajo `try_seal_on_first_advance` en Task 4 Refactor, este patron es simetrico y recomendado.

- [ ] **5.10: Run Refactor checks**

```bash
cargo nextest run 2>&1 | tail -3
cargo clippy --tests -- -D warnings 2>&1 | tail -3
cargo fmt --check
```

- [ ] **5.11: Invocar `/verification-before-completion` (Refactor)**

- [ ] **5.12: Commit Refactor (si hubo cambios)**

```bash
git add src/training/continuous.rs
git commit -m "refactor: extract maybe_fire_replay helper (Task 5 Refactor)"
```

### Cierre de tarea

- [ ] **5.13: Marcar tarea completa + cerrar plan**

```bash
# Editar planning/claude-plan-tdd.md: Task 5 -> [x]
# Si es la ultima tarea, el plan queda con todas [x]
git add planning/claude-plan-tdd.md
git commit -m "chore: mark task 5 complete"
```

- [ ] **5.14: Actualizar state file a plan completo**

Per CLAUDE.local.md §2.3 "Al cerrar el plan": `.claude/session-state.json` debe reportar:
```json
{
  "current_task_id": null,
  "current_task_title": null,
  "current_phase": "done",
  ...
}
```

Esto habilita §7 finalizacion.

---

## Post-tasks: Pre-merge validation

Estos pasos NO son tareas TDD — son el gate pre-merge per CLAUDE.local.md §6.

- [ ] **PM.1: Verificar §0.1 limpio**

```bash
cargo nextest run 2>&1 | tail -3
cargo clippy --tests -- -D warnings 2>&1 | tail -3
cargo fmt --check
cargo build --release 2>&1 | tail -3
cargo doc --no-deps 2>&1 | tail -3
cargo audit 2>&1 | tail -5
```
Expected: todos verdes, sin vulnerabilidades.

- [ ] **PM.2: Verificar `git status` limpio per §7 del CLAUDE.local.md**

```bash
git status
```
Expected: working tree clean respecto al alcance del plan. Untracked permitidos solo los documentados en `CLAUDE.md` como ignorables.

- [ ] **PM.3: Loop 1 — `/requesting-code-review`**

Invocar el skill. Procesar findings `[CRITICAL]`/`[WARNING]` con `/receiving-code-review` → mini-ciclos de fix (test:/fix:/refactor:) → repetir hasta clean-to-go.

- [ ] **PM.4: Loop 2 — `/magi:magi` gate**

Solo despues de Loop 1 clean-to-go. Invocar MAGI. Veredicto minimo: `GO WITH CAVEATS`. Si HOLD o STRONG NO-GO, entrar loop de correccion (max 3 iteraciones).

- [ ] **PM.5: Checklist final §7 CLAUDE.local.md**

Verificar los 9 items del checklist manualmente:
- [ ] Todas las tareas del plan `[x]`
- [ ] `.claude/session-state.json` con `current_phase: "done"`
- [ ] §0.1 limpio
- [ ] git status limpio
- [ ] spec + plan reflejan estado final
- [ ] `/requesting-code-review` clean-to-go
- [ ] MAGI gate aprobado
- [ ] Commits siguen convencion §5
- [ ] `CLAUDE.md` actualizado si agregaron decisiones duraderas

- [ ] **PM.6: Invocar `/finishing-a-development-branch`**

El skill guia la decision entre merge directo / PR / cleanup.

---

## Post-merge: Validacion empirica (Scenario 4.9)

NO es parte del TDD — es validacion del success criterion empirico post-merge per spec §10.2.

- [ ] **EMP.1: Extender `config_seedtest_p1p2.toml` con `replay_interval = 100`**

Agregar `replay_interval = 100` en la seccion `[training]` del config.

- [ ] **EMP.2: Correr seed-test 35 × 200k**

```bash
cargo run --release -- seed-test -c config_seedtest_p1p2.toml --n 35 --continuous
mv experiment.txt Experiments/2026-04-19_seedtest_p1p2_integrated/experiment.txt
```

- [ ] **EMP.3: Comparar contra baseline p1**

Baseline: `Experiments/2026-04-18_seedtest_selfrecovery/experiment_p1.txt` (mean=6.60, D=9 reach=14%).

Hipotesis a validar:
- **Principal:** mean > 6.60
- **Secundaria:** D=9 reach > 14%
- **Fallback:** si ≤ baseline, documentar "replay no util bajo curriculum monotonico" en `Experiments/.../analysis.md`.

- [ ] **EMP.4: Documentar resultado en `Experiments/.../analysis.md`**

Mismo formato que `2026-04-18_seedtest_selfrecovery/analysis.md`. Incluir distribucion, mean/median, comparacion vs p1, hipotesis confirmada/rechazada.

---

## Grafo de dependencias entre tareas

```
Task 1 (replay_interval field)
  │
  ├─> Task 2 (validation)
  │
  └─> Task 3 (trainer fields/getters)
        │
        └─> Task 4 (seal on advance)
              │
              └─> Task 5 (replay trigger)
```

Tasks 2 y 3 son **mutuamente independientes** tras Task 1 — pueden paralelizarse si el runtime lo soporta. La cadena 3→4→5 es estrictamente serial.

`/subagent-driven-development` debe ejecutar:
- Task 1 primero (serial).
- Tasks 2 y 3 en paralelo (opcionalmente) — requiere worktrees distintos si TDD-Guard esta ON, per §3 "TDD-Guard bajo paralelismo" del CLAUDE.local.md.
- Task 4 tras Task 3.
- Task 5 tras Task 4.

Para simplicidad y sin necesidad de worktrees, ejecucion serial 1→2→3→4→5 es aceptable y minimiza riesgo.

---

## Resumen de commits esperados

Por tarea: Red + Green + (Refactor si hay cambios) + chore de cierre = 3-4 commits.

- Task 1: 3-4 commits (test:, feat:, [refactor:], chore:)
- Task 2: 3-4 commits
- Task 3: 3-4 commits
- Task 4: 3-4 commits
- Task 5: 3-4 commits

Total: **15-20 commits de implementacion**, + fix: commits de Loop 1 y Loop 2 post-review (cantidad depende de findings).

## Tests esperados

- Baseline: 162 tests passing
- Post-plan: 173 tests passing (162 + 11 nuevos):
  - Task 1: 2 (default + parse)
  - Task 2: 2 (validation reject + allow)
  - Task 3: 2 (phase2 off + on initial state)
  - Task 4: 2 (seal + sealed=false pre-advance)
  - Task 5: 3 (4.1 + 4.2 + 4.4)

Scenario 4.7 (backward compat con 162 tests originales) se verifica implicitamente: si los 162 tests originales siguen verdes, 4.7 pasa.

Scenario 4.8 (error handling) se cubre en code review post-implementacion si surge forma natural de forzarlo; si no, validacion empirica en experimentos post-merge.
