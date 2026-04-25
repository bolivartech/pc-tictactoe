# spec-behavior.md — Opcion B: Wire `replay_learn()` en ContinuousTrainer

**Feature:** Integracion del replay buffer Phase 2 en el loop de entrenamiento continuo.
**Branch objetivo:** `self-recovery` (PC-TicTacToe).
**Fecha:** 2026-04-18.
**Autor:** Julian Bolivar.
**Input de brainstorming:** `sbtdd/spec-behavior-base.md` (commit previo en este branch).
**Output objetivo:** Este archivo es input a `/writing-plans` → generara `planning/claude-plan-tdd-org.md`.

---

## 1. Objetivo

Integrar la API `replay_learn()` de `pc-rl-core` en `ContinuousTrainer` (`src/training/continuous.rs`) para que el replay buffer Phase 2 deje de ser inerte y contribuya activamente al entrenamiento bajo curriculum. La integracion debe ser configurable via TOML y medible empiricamente — el objetivo final es validar si el stack Phase 1 + Phase 2 (con replay activo) supera la performance de Phase 1 solo.

## 2. Contexto

### 2.1 Estado actual

`pc-rl-core` provee estas APIs de Phase 2 (branch `self-recovery`, post commits 1–19 del plan `2026-04-13-self-recovery.md`):

- `PcActorCritic::replay_learn(batch_size: usize) -> Result<(), PcError>` — **safe bajo empty buffer** (returns `Ok(())`); documentacion del core dice *"callers typically invoke replay_learn on a fixed cadence and should not crash on startup"*.
- `PcActorCritic::seal_replay_training_memories() -> Result<(), PcError>` — idempotente internamente (setea `training_phase = false`, sin side-effects en re-invocaciones).
- (Fuera de scope: `clear_recent_memories`, `rollback_soft`, `rollback_hard`, `champion_update` — ver §8 Exclusiones.)

El auto-recording en `step_masked()` ya funciona cuando `replay_training_capacity > 0`. Lo que falta es el **consumer** (`replay_learn`) y el **trigger de seal** en el trainer de TTT.

### 2.2 Evidencia empirica motivadora

Experimento seed-test (35 seeds × 3 configs × 200k episodios), resultados en `Experiments/2026-04-18_seedtest_selfrecovery/analysis.md`:

| Config | Mean `max_depth` | D=9 reach | D>=7 reach |
|---|---|---|---|
| base (P1/P2 off) | 5.74 | 3% | 34% |
| **p1** (distillation on) | **6.60** | **14%** | **63%** |
| p1+p2 (buffer dormant) | 6.20 | 3% | 57% |

Phase 1 alone mejora significativamente. Phase 2 con buffer **dormant** introduce regresion leve vs p1 (cola D=9 cae).

**Hipotesis a validar:** con `replay_learn()` activo, p1+p2-integrated supera a p1 en D=9 reach y mean, porque las transitions de memorias exitosas (compartment A sellado) proporcionan gradient signal positivo durante transiciones de curriculum.

## 3. Requerimientos funcionales (SDD)

### 3.1 Wire `replay_learn()` en ContinuousTrainer

**R1.** `ContinuousTrainer` debe invocar `agent.replay_learn(batch_size)` periodicamente durante el loop de entrenamiento cuando el replay buffer este configurado (`replay_training_capacity > 0`) **y** el warmup gate haya abierto (ver §3.5).

**R2.** La frecuencia de invocacion es configurable via campo TOML nuevo `[training] replay_interval: usize` (episodios entre invocaciones). Default: **100**.

**R3.** `replay_learn` se invoca sin check manual de buffer — el core ya maneja `Ok(())` silencioso para buffer vacio. **Simplificacion vs base spec:** R3 del base requeria verificar buffer total antes; eso es ahora automatico via warmup gate (§3.5) + contrato safe del core.

### 3.2 Trigger `seal_replay_training_memories()`

**R4.** Al avanzar el curriculum por primera vez (depth N → N+1 con N primera vez), invocar `agent.seal_replay_training_memories()` una sola vez. Esto congela compartment A (training memories) y abre compartment B (recent memories) para nuevas transitions.

**R5.** Idempotencia: el trainer mantiene un flag interno `training_memories_sealed: bool` que se marca `true` solo ante Ok del core. Invocaciones posteriores del seal (advances subsecuentes) son skipped por el flag. No se depende de try/catch.

### 3.3 Campo TOML nuevo

**R6.** Extender `TrainingSection` con `replay_interval: usize` (default 100).

**R7.** Validacion en `validate_cl()`:
```
Si a.replay_training_capacity > 0 && self.training.replay_interval == 0:
    ConfigError: "training.replay_interval must be > 0 when agent.replay_training_capacity > 0"
```

**R8.** Documentar el campo con rustdoc: proposito, default, trade-off (bajo = mas overhead, alto = buffer mas sesgado por FIFO eviction en compartment B), relacion con `replay_training_capacity`.

### 3.4 Observabilidad

**R9.** Al invocar `replay_learn` con Ok: log level info:
```
[ep {episode_count}] replay_learn batch={replay_batch_size} (invocation #{replay_invocations})
```

**R10.** Al invocar `replay_learn` con Err: log level warn:
```
[ep {episode_count}] replay_learn failed: {error} (skipped)
```

**R11.** Al invocar `seal_replay_training_memories` con Ok: log level info:
```
[ep {episode_count}] replay training memories sealed (curriculum advance 1→2+)
```

**R12.** Al invocar `seal_replay_training_memories` con Err: log level warn:
```
[ep {episode_count}] seal_replay_training_memories failed: {error} (will retry next advance)
```

### 3.5 Warmup gate (depth advancement)

**R13.** `replay_learn` **NO se invoca hasta que `training_memories_sealed == true`**. El seal solo se dispara al primer curriculum advancement (§3.2).

**R14.** Consecuencia: si el curriculum nunca avanza (seed desafortunado que cascadea en depth 1), replay nunca corre para ese seed. Comportamiento correcto — sin memorias "sealed" de entrenamiento exitoso, replay no tiene signal util.

### 3.6 Error handling policy

**R15.** `replay_learn` retorna `Err`: log warn (R10) + skip. **NO abortar el training**. `replay_invocations` NO se incrementa (contador solo cuenta Ok).

**R16.** `seal_replay_training_memories` retorna `Err`: log warn (R12) + `training_memories_sealed` permanece `false` → retry automatico en el proximo curriculum advancement.

**R17.** Errores de validacion de config son fatales al load — `AppConfig::validate()` retorna `Err(ConfigError)` con mensaje claro.

### 3.7 Backward compatibility

**R18.** Si la config no define `[training] replay_interval`, default a 100 episodios. Si `replay_training_capacity == 0` (Phase 2 off), el campo no tiene efecto — mantiene comportamiento existente.

**R19.** Los **162 tests actuales deben seguir pasando** sin modificar ninguno. Integracion aditiva pura sobre el trainer.

**R20.** Configs TOML existentes sin `replay_interval` (p.ej. `config_seedtest_p1.toml`, `config_phase1_regression.toml`) deben seguir funcionando sin cambio.

### 3.8 Getters para tests / diagnostico

**R21.** Exponer getters publicos en `ContinuousTrainer`:
- `pub fn replay_invocations(&self) -> usize` — counter de invocaciones exitosas de `replay_learn`.
- `pub fn training_memories_sealed(&self) -> bool` — estado del flag de seal.
- `pub fn seal_attempts(&self) -> usize` — counter de intentos de seal (Ok+Err), usado para verificar idempotencia precisa en tests (ver Scenario 4.3).
- `pub fn replay_enabled(&self) -> bool` — introspeccion del flag cached desde `config.agent.replay_training_capacity > 0`. Permite a tests verificar que construction leyo el campo correctamente.

**R22.** Estos getters son principalmente para tests. No se invocan desde el loop productivo ni afectan performance.

### 3.9 Validacion adicional TOML (MAGI Checkpoint 2 amendment)

**R23.** `validate_cl()` debe rechazar `replay_batch_size > replay_training_capacity` con mensaje:
```
"replay_batch_size ({X}) must be <= replay_training_capacity ({Y}) — cannot sample more than buffer holds"
```
Motivacion: bajo `batch_size > capacity`, `replay_learn` samplearia con `total_len < batch_size`; aunque el core retorna Ok silenciosamente, la config es semanticamente incorrecta. Fail fast al load.

## 4. Escenarios BDD (Given / When / Then)

### Scenario 4.1: Phase 2 off → replay inactivo (backward compat)

```
Given un config con `replay_training_capacity = 0`
  And `replay_interval = 100` (default)
  And el ContinuousTrainer configurado con Phase 2 off
When el trainer completa 200 episodios
Then `replay_invocations() == 0`
  And `training_memories_sealed() == false`
  And `seal_attempts() == 0`
  And los logs no contienen mensajes "replay_learn" ni "replay training memories sealed"
```

**Nota (MAGI Checkpoint 2 amendment):** la assertion previa "performance bit-a-bit identica" fue eliminada. Razon: los nuevos fields del trainer (`replay_enabled`, `replay_interval`, `replay_batch_size`, `training_memories_sealed`, `replay_invocations`, `seal_attempts`) no alteran el state del PRNG ni el flujo de `step_masked()`, pero la garantia bit-a-bit es fragil de probar y no aporta al gate de correctitud — los 4 asserts de arriba son suficientes para demostrar "replay no se dispara cuando Phase 2 off".

### Scenario 4.2: Phase 2 on + advance + replay → fires at intervals

```
Given un config con `replay_training_capacity = 256`, `replay_recent_capacity = 128`, `replay_interval = 10`
  And `advance_threshold = 0.30`, `window_size = 20` (para forzar advance temprano en tests)
  And `distillation_lambda_polyak = 0.01` (Phase 1 activo)
  And el ContinuousTrainer configurado
When el trainer completa 100 episodios
  And el curriculum advanza al menos una vez (ep ~30-50 esperado)
Then `training_memories_sealed() == true` despues del primer advance
  And `replay_invocations() > 0`
  And `replay_invocations() <= 10` (maximo teorico: 100 episodes / interval 10 = 10)
  And logs contienen al menos una linea "replay training memories sealed"
  And logs contienen al menos una linea "replay_learn batch=..."
```

### Scenario 4.3: Seal solo una vez (idempotencia)

```
Given un config con Phase 2 active, advance_threshold = 0.30, window_size = 20
When el trainer corre 200 episodios forzando multiples advances (1→2, 2→3, ...)
Then `training_memories_sealed() == true` (invariante despues del primer advance)
  And `seal_attempts() == 1` (exactamente un intento — idempotencia garantizada por el flag guard)
```

**Nota (MAGI Checkpoint 2 amendment):** la verificacion original mediante log parsing ("log aparece exactamente una vez") fue reemplazada por `seal_attempts() == 1`. El counter `seal_attempts` (ver §3.8 R21) incrementa dentro del bloque `if !sealed && replay_enabled`, por lo que el flag `sealed` tras el primer Ok garantiza que el contador no vuelva a incrementar en advances subsecuentes. Test mas fuerte y sin acoplamiento a strings de log.

### Scenario 4.4: Sin advance → sin seal → sin replay (warmup B respetado)

```
Given un config con Phase 2 active, `replay_interval = 5`
  And `advance_threshold = 0.999` (imposible de alcanzar en tests cortos)
When el trainer completa 50 episodios sin advance
Then `training_memories_sealed() == false`
  And `replay_invocations() == 0`
  And logs NO contienen "replay_learn" ni "replay training memories sealed"
```

### Scenario 4.5: Validacion TOML rechaza config invalida

```
Given un TOML con:
  [agent]
  replay_training_capacity = 1024
  [training]
  replay_interval = 0
When `AppConfig::load()` + `config.validate()`
Then retorna `Err(ConfigError)`
  And el mensaje contiene "replay_interval must be > 0"
```

### Scenario 4.6: Validacion permite interval=0 si buffer off

```
Given un TOML con:
  [agent]
  replay_training_capacity = 0
  [training]
  replay_interval = 0
When `config.validate()`
Then retorna `Ok(())` (backward compat)
```

### Scenario 4.7a: Backward compat (unit — verificable en TDD)

```
Given un AppConfig::default() o un TOML sin `replay_interval` explicito
When se carga y valida
Then `config.training.replay_interval == 100` (default implicito funciona)
  And validacion pasa sin errores
  And los 162 tests pre-existentes siguen passing (sin modificar ninguno)
```

### Scenario 4.7b: Equivalencia empirica con experimento p1 (empirico post-merge)

```
Given `config_seedtest_p1.toml` (Phase 2 disabled)
  And el trainer Opcion B-integrated (post-merge)
When seed-test 35 × 200k corre
Then la distribucion resultante es estadisticamente indistinguible del experimento p1 del 2026-04-18
  (test Mann-Whitney U, alpha=0.05, p > 0.05)
```

**Nota (MAGI Checkpoint 2 amendment):** se separo Scenario 4.7 en (a) verificable en TDD y (b) validacion empirica post-merge. El (a) no requiere correr training real — solo confirma el default y que los 162 tests previos no regresan. El (b) requiere experimento dedicado y se reporta junto a EMP.1-4 del plan.

### Scenario 4.8: replay_learn Err no aborta training

```
Given un trainer con Phase 2 active y un escenario que fuerce replay_learn Err
  (p.ej. mock/fault injection, o NaN propagation sintetico)
When el trainer continua corriendo tras un Err
Then logs contienen "replay_learn failed: ... (skipped)"
  And `replay_invocations()` no se incrementa por esa invocacion
  And el training completa los episodios restantes normalmente
```

### Scenario 4.9 (empirico post-merge): p1+p2-integrated supera a p1

```
Given `config_seedtest_p1p2.toml` extendido con `replay_interval = 100`
When `seed-test --n 35 --continuous` corre
Then la distribucion resultante supera estadisticamente a la distribucion p1 del 2026-04-18
  And mean `max_depth` > 6.60
  And D=9 reach > 14%
```

(Scenario 4.9 es el success criterion empirico post-implementacion — no se valida durante TDD, solo post-merge.)

## 5. Arquitectura e implementacion

### 5.1 Approach seleccionado: extender `ContinuousTrainer` directamente (Approach 1)

Alternativas descartadas:
- **Approach 2 (wrapper):** `ReplayEnabledContinuousTrainer` como wrapper del trainer existente. Descartado: duplica el API public, complica callers (experiments, seed-test), YAGNI.
- **Approach 3 (hook pattern):** Vec<Box<dyn EpisodeHook>>. Descartado: over-engineering para un solo use case sin evidencia futura de multiples hooks.

Approach 1 es aditivo (alineado con restriccion C2 del base), no rompe API publica, y tiene footprint minimo.

### 5.2 Cambios concretos en el codigo

**`src/training/continuous.rs`:**

Nuevos fields en `ContinuousTrainer` (6):
```rust
pub struct ContinuousTrainer {
    // ... fields existentes ...
    replay_enabled: bool,             // true sii config.agent.replay_training_capacity > 0
    replay_interval: usize,
    replay_batch_size: usize,
    training_memories_sealed: bool,
    replay_invocations: usize,
    seal_attempts: usize,             // Counter: intentos de seal (Ok+Err), para test de idempotencia (§4.3)
}
```

`replay_enabled` permite al trainer saber si Phase 2 esta activa sin referenciar la config en cada iteracion ni acoplarse a internals del agent. Se calcula una vez en `new()`.

`seal_attempts` incrementa dentro del bloque `if !sealed && replay_enabled` antes del `match agent.seal_replay_training_memories()`. Post primer Ok, el flag `sealed` previene re-entrada → counter queda en 1. Si el primer attempt retorna Err, next advance puede re-intentar (counter → 2, 3, ...) hasta lograr Ok.

Modificacion de `ContinuousTrainer::new()`:
```rust
pub fn new(agent: PcActorCritic, config: &AppConfig, stop_flag: Arc<AtomicBool>) -> Self {
    Self {
        // ... campos existentes ...
        replay_enabled: config.agent.replay_training_capacity > 0,
        replay_interval: config.training.replay_interval,
        replay_batch_size: config.agent.replay_batch_size,
        training_memories_sealed: false,
        replay_invocations: 0,
    }
}
```

Modificacion del loop en `train()` (puntos marcados `[P2]`):

```
while !stop && episode_count < max_episodes:
    run_episode()
    record_outcome()
    prev_depth = self.current_depth

    if non_loss_rate > advance_threshold && current_depth < 9:
        current_depth += 1
        minimax = MinimaxPlayer::new(current_depth)

        # [P2] seal al primer advance
        if !self.training_memories_sealed && self.replay_enabled:
            match agent.seal_replay_training_memories():
                Ok => sealed = true; log::info!(...)
                Err(e) => log::warn!(...)  # retry en proximo advance

    episode_count += 1

    # [P2] replay fire
    if self.training_memories_sealed
       && self.replay_interval > 0
       && episode_count.is_multiple_of(self.replay_interval):
        match agent.replay_learn(self.replay_batch_size):
            Ok => replay_invocations += 1; log::info!(...)
            Err(e) => log::warn!(...)  # skip, continua training

    # ... logging existente ...
```

Nuevos getters:
```rust
pub fn replay_invocations(&self) -> usize { self.replay_invocations }
pub fn training_memories_sealed(&self) -> bool { self.training_memories_sealed }
```

**`src/utils/config.rs`:**

Nuevo field en `TrainingSection`:
```rust
#[derive(Debug, Clone, Deserialize)]
pub struct TrainingSection {
    // ... existentes ...
    /// Episodios entre invocaciones de `replay_learn` cuando el replay buffer
    /// esta activo (`agent.replay_training_capacity > 0`). Default: 100.
    ///
    /// Trade-off: valores bajos → mas overhead de replay por run; valores altos
    /// → buffer mas sesgado por FIFO eviction en compartment B.
    /// Ignorado silenciosamente si `replay_training_capacity == 0`.
    #[serde(default = "default_replay_interval")]
    pub replay_interval: usize,
}

fn default_replay_interval() -> usize { 100 }
```

`impl Default for TrainingSection` — agregar `replay_interval: default_replay_interval()`.

Nueva regla en `validate_cl()` (al final del bloque Phase 2):
```rust
if a.replay_training_capacity > 0 && self.training.replay_interval == 0 {
    return Err(ConfigError {
        message: "training.replay_interval must be > 0 when agent.replay_training_capacity > 0".to_string(),
    });
}
```

### 5.3 Archivos NO modificados

- `src/training/champion.rs` — `find-champion` out-of-scope (X2).
- `src/training/stress_test.rs` — `stress-test` out-of-scope (X1).
- `src/training/experiment.rs` — consume `ContinuousTrainer`; hereda el nuevo comportamiento automaticamente.
- `src/training/trainer.rs` — trainer episodico, fuera del scope Phase 2.
- `Cargo.toml` — no nuevas dependencias.
- PC-RL-Core (cualquier archivo) — integracion vive 100% en PC-TicTacToe (C1).

### 5.4 Data flow end-to-end

```
TOML → AppConfig.training.replay_interval (+ validacion)
                      ↓
ContinuousTrainer::new(config) → guarda replay_interval + replay_batch_size + sealed=false + invocations=0

Loop episodio N:
    run_episode()
    ├─ step_masked() auto-records a buffer (pc-rl-core, ya existente)
    ↓
    record_outcome()
    ↓
    curriculum advance check
    ├─ si advance:
    │   ├─ (si !sealed && replay_enabled)
    │   │   └─ seal_replay_training_memories() → sealed = true on Ok
    │   └─ log info "sealed" on Ok, warn on Err
    ↓
    episode_count += 1
    ↓
    (si sealed && episode_count % replay_interval == 0)
    ├─ replay_learn(replay_batch_size)
    │   ├─ Ok → invocations++, log info
    │   └─ Err → log warn, skip
    ↓
    logging existente
```

## 6. Testing strategy

### 6.1 Estrategia general

Dos capas complementarias, **sin mocks de agent**:

1. **Unit tests** sobre `AppConfig` validation — rapidos, sin instanciar agent. Cubren Scenarios 4.5, 4.6, 4.7 (parcial).
2. **Integration tests** sobre `ContinuousTrainer::train()` — lentos-medios, con real `PcActorCritic`. Cubren Scenarios 4.1-4.4, 4.7, 4.8.

`PcActorCritic` no implementa trait abstractible publicamente; mocks dobles serian fragiles y no captan invariantes reales. Integration tests con agent real corren en 1-5s por test para planes cortos.

### 6.2 Tests nuevos — distribucion

**`src/utils/config.rs` (unit, 4 tests):**
- `test_replay_interval_default_is_100`
- `test_replay_interval_parses_from_toml`
- `test_validation_rejects_interval_zero_with_buffer_active`
- `test_validation_allows_interval_zero_with_buffer_off`

**`src/training/continuous.rs` (integration, 5-6 tests):**
- `test_scenario_4_1_replay_inactive_when_phase_2_off`
- `test_scenario_4_2_replay_active_fires_at_intervals`
- `test_scenario_4_3_seal_only_once_on_first_advance`
- `test_scenario_4_4_replay_deferred_before_seal`
- `test_scenario_4_7_default_interval_value_accessible`
- (Opcional) `test_replay_error_handling_skips_gracefully` — depende de si se puede forzar Err sinteticamente sin mocks.

**Total esperado:** 9-10 tests nuevos, 162 + 10 = ~172 tests passing.

### 6.3 Test helpers

Helper privado `#[cfg(test)]` en `src/training/continuous.rs`:
```rust
#[cfg(test)]
fn build_test_trainer(
    replay_training_capacity: usize,
    replay_interval: usize,
    advance_threshold: f64,
    window_size: usize,
    max_episodes: usize,
) -> ContinuousTrainer { /* ... */ }
```

Setter `#[cfg(test)]` si no existe:
```rust
#[cfg(test)]
pub(crate) fn set_max_episodes(&mut self, n: usize) {
    self.max_episodes = n;
}
```

### 6.4 Errores de replay — cobertura pragmatica

Forzar `Err` de `replay_learn` sinteticamente requiere NaN propagation o invariantes violados; dificil de construir en test. **Decision:** cubrir happy path + no-op exhaustivamente via integration tests. El error path se valida empiricamente post-merge via logs warn visibles en experimentos.

Si durante implementation surge forma natural de forzar Err (p.ej. config con batch_size > buffer capacity), agregarlo entonces. No bloquear el plan por este test.

## 7. Plan TDD — 5 tareas atomicas

Cada tarea es un ciclo Red→Green→Refactor completo, con `/verification-before-completion` al cierre de cada fase, y prefijos de commit segun §5 de `CLAUDE.local.md`.

### Tarea 1 — Campo `replay_interval` en `TrainingSection`

**Alcance:** field + default + rustdoc. Sin validacion (eso es Tarea 2).

**Red:**
- `test_replay_interval_default_is_100`
- `test_replay_interval_parses_from_toml`

**Green:** agregar field, helper `default_replay_interval()`, actualizar `impl Default`.

**Refactor:** rustdoc completo (proposito, default, trade-off, relacion con replay_training_capacity).

**Depende de:** nada.

### Tarea 2 — Validacion TOML

**Red:**
- `test_validation_rejects_interval_zero_with_buffer_active`
- `test_validation_allows_interval_zero_with_buffer_off`

**Green:** regla en `validate_cl()` al final del bloque Phase 2.

**Refactor:** mensaje de error consistente con otros.

**Depende de:** Tarea 1.

### Tarea 3 — Campos de replay en `ContinuousTrainer` + constructor + getters

**Red:**
- Test verificando estado inicial: `replay_invocations() == 0`, `training_memories_sealed() == false`, construction no rompe con Phase 2 on/off.

**Green:** agregar los 5 fields (`replay_enabled`, `replay_interval`, `replay_batch_size`, `training_memories_sealed`, `replay_invocations`) + getters (`replay_invocations`, `training_memories_sealed`) + leer values desde `AppConfig` en `new()`.

**Refactor:** rustdoc en getters (diagnostic/test purpose); rustdoc en `replay_enabled` explicando que es un cached check de `config.agent.replay_training_capacity > 0`.

**Depende de:** Tarea 1.

### Tarea 4 — Seal al primer curriculum advance

**Red:**
- `test_scenario_4_3_seal_only_once_on_first_advance`
- `test_scenario_4_4_replay_deferred_before_seal` (parcial: verifica sealed=false antes de advance)

**Green:** en el bloque curriculum advance de `train()`, agregar check `!sealed && replay_enabled` → invocar seal → marcar sealed on Ok, log.

**Refactor:** si la logica queda densa, extraer a helper privado `fn try_seal_on_first_advance(&mut self, episode_count: usize)`.

**Depende de:** Tarea 3.

### Tarea 5 — Replay trigger interval-based

**Red:**
- `test_scenario_4_1_replay_inactive_when_phase_2_off`
- `test_scenario_4_2_replay_active_fires_at_intervals`
- `test_scenario_4_4_replay_deferred_before_seal` (completo)
- (Opcional) `test_scenario_4_7_default_interval_value_accessible`

**Green:** despues de `episode_count += 1`, check `sealed && replay_interval > 0 && episode_count % replay_interval == 0` → `replay_learn(batch)` → invocations++ + log.

**Refactor:** extraer a helper `fn maybe_fire_replay(&mut self, episode_count: usize)`. Unificar pattern de logging entre seal y replay si hay duplicacion.

**Depende de:** Tarea 4 (el warmup B depende de `sealed`).

### 7.1 Grafo de dependencias

```
Tarea 1 (field) ─┬─> Tarea 2 (validation)
                 │
                 └─> Tarea 3 (struct + constructor) ─> Tarea 4 (seal) ─> Tarea 5 (replay)
```

`/subagent-driven-development` ejecutara serial (por dependencias transitivas). Paralelismo parcial posible entre Tareas 2 y 3-4-5 si el runtime lo permite, pero la cadena 3→4→5 es estrictamente serial.

### 7.2 Commits esperados

Por cada tarea × 3 fases = ~15 commits TDD + 5 commits `chore:` de cierre de tarea. Total inicial: **~20 commits**. Mas posibles `fix:` del Loop 1 de code review y Loop 2 de MAGI post-review.

## 8. Restricciones

### 8.1 Arquitectura

**C1.** Integracion vive **100% en PC-TicTacToe**. No se modifica `pc-rl-core` — la API Phase 2 ya esta shipped y es suficiente.

**C2.** Extender aditivamente `src/training/continuous.rs`. No re-escribir el loop principal.

**C3.** Cero cambios breaking a la API publica de `ContinuousTrainer`. Consumers existentes (experiments, seed-test, find-champion) siguen compilando sin modificar.

### 8.2 Testing

**C4.** TDD estricto Red-Green-Refactor. Cada cambio atomico pasa `/verification-before-completion` antes del commit.

**C5.** 162/162 tests actuales pasan sin modificar. Tests nuevos aditivos.

**C6.** Cero `cargo clippy --tests -- -D warnings` warnings. Cero `cargo fmt --check` drift.

### 8.3 Config

**C7.** El campo `replay_interval` va en `[training]` section (trainer-concern, no agent-architecture).

**C8.** Defaults backward-compat: sin `replay_interval` en TOML → default 100 → sin efecto visible cuando `replay_training_capacity == 0`.

### 8.4 Observabilidad

**C9.** Logs de replay/seal a nivel `info` (matching R9, R11); errores a nivel `warn` (matching R10, R12). No introducir nuevos archivos de output.

**C10.** `replay_invocations` y `training_memories_sealed` getters son **publicos** pero para diagnostico — no se documentan como parte de la API estable. Marcar con rustdoc `/// Primarily for testing and diagnostic purposes.`

## 9. Lo que NO debe hacer (exclusiones)

### 9.1 Fuera de scope — deferido a fases siguientes

**X1.** NO implementar fitness drift detection ni `rollback_soft`/`rollback_hard`. Requiere diseño adicional (umbral, sliding window). Opcion B se limita a replay + seal.

**X2.** NO invocar `champion_update()` en `find-champion`. Frozen anchor permanece trackeando weights iniciales; integration de champions es una fase separada.

**X3.** NO invocar `clear_recent_memories()` automaticamente. Sin fitness drift detection, no hay trigger confiable. Diferido.

**X4.** NO modificar el schema de `.claude/session-state.json` ni el contrato de artefactos §2 del `CLAUDE.local.md`. Feature es interno al trainer.

**X5.** NO extender `StressTester` ni `ChampionFinder` (directamente). `ChampionFinder` usa `ContinuousTrainer` internamente y hereda el comportamiento automaticamente (sin cambios requeridos). `StressTester` es un loop distinto fuera de scope.

### 9.2 Decisiones de diseño prohibidas

**X6.** NO convertir el loop en async ni introducir threading. Single-threaded, reproducible.

**X7.** NO hacer el trigger de replay probabilistico/random. Interval-based deterministic es predecible y facilita debugging.

**X8.** NO cambiar la signature de `replay_learn` ni los defaults de `replay_batch_size` en el core. Usar los parametros shipped.

**X9.** NO introducir un modo "dry-run" o flag de solo-logs. Replay es binary-gated por `replay_training_capacity`.

**X10.** NO abortar el training run por un `Err` de `replay_learn` o `seal_replay_training_memories`. Log warn + skip (R15, R16).

### 9.3 No-goals de este feature

**X11.** NO implementar GUI, CLI flag, o mecanismo interactivo. Todo via TOML.

**X12.** NO instrumentar con metrics externos (prometheus, statsd). Logs + post-hoc analisis de experimentos.

**X13.** NO documentar decisiones arquitectonicas en `CLAUDE.md` (local, gitignored per politica de tracking §1 del `CLAUDE.local.md`). Las decisiones de diseño duraderas van en el plan TDD (`planning/`) y este spec (`sbtdd/`).

## 10. Criterios de exito

### 10.1 Exito tecnico (TDD)

- Todos los scenarios BDD 4.1-4.7 implementados y passing.
- 162 + ~10 tests nuevos, todos verdes.
- `/verification-before-completion` limpio en cada fase TDD.
- Loop 1 code review (`/requesting-code-review`) clean-to-go.
- MAGI gate (`/magi:magi`) ≥ `GO WITH CAVEATS`.
- Scenario 4.8 (error handling) parcial — si no se puede forzar Err sinteticamente, se valida post-hoc via experimentos.

### 10.2 Exito empirico (post-merge)

Scenario 4.9 — Seed-test 35 × 200k con `config_seedtest_p1p2.toml` extendida con `replay_interval = 100`:

**Hipotesis principal** (mean superior):
- H0: `mean(p1p2_integrated) <= mean(p1)` donde `mean(p1) = 6.60`.
- H1: `mean(p1p2_integrated) > 6.60`.
- **Test estadistico:** Mann-Whitney U test one-sided sobre la distribucion de `max_depth` de los 35 seeds de p1p2_integrated vs los 35 seeds de p1 (baseline `Experiments/2026-04-18_seedtest_selfrecovery/experiment_p1.txt`).
- **Significancia:** α = 0.05.
- **Criterio de rechazo de H0:** `p-value < 0.05`.

**Hipotesis secundaria** (tail D=9 recovery):
- H0: `P(max_depth == 9 | p1p2_integrated) <= 14%`.
- H1: `P(max_depth == 9 | p1p2_integrated) > 14%`.
- **Test estadistico:** one-sided proportion test (z-test o exact binomial) sobre count(max_depth == 9) / 35.
- **Significancia:** α = 0.05.

**Fallback aceptable:** si ambas H0 se mantienen (p > 0.05 en Mann-Whitney + proportion test), documentar en `Experiments/.../analysis.md` como "replay no util bajo curriculum monotonico, relevante solo bajo stress" y ajustar decision de merge (p.ej. ship solo Phase 1 a main, dejar Phase 2 integrated como feature opcional).

**Caveat (MAGI Checkpoint 2 amendment):** el compartment A (training memories) captura transitions de **depth=1 solamente** — el seal se dispara al primer advance (D=1 → D=2), antes de que el agente experimente D=2+. Esto significa que el replay retroalimenta con memorias de depth-1 solamente, lo cual puede ser limitante para mejoras en D>=7. Si EMP falla contra H1, considerar en Fase siguiente un `seal_on_advance_depth` tunable (default 1 preserva comportamiento actual).

### 10.2.1 KNOWN GAPS en el criterio empírico (MAGI Loop 2 amendment)

Antes de interpretar los resultados de EMP.2 (seed-test 35 × 200k), considerar estos sesgos estructurales del diseño actual que pueden confundir el veredicto:

1. **Compartment A depth-1-only:** como se describe arriba, el replay nutre siempre desde experiencia de depth=1. Un resultado negativo contra H1 NO necesariamente invalida el concepto de Phase 2 self-recovery — puede indicar que `seal_on_advance_depth = 1` es demasiado temprano. Lectura honesta requiere distinguir "replay inútil bajo curriculum" vs "seal timing sub-óptimo".

2. **`replay_learn` Err path no unit-testeado:** solo el path Ok está bajo TDD. Si el Err path se activa durante el experimento (NaN propagation, invariantes internos violados), se verá en logs del experiment.txt pero no hay guarantee de behavior correcto bajo esas condiciones.

3. **Tests con threshold=0.0/window=1:** los tests de Task 4-5 usan thresholds trivialmente permisivos para garantizar advance. Esto prueba estructura (seal+replay fires) pero no realismo (advance bajo performance significativa). Comportamiento bajo thresholds reales (0.95/1000 como en configs de experimento) no está cubierto por unit tests — solo por el experimento empírico en sí.

4. **Replay actor-update gateado por hysteresis (post-experiment finding 2026-04-19):** el path `replay_learn → learn_continuous_inner → apply_actor_update_and_bookkeeping` aplica `s_scale = effective_actor_scale(surprise)` que retorna `scale_floor` cuando el actor está FROZEN. Adicionalmente, `skip_kl = is_actor_frozen() || ...` desactiva incluso la inyección del KL distillation gradient (P1 anchors) durante FROZEN. Bajo configs CL-balanced de stress (~70-72% FROZEN), **~72% de las invocaciones de `replay_learn` no modifican weights del actor** — solo el crítico se actualiza (no gateado por hysteresis hasta v3.0.0). Esto NO es un bug; es la disciplina de diseño del stack M2 + Phase 2 interactuando. Pero invalida la lectura naive de R21 ("replay reinforces successful training memories") bajo CL-balanced — la lectura correcta es "replay alinea principalmente el crítico, y solo aporta al actor durante windows PLASTIC del 28%". **Upstream fix (pc-rl-core v2.2.1, 2026-04-24):** field opt-in `scale_floor_replay: f64` agregado a `PcActorCriticConfig`. Default sentinel `-1.0` preserva v2.2.0 behavior; valor estrictamente positivo opta al replay path en actor learning bajo FROZEN, también bypassing `skip_kl` para que P1 KL anchors contribuyan. Actualmente expuesto en `[agent]` TOML de TTT (default sentinel preserva backward compat).

5. **Asimetría P1 vs P2 en frecuencia de activación efectiva (post-experiment finding 2026-04-19):** P1 distillation se activa cada `step_masked` (on-policy, ~4-5 calls por episodio TTT). P2 replay se activa cada `replay_interval` episodios (off-policy). En 500k eps con 28% PLASTIC: P1 produce ~560k oportunidades efectivas de aplicar KL gradient al actor; P2 produce ~1,400 oportunidades efectivas con `replay_interval = 100`. **P1 tiene ~400x más "dosis" de actor-modification que P2** bajo el mismo gate de hysteresis. Esto explica el ranking empírico observado (P1 previene cascade, P2+P1 ≈ P1 con marginal extra del crítico). **Upstream fix availability (pc-rl-core v3.0.0, 2026-04-25):** combinando `scale_floor_replay > 0.0` (actor opt-in) y `critic_floor_replay > 0.0` (critic opt-in, también v3.0.0) consumers pueden hacer que replay actúe como mecanismo de actor-reinforcement bajo FROZEN. Recomendación de pareo simétrico del CHANGELOG: `(scale_floor_replay=0.3, critic_floor_replay=0.3)` para mild recovery, `(1.0, 1.0)` para aggressive. Asimétrico permitido pero produce desincronización actor-critic. Si el goal es validar empíricamente esta opt-in, alternativa de menor compute es bajar `replay_interval` a ~4-10 (sin cambio de código).

6. **Critic hysteresis era state-tracker pre-v3.0.0 (post-experiment finding 2026-04-19, upstream fix 2026-04-25):** prior to pc-rl-core v3.0.0, `critic_hysteresis = true` trackeaba un state machine pero el state nunca se consultaba en el weight-update site del crítico (asimetría con actor, gateado desde v2.0.0). El crítico aprendía únicamente por magnitude del TD error vía `critic_surprise_scale`. Confirmado experimentalmente en el champion-stress de 2026-04-19: la mejor performance "critic FROZEN %" (76.8% en P1+P2) NO bloqueaba updates del crítico. **Upstream fix v3.0.0 (BREAKING):** `effective_critic_scale_for_mode` ahora consulta `critic_hysteresis.state == Frozen` y aplica clamp a `scale_floor` (online) o `critic_floor_replay` (replay). Para preservar comportamiento empírico de v2.2.x ("critic always learns under FROZEN"), set `critic_floor_replay = scale_ceil` en configs de stress.

**Acción para EMP analysis:** al reportar resultados, explicitar estos 6 gaps en la sección de caveats del `analysis.md` post-experimento. **Notar versión de pc-rl-core usada** — los gaps 4, 5, 6 tienen comportamiento distinto en v2.2.0 vs v2.2.1 vs v3.0.0.

### 10.3 Exito de proceso

- Ningun commit rompe la secuencia TDD (verificado por TDD-Guard + `/verification-before-completion`).
- Ningun archivo fuera del alcance del plan en el diff final (verificado por §7 git status limpio del `CLAUDE.local.md`).
- Plan ejecutado de punta a punta sin intervencion manual adicional (verificado por `/finishing-a-development-branch` clean run).
- State file (`.claude/session-state.json`) reporta `current_phase: "done"` al finalizar.

## 11. Referencias

- `sbtdd/spec-behavior-base.md` — input pre-brainstorming (commit previo en este branch).
- `CLAUDE.local.md` — metodologia, protocolos, checklists (este proyecto).
- `Experiments/2026-04-18_seedtest_selfrecovery/analysis.md` — baseline empirico motivador.
- `D:/jbolivarg/RustProjects/PC-RL-Core/CHANGELOG.md` — documentacion Phase 2 API.
- `D:/jbolivarg/RustProjects/PC-RL-Core/docs/superpowers/plans/2026-04-13-self-recovery.md` — plan de PC-RL-Core que ship'o Phase 2.
- `src/training/continuous.rs` §"Phase 2 self-recovery orchestration" TODO (L5-25) — gap que este feature cierra.
