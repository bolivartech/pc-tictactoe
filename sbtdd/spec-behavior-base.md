# spec-behavior-base.md — Opcion B: Wire `replay_learn()` en ContinuousTrainer

**Feature:** Integracion del replay buffer Phase 2 en el loop de entrenamiento continuo.
**Branch objetivo:** `self-recovery` (PC-TicTacToe).
**Fecha:** 2026-04-18.
**Autor:** Julian Bolivar.
**Input a:** `/brainstorming` → generara `sbtdd/spec-behavior.md`.

---

## 1. Objetivo

Integrar la API `replay_learn()` de `pc-rl-core` en `ContinuousTrainer` (`src/training/continuous.rs`) para que el replay buffer Phase 2 deje de ser inerte y contribuya activamente al entrenamiento bajo curriculum. La integracion debe ser configurable via TOML y medible empiricamente — el objetivo final es validar si el stack Phase 1 + Phase 2 (con replay activo) supera la performance de Phase 1 solo.

## 2. Contexto

### 2.1 Estado actual

`pc-rl-core` (branch `self-recovery`, post commits 1–19 del plan `2026-04-13-self-recovery.md`) provee estas APIs de Phase 2:

- `PcActorCritic::replay_learn(batch_size: usize) -> Result<(), PcError>`
- `PcActorCritic::seal_replay_training_memories() -> Result<(), PcError>`
- `PcActorCritic::clear_recent_memories() -> Result<(), PcError>`
- `PcActorCritic::rollback_soft() -> Result<(), PcError>` (requiere `distillation_lambda_polyak > 0`)
- `PcActorCritic::rollback_hard() -> Result<(), PcError>` (requiere `distillation_lambda_frozen > 0`, con cooldown ~100-200 steps)
- `PcActorCritic::champion_update() -> Result<(), PcError>` (actualiza Frozen anchor)

El auto-recording en `step_masked()` ya funciona cuando `replay_training_capacity > 0` — el buffer se llena solo. **Lo que falta es wire el consumer (`replay_learn`) y los triggers de rollback / seal / clear / champion_update en el trainer de TTT.**

### 2.2 Evidencia empirica motivadora

Experimento seed-test (35 seeds × 3 configs × 200k episodios) sobre TTT — resultados en `Experiments/2026-04-18_seedtest_selfrecovery/`:

| Config | Mean `max_depth` | D=9 reach | D>=7 reach |
|---|---|---|---|
| base (P1/P2 off) | 5.74 | 3% | 34% |
| **p1** (distillation on) | **6.60** | **14%** | **63%** |
| p1+p2 (buffer dormant) | 6.20 | 3% | 57% |

Phase 1 alone mejora significativamente vs base. Phase 2 **con buffer dormant** introduce regresion leve vs p1 (cae cola D=9) — el buffer auto-records pero nunca se consume; el overhead reduce varianza positiva.

La hipotesis a validar en Opcion B es: **con `replay_learn()` activo, p1+p2-integrated supera a p1 en D=9 reach y mean**, porque los transitions de memorias exitosas (compartment A sellado) proporcionan gradient signal positivo durante transiciones de curriculum, reduciendo forgetting.

Si la hipotesis falla — resultado honesto: Phase 2 no es util bajo curriculum monotonico; solo es util bajo stress (cascadas). Esto seria input valioso para decidir merge parcial (ship solo Phase 1 a `main`).

## 3. Requerimientos funcionales (SDD)

### 3.1 Wire `replay_learn()` en ContinuousTrainer

**R1.** `ContinuousTrainer` debe invocar `agent.replay_learn(batch_size)` periodicamente durante el loop de entrenamiento cuando el replay buffer este configurado (`replay_training_capacity > 0`).

**R2.** La frecuencia de invocacion es configurable via un nuevo campo TOML `[training] replay_interval: usize` (episodios entre invocaciones). Default razonable: 100 episodios.

**R3.** `replay_learn` NO se invoca si el buffer total esta vacio (first batch requires data) — el trainer debe verificar `agent.replay_buffer_total() >= min_batch_size` antes de llamar.

**R4.** Precondiciones de seguridad:
- Si `replay_training_capacity == 0`, el trainer NO invoca `replay_learn` (no-op gracefully).
- Si el agente aun esta en curriculum temprano (depth baja), replay puede ser prematuro — diferir hasta que el agente acumule X episodios exitosos (criterio a definir en brainstorming).

### 3.2 Trigger `seal_replay_training_memories()`

**R5.** Al avanzar el curriculum por primera vez (depth N → N+1 con N primera vez), invocar `agent.seal_replay_training_memories()` una sola vez. Esto congela el compartment A (training memories) y abre compartment B (recent memories) para nuevas transitions.

**R6.** Idempotencia: invocaciones subsecuentes de `seal_replay_training_memories()` deben ser no-op (ya sellado). El trainer debe trackear si ya invoco el seal via state interno o invocando siempre con try/catch.

### 3.3 Campo TOML nuevo

**R7.** Extender `TrainingSection` con `replay_interval: usize` (default 100). Validacion: si `replay_training_capacity > 0 && replay_interval == 0`, validation error ("replay_interval must be > 0 when replay buffer is active").

**R8.** Documentar el campo en `src/utils/config.rs` con rustdoc explicando la cadencia y el trade-off (bajo = mas overhead, alto = buffer mas sesgado por FIFO eviction en compartment B).

### 3.4 Observabilidad

**R9.** Al invocar `replay_learn`, el trainer debe loggear (nivel info) un mensaje tipo `[ep N] replay_learn batch=B buffer_total=T` para que los logs de entrenamiento reflejen la actividad de replay.

**R10.** Al invocar `seal_replay_training_memories`, log info tipo `[ep N] replay training memories sealed (compartment A: K entries)`.

### 3.5 Backward compatibility

**R11.** Si la config no define `[training] replay_interval`, default a 100 episodes. Si `replay_training_capacity == 0` (phase 2 off), el nuevo campo no tiene efecto — mantiene comportamiento existente.

**R12.** Los 162 tests actuales deben seguir pasando sin modificar ninguno (aditivo puro sobre el trainer).

## 4. Escenarios BDD (Given / When / Then)

### Scenario 4.1: replay inactivo cuando Phase 2 off

```
Given un config con `replay_training_capacity = 0`
 And el ContinuousTrainer corriendo 200 episodios
When el trainer completa los 200 episodios
Then `replay_learn` NO fue invocado ni una sola vez
 And los logs no contienen mensajes de "replay_learn"
 And la performance final es bit-a-bit identica a la del trainer pre-Phase 2
```

### Scenario 4.2: replay activo con buffer lleno

```
Given un config con `replay_training_capacity = 2048`, `replay_interval = 100`
 And el ContinuousTrainer con el buffer auto-recording activo
When el trainer completa 500 episodios
 And el buffer tiene >= batch_size transitions
Then `replay_learn` fue invocado exactamente N veces donde N = floor(500 / 100)
 And cada invocacion genero un log info con formato esperado
```

### Scenario 4.3: seal al primer curriculum advancement

```
Given un config con Phase 2 activo
 And el agente en depth = 1 (curriculum inicial)
When el agente avanza de depth 1 -> 2 por primera vez
Then `seal_replay_training_memories` fue invocado exactamente 1 vez
 And logs contienen "replay training memories sealed"
When el agente avanza de depth 2 -> 3
Then `seal_replay_training_memories` NO fue invocado de nuevo (idempotencia)
```

### Scenario 4.4: replay respeta buffer vacio

```
Given un config con Phase 2 activo, replay_interval = 10
 And el ContinuousTrainer en los primeros 30 episodios (buffer aun acumulando)
When el contador de episodios llega a multiplo de 10 (ep 10, 20, 30)
 And el buffer_total < min_batch_size
Then `replay_learn` NO se invoca (se difiere hasta que haya suficiente data)
 And el log info indica "replay skipped — buffer insufficient"
```

### Scenario 4.5: validation config

```
Given un TOML con `replay_training_capacity = 1024` y `replay_interval = 0`
When se intenta cargar el config via AppConfig::load
Then `ConfigError` con mensaje "replay_interval must be > 0 when replay buffer is active"
```

### Scenario 4.6: backward compat con configs existentes

```
Given `config_seedtest_p1.toml` (Phase 2 disabled, sin replay_interval definido)
When el trainer corre
Then todas las 162 tests pre-existentes siguen passing
 And performance es identica a la del experimento p1 del 2026-04-18
```

### Scenario 4.7: re-experiment p1+p2 integrated

```
Given `config_seedtest_p1p2.toml` extendido con `replay_interval = 100`
When seed-test --n 35 --continuous corre
Then la distribucion resultante supera estadisticamente a la distribucion p1 del 2026-04-18
 And mean `max_depth` > 6.60
 And D=9 reach > 14%
```

(Scenario 4.7 es el success criterion empirico de Opcion B — se valida post-implementacion, no durante TDD.)

## 5. Restricciones

### 5.1 Arquitectura

**C1.** La integracion debe vivir en **PC-TicTacToe solamente**. No se modifica `pc-rl-core` — la API Phase 2 ya esta shipped y es sufficient.

**C2.** El trainer existente (`src/training/continuous.rs`) se extiende de forma aditiva. No se re-escribe el loop principal.

**C3.** Cero cambios breaking a la API publica de `ContinuousTrainer`. Los consumers existentes (experiments, seed-test, find-champion si aplica) siguen compilando.

### 5.2 Testing

**C4.** Estricto TDD Red-Green-Refactor (ver §2-3 del CLAUDE.local.md). Cada cambio atomico pasa por `/verification-before-completion`.

**C5.** 162/162 tests actuales pasan sin modificar. Tests nuevos para los scenarios 4.1-4.6 — 4.7 se valida empiricamente post-merge.

**C6.** Cero `cargo clippy` warnings bajo `-D warnings`. Cero `cargo fmt` drift.

### 5.3 Config

**C7.** El nuevo campo `replay_interval` va en `[training]` section (no en `[agent]`), porque es trainer-concern, no agent-architecture.

**C8.** Defaults backward-compat: sin `replay_interval` en TOML → default 100 → sin efecto visible cuando `replay_training_capacity == 0`.

### 5.4 Observabilidad

**C9.** Logs de replay a nivel `info` (no `debug`), para que aparezcan en los experimentos sin necesidad de aumentar verbosidad.

**C10.** No introducir nuevos archivos de output o CSV — los logs existentes son suficientes para analisis.

## 6. Lo que NO debe hacer

### 6.1 Fuera de scope — a deferir para fases siguientes

**X1.** NO implementar fitness drift detection ni `rollback_soft`/`rollback_hard` en este ciclo. Esos requieren diseño adicional (umbral de drift, sliding window) que se explorara en una fase posterior. Opcion B se limita a replay + seal.

**X2.** NO invocar `champion_update()` en `find-champion` en este ciclo. El Frozen anchor permanecera tracking weights iniciales en esta iteracion; el downstream de champions es un integration point separado.

**X3.** NO invocar `clear_recent_memories()` automaticamente. Sin fitness drift detection, no hay trigger confiable. Diferido.

**X4.** NO modificar el schema de `.claude/session-state.json` ni el contrato de artefactos §2 del CLAUDE.local.md. El feature es interno al trainer.

### 6.2 Decisiones de diseño prohibidas

**X5.** NO convertir el loop de training en async ni introducir threading. Single-threaded por simplicidad + reproducibilidad.

**X6.** NO hacer el trigger de replay probabilistico (random sampling). Interval-based es deterministic y predecible — random introduce varianza adicional que complica debugging.

**X7.** NO cambiar la signature de `replay_learn` ni los defaults de `replay_batch_size`. Usar los parametros shipped por pc-rl-core.

**X8.** NO introducir un modo "dry-run" o flag de solo-logs. El replay es un feature binary-gated por `replay_training_capacity`.

### 6.3 No-goals de este feature

**X9.** NO se va a implementar una GUI, CLI flag, o mecanismo interactivo. Todo via TOML.

**X10.** NO se va a instrumentar con metrics externos (prometheus, statsd). Logs + post-hoc analisis de experimentos.

## 7. Criterios de exito

### 7.1 Exito tecnico (TDD)

- Todos los scenarios BDD 4.1-4.6 implementados y passing.
- 162 + N nuevos tests, todos verdes.
- `/verification-before-completion` limpio en cada fase.
- Code review Loop 1 clean-to-go.
- MAGI gate >= `GO WITH CAVEATS`.

### 7.2 Exito empirico (post-merge)

Seed-test 35 × 200k con config `config_seedtest_p1p2.toml` extendida con `replay_interval = 100`:

- **Hipotesis principal:** mean `max_depth` > 6.60 (supera p1 alone).
- **Hipotesis secundaria:** D=9 reach > 14% (recupera la cola que p1+p2 dormant perdio).
- **Test de falla aceptable:** si p1+p2-integrated <= p1 estadisticamente, documentar como "replay no util bajo curriculum monotonico, relevante solo bajo stress" e informar decision de merge.

### 7.3 Exito de proceso

- Ningun commit rompe la sequencia TDD (verificado por TDD-Guard + `/verification-before-completion`).
- Ningun archivo fuera del alcance del plan en el diff final (verificado por §7 git status limpio).
- Plan ejecutado de punta a punta sin intervencion manual adicional del usuario (verificado por `/finishing-a-development-branch` clean run).

## 8. Referencias

- `CLAUDE.local.md` (este proyecto) — metodologia, protocolos, checklists.
- `Experiments/2026-04-18_seedtest_selfrecovery/analysis.md` — baseline empirico.
- `D:/jbolivarg/RustProjects/PC-RL-Core/CHANGELOG.md` — documentacion de Phase 2 API.
- `D:/jbolivarg/RustProjects/PC-RL-Core/docs/superpowers/plans/2026-04-13-self-recovery.md` — plan de PC-RL-Core que ship'o Phase 2.
- `src/training/continuous.rs` §"Phase 2 self-recovery orchestration" TODO — el gap documentado que este feature cierra.
