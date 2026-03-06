# Pallas

Pallas is a C++ war-game simulation platform for training neural decision models, executing model-vs-model battles, observing the full theater state in a browser, and capturing replay artifacts for later inspection. The codebase is structured so that the same decision interface is used across offline dataset generation, training, live simulation, tournament evaluation, and replay analysis.

Countries can carry explicit ground, air, and naval unit classes, supply posture, treaty state, trade exposure, internal political pressure, escalation dynamics, intelligence confidence, and second-strike capability. The documentation, sample configs, and web UI in this repository now match that runtime.

## What Pallas Does

Pallas combines four workflows in one repository:

1. Synthetic battle dataset generation for supervised experimentation.
2. Neural network training for a fixed battle policy interface.
3. Live battle execution with HTTP control and browser observability.
4. Replay capture, replay reading, and round-robin tournament evaluation.

This makes the project useful both as an experimentation harness and as an operational sandbox for debugging policy behavior under complex geopolitical state.

## Canonical Battle Interface

he battle policy interface is defined in `include/battle_common.h` and is the single source of truth for model compatibility.

Current dimensions:

1. Input dimension: `90`
2. Output dimension: `26`

The `90` input features encode force posture, readiness, logistics, economy, resources, trade, treaty exposure, internal politics, nuclear posture, intelligence confidence, and strategic depth. The `26` output actions are:

1. `attack`
2. `defend`
3. `negotiate`
4. `surrender`
5. `transfer_weapons`
6. `focus_economy`
7. `develop_technology`
8. `form_alliance`
9. `betray`
10. `cyber_operation`
11. `sign_trade_agreement`
12. `cancel_trade_agreement`
13. `impose_embargo`
14. `invest_in_resource_extraction`
15. `reduce_military_upkeep`
16. `suppress_dissent`\
]'
17. `hold_elections`
18. `coup_attempt`
19. `propose_defense_pact`
20. `propose_non_aggression`
21. `break_treaty`
22. `request_intel`
23. `deploy_units`
24. `tactical_nuke`
25. `strategic_nuke`
26. `cyber_attack`

If a model state file is not `90 x 26`, the battle runtime will reject it.

## Binaries

The default build produces four first-class executables.

### `pallas`

Primary training and data pipeline binary.

Responsibilities:

1. Regenerate `battle_train.json`, `battle_train.txt`, and `vocab.txt`.
2. Load `model_config.json` and `train_config.json`.
\. Train or resume compatible model states.
4. Inspect serialized model state metadata and dimensions.

### `pallas_battle`

Live battle server and HTTP control surface.

Responsibilities:

1. Load scenario state and optional configured model states.
2. Run turn-based or continuous battles.
3. Expose `/api/*` state, diagnostics, and control endpoints.
4. Serve the browser UI from `web/`.
5. Write replay frames.
6. Optionally coordinate distributed decision exchange.

### `pallas_replay`

Replay inspection utility.

Responsibilities:

1. Read replay logs emitted by `pallas_battle`.
2. Print per-tick country state summaries and decisions.
3.\
]"


### `pallas_tests`

Repository test binary.

Responsibilities:

1. Model serialization and shape checks.
2. Config validation.
3. Replay round-trip coverage.
4. Deterministic simulation behavior checks.

## Data Directory

The `data/` directory contains both hand-maintained configuration and generated artifacts.

### Hand-Maintained Files

1. `data/model_config.json`
  Network topology and inference-time compatible architectural settings.
2. `data/train_config.json`
  Training schedule, optimizer settings, validation split, and class-weight usage.
3. `data/scenario_example.json`
  Sample battle scenario using the current runtime schema.
4. `data/class_weights.txt`
\]Per-action loss weights for all `26` battle actions.

### Generated Files

1. `data/battle_train.json`
  Synthetic dataset in JSON form.
2. `data/battle_train.txt`
  Flat text projection of the same dataset for tooling convenience.
3. `data/vocab.txt`
  Feature and action vocabulary aligned with the current `90/26` interface.

### Model Zoo

`data/model_zoo/` is used for serialized model artifacts and manifest entries written during training.

## Configuration Schema

### Model Configuration

`data/model_config.json` supports the following fields:

1. `hidden_layers`
2. `activation`
3. `norm`
4. `use_dropout`
\
. `dropout_prob`
6. `leaky_relu_alpha`

Supported activations:

1. `relu`
2. `sigmoid`
3. `tanh`
4. `leaky_relu`

Supported normalization modes:

1. `layernorm`
2. `batchnorm`
3. `none`

### Training Configuration

`data/train_config.json` supports the following fields:

1. `epochs`
2. `batch_size`
3. `validation_split`
4. `early_stopping_patience`
5. `base_lr`
\]
optimizer`
7. `weight_decay`
8. `adam_beta1`
9. `adam_beta2`
10. `adam_epsilon`
11. `scheduler`
12. `step_size`
13. `gamma`
14. `min_lr`
15. `label_smoothing`
16. `use_class_weights`
17. `class_weights_path`

Supported optimizers:

1. `adam`
2. `sgd`

Supported schedulers:

1. `step`
2. `exponential`
3. `cosine`

### Scenario Configuration
\]=
\a/scenario_example.json` now uses only fields that are actually consumed by the loader and world builder.

Top-level supported fields:

1. `seed`
2. `tick_seconds`
3. `ticks_per_match`
4. `map.width`
5. `map.height`
6. `map.cells`
7. `models[]`
8. `countries[]`

Supported country fields include:

1. Identity: `id`, `name`, `color`, `team`, `controller`
2. Population and legacy force totals: `population`, `army`, `navy`, `air_force`, `missiles`
3. Explicit force classes: `units_infantry`, `units_armor`, `units_artillery`, `units_air_fighter`, `units_air_bomber`, `units_naval_surface`, `units_naval_submarine`
4. State metrics: `economic_stability`, `civilian_morale`, `logistics_capacity`, `intelligence_level`, `industrial_output`, `technology_level`, `resource_reserve`
5. Sustainment and posture: `supply_level`, `supply_capacity`, `reputation`, `escalation_level`, `second_strike_capable`, `diplomatic_stance`
6. Graph structure: `adjacent`, `alliances`, `defense_pacts`, `non_aggression_pacts`, `trade_treaties`
7. Intelligence priors: `intel_on_enemy`

Fields such as `terrain`, `weather_severity`, `technology_tree`, or custom political blocks are not parsed by the scenario loader and should not be relied on in scenario JSON.

## Build
\]# Prerequisites

1. CMake `3.10+`
2. A C++17-capable compiler
3. OpenMP
4. Linux or another POSIX-like environment

### Configure And Compile

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Repository memory for this workspace uses the same build command:

```bash
cmake --build build -j$(nproc)
```

## Training And Dataset Generation

### Rebuild Dataset Artifacts

```bash
cd build
[;\0-9i97]
```

This regenerates:

1. `data/battle_train.json`
2. `data/battle_train.txt`
3. `data/vocab.txt`

### Resume Training From A Saved State

```bash
cd build
./pallas --train-only --resume ../data/best_state.bin
```

### Inspect A Serialized Model State

```bash
cd build
./pallas --inspect-model ../data/best_state.bin
```

Inspection reports:

1. Input dimension
2. Output dimension
3. Battle compatibility against the canonical `90 x 26` interface

## Running The Battle Server

```bash
cd build
./pallas_battle \
  --port 8080 \
  --web-root ../web \
  --replay ../logs/battle_replay.bin \
  --scenario ../data/scenario_example.json \
  --distributed-node-id 0 \
  --distributed-total-nodes 1 \
  --distributed-bind-host 0.0.0.0 \
  --distributed-bind-port 19090 \
  --distributed-timeout-ms 40
```

Then open `http://127.0.0.1:8080`.

### Tournament Mode

`pallas_battle` can also run a round-robin tournament without the browser server loop:

```bash
cd build
./pallas_battle \
  --tournament \
  --tournament-rounds 3 \
  --tournament-output ../logs/tournament_results.json \
  --scenario ../data/scenario_example.json
```

## Browser UI

The `web/` folder serves a control and observability console for the current runtime.

Current UI responsibilities:

1. Live state polling from `/api/state`, `/api/leaderboard`, `/api/diagnostics`, and `/api/models`
2. Turn-based and continuous battle controls
3. Country-level model upload and readiness inspection
4. Manual override submission across the full current action surface
5. Visual attack effects, result modal, and per-country tooltip detail
6. Strategic climate summaries for trade, pacts, escalation, and supply

## HTTP API

All endpoints below are served by `pallas_battle`.

### Read Endpoints

1. `GET /api/state`
  Returns the full battle snapshot including countries, map, decisions, battle timers, distributed config, competition status, and model-load errors.
2. `GET /api/leaderboard`
  Returns ranked model rows and the currently leading country metadata.
3. `GET /api/models`
  Returns country slot readiness, uploaded model inventory, and model-load failures.
4. `GET /api/diagnostics`
  Returns distributed exchange counters and battle-active state.

### Control Endpoints

1. `POST /api/control/step`
2. `POST /api/control/start`
3. `POST /api/control/pause`
4. `POST /api/control/end`
5. `POST /api/control/reset`
6. `POST /api/control/speed?ticks_per_second=<value>`
7. `POST /api/control/duration?seconds=<value>`
8. `POST /api/control/duration?min_seconds=<min>&max_seconds=<max>`
9. `POST /api/control/override?...`

The manual override surface now recognizes all current action names supported by the runtime action parser, including treaty, trade, intelligence, deployment, and nuclear branches.

### Model Upload Endpoint

1. `POST /api/upload-model?country_id=<id>&label=<name>`

The upload body is the raw `.bin` state payload. The runtime validates the serialized model state before applying it to the selected country slot.

## Distributed Runtime

Distributed mode partitions decision ownership by country modulo `total_nodes` and exchanges serialized decision envelopes over UDP.

Relevant CLI flags:

1. `--distributed-node-id`
2. `--distributed-total-nodes`
3. `--distributed-bind-host`
4. `--distributed-bind-port`
5. `--distributed-timeout-ms`
6. `--distributed-peers host:port,host:port,...`

Distributed diagnostics expose:

1. `exchange_count`
2. `packets_sent`
3. `packets_received`
4. `packets_dropped`
5. `peer_count`

## Replay Logging And Compatibility

Replay logging is enabled by `--replay <path>` when running `pallas_battle`.

Current replay characteristics:

1. Replay format version `6`
2. Chunked storage for large logs
3. Lightweight RLE compression of chunk payloads
4. Compatibility handling for legacy versions `2` through `6`
5. Replay frames now include richer unit classes, trade state, pact state, escalation, coup risk, and strategic depth

Inspect a replay with:

```bash
cd build
./pallas_replay --log ../logs/battle_replay.bin --ticks 40
```

## Tests

Build and run the test suite:

```bash
cmake --build build -j"$(nproc)"
./build/pallas_tests
```

Or use CTest:

```bash
ctest --test-dir build --output-on-failure
```

## Troubleshooting

### `battle blocked: every country must have a loaded model (.bin)`

Cause:
One or more country slots do not currently have a valid model state.

Resolution:
Upload a compatible `.bin` model for every country shown in the readiness panel.

### `target_country_id is required for this strategy`

Cause:
A manual override action that operates on another state was submitted without a target.

Resolution:
Use a valid country id shown in the UI tooltip or `GET /api/state` payload.

### `invalid battle model architecture ... expected 90x26`

Cause:
The uploaded or configured model was trained for an older action space.

Resolution:
Regenerate dataset artifacts, retrain against the current interface, and inspect the saved state with `./pallas --inspect-model`.

### Static UI returns `not found`

Cause:
`--web-root` points at the wrong directory.

Resolution:
When running from `build/`, use `--web-root ../web`.

### Distributed counters stay at zero

Cause:
You are in single-node mode, or peers are not reachable.

Resolution:
Verify `total_nodes > 1`, peer endpoints, bind ports, and UDP reachability.

## Repository Layout

1. `include/`
  Public headers for model, runtime, training, scenario loading, metrics, and simulation code.
2. `src/`
  Implementations for the data pipeline, neural model, battle runtime, HTTP server, replay tooling, and tournament execution.
3. `data/`
  Configuration plus generated training artifacts.
4. `web/`
  Browser UI assets served directly by the battle server.
5. `plugins/`
  Optional plugin source.
6. `tests/`
  Test coverage for the model and runtime.
7. `logs/`
  Replays, uploads, and operational logs.

## Suggested Workflow

1. Rebuild the dataset and vocabulary when the battle interface changes.
2. Train or resume compatible `90 x 26` model states.
3. Launch `pallas_battle` with `data/scenario_example.json`.
4. Upload or assign model binaries per country.
5. Observe behavior in the Warroom UI and capture a replay.
6. Inspect the replay or run tournament mode for broader evaluation.
