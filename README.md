# Pallas

**Pallas** is a geopolitical strategy simulation and model-training framework built in C++ for users who want more than a toy battlefield and more than a black-box AI demo. It is designed as a full operational stack where deterministic simulation, policy learning, live command-and-control visibility, and reproducible evaluation all work together in one codebase.

At its core, Pallas models a world of AI-governed countries competing, negotiating, escalating, recovering, and adapting over time. Every simulation tick is meaningful: military posture shifts, supply and logistics constraints propagate, diplomatic treaties form and break, strategic pressure changes, and model decisions produce visible consequences in both state trajectories and leaderboard outcomes.

The project is intentionally engineered for serious iteration loops. You can train models with configurable architecture and optimization controls, deploy those models directly into a live battle runtime, hot-swap uploaded `.bin` weights per country, inspect world state and diagnostics through REST APIs and the Warroom UI, and then replay outcomes frame by frame to understand exactly what happened and why.

Pallas is built around deterministic fixed-point world math so experiments are repeatable and debuggable. This makes it practical for benchmarking, regression validation, tournament-style ranking, and scenario-bank evaluation, rather than relying on unstable one-off runs. In other words, Pallas is not just about generating interesting battles; it is about producing reliable strategic AI workflows you can measure, compare, and improve.

If your goal is to research model behavior under pressure, run controlled strategic simulations, or build a robust training-to-deployment pipeline for geopolitical action policies, Pallas gives you a complete foundation: simulation engine, model runtime, web observability layer, replay tooling, benchmark harnesses, and distributed execution support.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Building](#building)
- [Project Structure](#project-structure)
- [Binaries and Usage](#binaries-and-usage)
  - [pallas - Training](#pallas---training)
  - [pallas_battle - Battle Server](#pallas_battle---battle-server)
  - [pallas_replay - Replay Reader](#pallas_replay---replay-reader)
  - [pallas_tests - Unit Tests](#pallas_tests---unit-tests)
- [Web Interface](#web-interface)
- [REST API](#rest-api)
- [Scenario Configuration](#scenario-configuration)
- [Model Configuration](#model-configuration)
- [Training Configuration](#training-configuration)
- [Class Weights](#class-weights)
- [Model Zoo](#model-zoo)
- [Distributed Mode](#distributed-mode)
- [Tournament Mode](#tournament-mode)
- [Plugin System](#plugin-system)
- [Data Formats](#data-formats)
- [Environment Variables](#environment-variables)
- [License](#license)

---

## Overview

Pallas simulates a world of nation-states where each country can be controlled by an AI model. On each simulation tick, the runtime gathers world/country features, asks the active model to produce a strategic action, coordinates diplomatic/military interactions, applies outcomes, and advances simulation state.

The project supports two major loops:

- **Training loop (`pallas`)**: generate or load battle data, train policy/value heads, validate, checkpoint, and benchmark.
- **Battle loop (`pallas_battle`)**: run live battles (turn-based and continuous), upload model binaries at runtime, inspect outcomes via browser UI/API, and record replay logs.

---

## Features

- **Deterministic world engine** with fixed-point arithmetic for stable reproducibility.
- **26 strategy actions** spanning diplomacy, military, cyber, trade, internal politics, and escalation.
- **Battle readiness gating** that ensures each country has a loaded `.bin` model before running.
- **Live web Warroom** served directly by the embedded HTTP server (`web/index.html`, `web/app.js`, `web/style.css`).
- **Model hot-swap uploads** per model/team/country via `/api/upload-model`.
- **Replay logging** (`battle_replay.bin`) for post-run analysis with `pallas_replay`.
- **Distributed battle mode** with UDP decision exchange between nodes.
- **Scenario-bank benchmark support** with JSON + TSV reports.
- **Round-robin tournament mode** with persisted leaderboard output.
- **Plugin support** for runtime strategy tooling (`plugins/default_tools.cpp`).

---

## Architecture

```text
pallas_core (static library)
  |- model / tensor / linear
  |- simulation_engine
  |- scenario_config
  |- battle_runtime
  |- battle_server
  |- data_pipeline / dataloader
  |- tournament / tool_registry

Executables
  |- pallas         (trainer + benchmark runner)
  |- pallas_battle  (battle runtime + HTTP + web assets)
  |- pallas_replay  (replay reader)
  |- pallas_tests   (unit tests)
```

High-level flow:

1. Load model/train/scenario configs.
2. Build world and model manager.
3. Run ticks: gather decisions, coordinate, apply outcomes.
4. Expose state via API and UI (battle runtime) or optimize weights (training runtime).
5. Persist artifacts: model states, logs, replay, benchmark outputs.

---

## Prerequisites

| Dependency | Minimum | Notes |
|---|---|---|
| CMake | 3.10+ | Build generation |
| C++ Compiler | C++17 | GCC/Clang with modern STL |
| OpenMP | Compiler-provided | Parallel training loops |
| pthreads | POSIX | Runtime threading |

---

## Building

```bash
git clone https://github.com/som1o/pallas.git
cd pallas
cmake -S . -B build
cmake --build build -j4
```

Run tests:

```bash
ctest --test-dir build --output-on-failure
```

---

## Project Structure

```text
pallas/
|- CMakeLists.txt
|- include/                  # public headers
|- src/                      # implementation + mains
|- data/
|  |- class_weights.txt
|  |- model_config.json
|  |- train_config.json
|  |- scenario_example.json
|  |- model_zoo/
|  |  |- manifest.tsv
|  |- scenario_bank/
|- plugins/
|  |- default_tools.cpp
|- web/
|  |- index.html
|  |- app.js
|  |- style.css
|- tests/
|  |- model_tests.cpp
|  |- test_framework.h
|- logs/
```

---

## Binaries and Usage

### pallas - Training

Main trainer entrypoint (`src/main.cpp`).

```bash
./build/pallas [options]
```

| Option | Description |
|---|---|
| `--train-only` | Train and skip evaluation stages |
| `--resume [path]` | Resume from model state (default `../data/best_state.bin`) |
| `--rebuild-battle-data` | Regenerate battle data before training |
| `--model-zoo-dir <path>` | Override model zoo directory |
| `--log-dir <path>` | Override logs output directory |
| `--data-dir <path>` | Override data directory |
| `--inspect-model <path>` | Inspect serialized model architecture |
| `--benchmark-only` | Skip training, run benchmark only |
| `--benchmark-model <path>` | Model used for benchmark-only mode |
| `--benchmark-bank <path>` | Scenario bank directory |
| `--benchmark-report <path>` | Benchmark report output JSON path |

Examples:

```bash
./build/pallas --rebuild-battle-data --log-dir ../logs/run_20260307
./build/pallas --benchmark-only --benchmark-model ../data/best_state.bin --benchmark-bank ../data/scenario_bank
```

---

### pallas_battle - Battle Server

Battle runtime + embedded HTTP server (`src/battle_main.cpp`).

```bash
./build/pallas_battle [options]
```

| Option | Default | Description |
|---|---|---|
| `--port <n>` | `8080` | HTTP server port |
| `--web-root <path>` | `../web` | Static web asset root |
| `--replay <path>` | `../logs/battle_replay.bin` | Replay output path |
| `--scenario <path>` | (default scenario) | Load scenario JSON |
| `--tournament` | off | Run round-robin mode and exit |
| `--tournament-rounds <n>` | `1` | Tournament rounds |
| `--tournament-output <path>` | `../logs/tournament_results.json` | Tournament result file |
| `--distributed-node-id <n>` | `0` | Node ID |
| `--distributed-total-nodes <n>` | `1` | Total cluster nodes |
| `--distributed-bind-host <host>` | `0.0.0.0` | UDP bind host |
| `--distributed-bind-port <n>` | `19090` | UDP bind port |
| `--distributed-timeout-ms <n>` | `40` | UDP receive timeout |
| `--distributed-peers <csv>` | empty | Peer list (`host:port,host:port`) |

Example:

```bash
./build/pallas_battle --port 8080 --scenario ../data/scenario_example.json --replay ../logs/battle_replay.bin
```

Open `http://127.0.0.1:8080` for the Warroom UI.

---

### pallas_replay - Replay Reader

CLI replay inspector (`src/replay_main.cpp`).

```bash
./build/pallas_replay [options]
```

| Option | Default | Description |
|---|---|---|
| `--log <path>` | `../logs/battle_replay.bin` | Replay file |
| `--ticks <n>` | unlimited | Stop after N ticks |

Example:

```bash
./build/pallas_replay --log ../logs/battle_replay.bin --ticks 100
```

---

### pallas_tests - Unit Tests

```bash
./build/pallas_tests
# or
ctest --test-dir build --output-on-failure
```

---

## Web Interface

Warroom tabs:

- **Overview**: step/start/pause/end/reset, duration/tick-rate controls, diagnostics, map tag summary.
- **Models**: readiness status, per-country upload cards, hierarchy view, finalists/eliminations, load errors.
- **Details**: leaderboard, theater snapshot, messages, decision stream.
- **ComCent**: manual override command panel.

The map renderer now uses:

- `map.cells` for ownership
- `map.tags` for sea/strategic/chokepoints/ports/pass/crossing overlays
- `map.sea_zones` for naval zone tinting

---

## REST API

All routes are served by `pallas_battle`.

Read routes:

- `GET /api/state`
- `GET /api/leaderboard`
- `GET /api/models`
- `GET /api/diagnostics`
- `GET /api/meta`

Control routes:

- `POST /api/control/step`
- `POST /api/control/start`
- `POST /api/control/pause`
- `POST /api/control/end`
- `POST /api/control/reset`
- `POST /api/control/speed?ticks_per_second=<float>`
- `POST /api/control/duration?seconds=<u64>`
- `POST /api/control/duration?min_seconds=<u64>&max_seconds=<u64>`
- `POST /api/control/override?actor_country_id=<u16>&target_country_id=<u16>&strategy=<name>&terms_type=<text>&terms_details=<text>`

Upload route:

- `POST /api/upload-model?name=<model>`
- `POST /api/upload-model?team=<team>`
- `POST /api/upload-model?country_id=<id>&label=<name>`

`/api/meta` includes machine-readable strategy list, targeted strategies, API version, upload limits, and route groups used by the web app.

Notes:

- Control routes can be protected with origin/token checks.
- Upload body is raw binary model state (`.bin`).

---

## Scenario Configuration

Reference: `data/scenario_example.json`

Top-level fields:

```json
{
  "seed": 20260307,
  "tick_seconds": 3600,
  "ticks_per_match": 220,
  "map": {
    "width": 24,
    "height": 12,
    "cells": [0, 1, 2],
    "tags": [0, 1, 66],
    "sea_zones": [0, 1, 2]
  },
  "models": [
    {"name": "aster_ai", "team": "aster", "model_path": ""}
  ],
  "countries": [
    {
      "id": 1,
      "name": "Aster",
      "controller": "aster_ai",
      "adjacent": [2],
      "defense_pacts": [],
      "non_aggression_pacts": [2],
      "trade_treaties": [2],
      "intel_on_enemy": {"2": 56}
    }
  ]
}
```

`map.tags` bit flags (`include/simulation_engine.h`):

- `1` sea
- `2` strategic
- `4` chokepoint strait
- `8` chokepoint canal
- `16` mountain pass
- `32` river crossing
- `64` port

---

## Model Configuration

Reference: `data/model_config.json`

```json
{
  "hidden_layers": [512, 384, 256, 192],
  "activation": "relu",
  "norm": "layernorm",
  "use_dropout": true,
  "dropout_prob": 0.08,
  "leaky_relu_alpha": 0.01
}
```

Validation rules include:

- `hidden_layers` must be non-empty and > 0
- `activation` in `{relu, sigmoid, tanh, leaky_relu}`
- `norm` in `{layernorm, batchnorm, none}`
- `dropout_prob` in `[0,1)`
- `leaky_relu_alpha >= 0`

---

## Training Configuration

Reference: `data/train_config.json`

```json
{
  "epochs": 32,
  "batch_size": 64,
  "validation_split": 0.15,
  "early_stopping_patience": 8,
  "base_lr": 0.0006,
  "optimizer": "adam",
  "weight_decay": 0.00012,
  "scheduler": "cosine",
  "step_size": 6,
  "gamma": 0.94,
  "min_lr": 0.00001,
  "label_smoothing": 0.02,
  "use_class_weights": true,
  "class_weights_path": "../data/class_weights.txt",
  "use_actor_critic": true,
  "policy_loss_weight": 1.0,
  "value_loss_weight": 0.5,
  "reward_scale": 1.0,
  "entropy_coeff": 0.012
}
```

Validation rules include:

- `epochs`, `batch_size`, `early_stopping_patience` > 0
- `optimizer` in `{adam, sgd}`
- `scheduler` in `{step, exponential, cosine}`
- `validation_split` in `[0.01, 0.5]`
- `min_lr` in `[0, base_lr]`
- non-negative decay/smoothing/entropy/value loss terms

---

## Class Weights

Reference: `data/class_weights.txt`

Format:

```text
<action_name> <weight>
```

The file now uses a neutral-balanced profile centered near `1.0` so the trainer does not over-bias escalatory actions by default.

Actions are mapped in `src/main.cpp` and must match known tokens, including:

- `attack`, `defend`, `negotiate`, `surrender`
- `form_alliance`, `sign_trade_agreement`, `impose_embargo`
- `coup_attempt`, `tactical_nuke`, `strategic_nuke`, `cyber_attack`
- and the full 26-action vocabulary

---

## Model Zoo

Model snapshots are stored under `data/model_zoo/`.

- Metadata index: `data/model_zoo/manifest.tsv`
- Custom location: `--model-zoo-dir <path>`

---

## Distributed Mode

Each node runs `pallas_battle` with a partition ID and peer list. Nodes exchange decisions over UDP.

Example (2 nodes):

Node 0:

```bash
./build/pallas_battle --distributed-node-id 0 --distributed-total-nodes 2 --distributed-bind-host 0.0.0.0 --distributed-bind-port 19090 --distributed-peers 192.168.1.11:19090
```

Node 1:

```bash
./build/pallas_battle --distributed-node-id 1 --distributed-total-nodes 2 --distributed-bind-host 0.0.0.0 --distributed-bind-port 19090 --distributed-peers 192.168.1.10:19090
```

Use identical scenario/seed across nodes for deterministic startup alignment.

---

## Tournament Mode

Run round-robin and write JSON standings:

```bash
./build/pallas_battle --tournament --tournament-rounds 5 --scenario ../data/scenario_example.json --tournament-output ../logs/tournament_results.json
```

---

## Plugin System

Default tool plugin source:

- `plugins/default_tools.cpp`

Core registry interfaces:

- `include/tool_registry.h`

---

## Data Formats

| File | Type | Purpose |
|---|---|---|
| `data/model_config.json` | JSON | Model architecture/config |
| `data/train_config.json` | JSON | Trainer hyperparameters |
| `data/class_weights.txt` | Text | Action loss multipliers |
| `data/scenario_example.json` | JSON | Scenario schema reference |
| `data/scenario_bank/*.json` | JSON | Benchmark/tournament scenario set |
| `data/model_zoo/manifest.tsv` | TSV | Snapshot index |
| `logs/battle_replay.bin` | Binary | Replay frames |
| `logs/tournament_results.json` | JSON | Tournament output |
| `logs/scenario_benchmark.json` | JSON | Benchmark aggregate report |
| `logs/scenario_benchmark.tsv` | TSV | Benchmark leaderboard table |

---

## Environment Variables

| Variable | Description |
|---|---|
| `PALLAS_DATA_DIR` | Overrides trainer data root |
| `PALLAS_ALLOWED_ORIGIN` | Restricts control/upload route CORS origin |
| `PALLAS_CONTROL_TOKEN` | Enables bearer/header/query token auth for control/upload routes |
| `PALLAS_MAX_UPLOAD_BYTES` | Max binary upload payload size for `/api/upload-model` |

---

## License

See [LICENSE](LICENSE).
