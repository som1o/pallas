# Pallas

**Pallas** is a geopolitical and military strategy AI simulation framework built in C++. It trains feedforward neural networks to act as autonomous nation-state decision-makers within a deterministic, fixed-point world simulation. Models learn to select from a vocabulary of 26 diplomatic and military actions based on a 90-dimensional representation of world state, enabling rigorous evaluation of AI strategic behavior under varied geopolitical conditions.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Building](#building)
- [Project Structure](#project-structure)
- [Binaries and Usage](#binaries-and-usage)
  - [pallas — Training](#pallas--training)
  - [pallas_battle — Battle Server](#pallas_battle--battle-server)
  - [pallas_replay — Replay Viewer](#pallas_replay--replay-viewer)
  - [pallas_tests — Unit Tests](#pallas_tests--unit-tests)
- [Web Interface](#web-interface)
- [Scenario Configuration](#scenario-configuration)
- [Model Configuration](#model-configuration)
- [Training Configuration](#training-configuration)
- [Class Weights](#class-weights)
- [Model Zoo](#model-zoo)
- [Distributed Mode](#distributed-mode)
- [Plugin System](#plugin-system)
- [Tournament Mode](#tournament-mode)
- [Data Formats](#data-formats)
- [Environment Variables](#environment-variables)
- [License](#license)

---

## Overview

Pallas simulates a world populated by nation-states, each governed by an AI model. At every simulation tick, each model receives a feature vector encoding that nation's military posture, economic indicators, diplomatic relationships, terrain conditions, and intelligence assessments. The model then selects the action it believes maximises its strategic position. Observed outcomes are recorded to a training corpus, which is subsequently used to further refine model weights.

The framework supports single-machine training and evaluation, live browser-based visualisation via the built-in Warroom interface, binary replay logging, round-robin tournament evaluation, and optionally a distributed multi-node simulation cluster communicated over UDP.

---

## Features

- **Neural network training** — Multilayer perceptron trained with Adam optimiser, cosine LR schedule, label smoothing, and configurable per-class loss weighting.
- **Deterministic simulation** — All world state arithmetic uses fixed-point representation (scale = 1000) to guarantee reproducibility across platforms.
- **26-action decision space** — Covering the full spectrum from diplomatic negotiation and trade agreements to cyber operations, conventional warfare, and nuclear escalation.
- **Web-based Warroom UI** — Single-page browser application served directly by `pallas_battle`, providing real-time map rendering, playback controls, and country detail panels.
- **Binary replay logging** — Every tick's state and model decisions are serialised to a compact binary log for offline review and regression testing.
- **Tournament mode** — Round-robin evaluation across a pool of models with an automatically generated leaderboard and JSON results export.
- **Distributed simulation** — Partitioned world simulation across multiple nodes using raw UDP messaging.
- **Model zoo** — Versioned model snapshots keyed by architecture hash, with a TSV manifest recording training metadata.
- **Plugin system** — Shared library (`.so`) plug-ins for extending the strategy tool registry at runtime.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        pallas_core (static library)          │
│                                                              │
│  Model ─── Tensor ─── Linear                                 │
│  SimulationEngine ─── World ─── Country ─── MilitaryPower    │
│  BattleRuntime ─── BattleServer                              │
│  ScenarioConfig ─── Tournament                               │
│  DataPipeline ─── DataLoader                                 │
│  ToolRegistry ─── Plugin Loader                              │
└──────────────┬──────────────┬──────────────┬────────────────┘
               │              │              │
          pallas         pallas_battle  pallas_replay
         (trainer)       (HTTP server   (CLI replay
                         + sim loop)     viewer)
```

The simulation engine maintains a priority-queue of scheduled events and advances the world tick-by-tick. At each tick, each living nation invokes its assigned `Model::decide()` with a fresh `CountrySnapshot` feature vector. The resulting action is dispatched to the appropriate handler, side-effects are applied to world state, and the (state, action) pair is optionally appended to the training corpus.

---

## Prerequisites

| Dependency | Minimum Version | Notes |
|---|---|---|
| CMake | 3.10 | Build system |
| C++ compiler | C++17 | GCC 8+ or Clang 7+ |
| OpenMP | — | Parallel training loops; typically bundled with the compiler |
| pthreads | — | POSIX threads |
| Internet access (first build) | — | `FetchContent` downloads nlohmann/json v3.11.3 |

---

## Building

```bash
git clone https://github.com/your-org/pallas.git
cd pallas
mkdir build && cd build
cmake ..
make -j$(nproc)
```

All build artifacts are placed inside the `build/` directory. The build system fetches [nlohmann/json](https://github.com/nlohmann/json) automatically if it is not already present in the build tree.

To run the unit test suite after building:

```bash
cd build
ctest --output-on-failure
```

---

## Project Structure

```
pallas/
├── CMakeLists.txt          # Top-level build definition
├── data/
│   ├── model_config.json   # Neural network architecture settings
│   ├── train_config.json   # Training hyperparameters
│   ├── class_weights.txt   # Per-action loss weights
│   ├── vocab.txt           # Action vocabulary (one token per line)
│   ├── battle_train.json   # Training corpus (state → action pairs)
│   ├── scenario_example.json
│   └── model_zoo/
│       └── manifest.tsv    # Versioned model snapshot index
├── include/                # Public headers for all modules
├── src/                    # Implementation files + binary entry points
├── plugins/
│   └── default_tools.cpp   # Default strategy tool plugin
├── tests/
│   ├── test_framework.h    # Lightweight test harness
│   └── model_tests.cpp
├── web/
│   ├── index.html          # Warroom single-page application
│   ├── app.js              # Map rendering and REST polling
│   └── style.css
└── logs/
    └── uploads/
```

---

## Binaries and Usage

### pallas — Training

The primary training binary. Loads the world simulation, generates or loads battle data, trains the model, and optionally evaluates performance.

```
./pallas [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--train-only` | — | Execute only the training phase; skip all evaluation |
| `--resume [path]` | `../data/best_state.bin` | Resume training from a saved model state |
| `--rebuild-battle-data` | — | Regenerate the training corpus from a fresh simulation run |
| `--model-zoo-dir <path>` | `../data/model_zoo` | Directory used for versioned model snapshots |
| `--log-dir <path>` | `../logs` | Directory for training logs and metric output |
| `--data-dir <path>` | `../data` | Root data directory (overridden by `$PALLAS_DATA_DIR`) |
| `--inspect-model <path>` | — | Print model architecture details without training |

**Example — resume training and write logs to a custom directory:**

```bash
./pallas --resume ../data/checkpoint.bin --log-dir ../logs/run_03
```

---

### pallas_battle — Battle Server

Runs the battle simulation loop and serves the Warroom browser interface and REST API over HTTP. Optionally executes round-robin tournaments between loaded models.

```
./pallas_battle [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--port <n>` | 8080 | TCP port for the embedded HTTP server |
| `--web-root <path>` | `../web` | Root directory for static web assets |
| `--replay <path>` | `../logs/battle_replay.bin` | Output path for the binary replay log |
| `--scenario <path>` | — | JSON scenario file to load at startup |
| `--tournament` | — | Run a round-robin tournament instead of a live battle |
| `--tournament-rounds <n>` | 1 | Number of rounds per tournament |
| `--tournament-output <path>` | `../logs/tournament_results.json` | JSON file for tournament leaderboard output |
| `--distributed-node-id <n>` | 0 | This node's identifier within a distributed cluster |
| `--distributed-total-nodes <n>` | 1 | Total number of nodes in the cluster |
| `--distributed-bind-host <h>` | `0.0.0.0` | UDP interface to bind for cluster communication |
| `--distributed-bind-port <n>` | 19090 | UDP port to bind for cluster communication |
| `--distributed-peers <csv>` | — | Comma-separated list of peer addresses (`host:port`) |

**Example — start a battle on port 9000 with a custom scenario:**

```bash
./pallas_battle --port 9000 --scenario ../data/scenario_example.json
```

Then open `http://localhost:9000` in a browser to access the Warroom.

---

### pallas_replay — Replay Viewer

Reads a binary replay log and prints per-tick state summaries to standard output. Useful for post-hoc analysis and regression verification.

```
./pallas_replay [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--log <path>` | `../logs/battle_replay.bin` | Binary replay file to read |
| `--ticks <n>` | unlimited | Maximum number of ticks to display |

**Example — inspect the first 200 ticks of a replay:**

```bash
./pallas_replay --log ../logs/battle_replay.bin --ticks 200
```

---

### pallas_tests — Unit Tests

The test binary is registered with CTest. Run it via:

```bash
cd build
ctest --output-on-failure
# or directly:
./pallas_tests
```

Tests cover model forward/backward passes, fixed-point arithmetic, tensor operations, and scenario loading.

---

## Web Interface

The **Pallas Warroom** is a single-page application served at the root of the `pallas_battle` HTTP server. No external web server is required.

The Warroom is divided into four tabs:

| Tab | Purpose |
|---|---|
| **Overview** | Start, step, pause, reset, and end the current battle. Adjust tick rate and simulation duration. |
| **Models** | Inspect loaded AI models, their architecture hashes, and action distribution statistics. |
| **Details** | View per-country state including military power, economic indicators, diplomatic stance, and active treaties. |
| **ComCent** | Command centre panel for issuing manual diplomatic or military orders and observing event logs. |

The map canvas in `app.js` renders the terrain grid, country ownership, and unit positions by polling the REST API exposed by `pallas_battle`.

---

## Scenario Configuration

Scenarios are defined as JSON files. A minimal scenario specifies a random seed, tick parameters, a map grid, and a list of countries with their initial state. See `data/scenario_example.json` for a fully annotated reference.

Key top-level fields:

```json
{
  "seed": 42,
  "max_ticks": 2000,
  "tick_interval_ms": 100,
  "map": { "width": 32, "height": 32, "cells": [ ... ] },
  "countries": [
    {
      "id": 1,
      "name": "Arandis",
      "military": { ... },
      "economy": { ... },
      "diplomacy": { "stance": "Neutral", "treaties": [ ... ] },
      "model_path": "../data/best_state.bin"
    }
  ]
}
```

Countries that do not specify a `model_path` will act randomly.

---

## Model Configuration

Neural network architecture is controlled by `data/model_config.json`:

```json
{
  "hidden_layers": [512, 384, 256, 192],
  "activation": "leaky_relu",
  "norm": "layernorm",
  "use_dropout": true,
  "dropout_prob": 0.1,
  "leaky_relu_alpha": 0.02
}
```

| Field | Description |
|---|---|
| `hidden_layers` | List of hidden layer widths. Input is 90-dimensional; output is 26 (one logit per action). |
| `activation` | Activation function. Supported: `relu`, `leaky_relu`, `tanh`. |
| `norm` | Normalisation layer. Supported: `layernorm`, `none`. |
| `use_dropout` | Whether to apply dropout during training. |
| `dropout_prob` | Dropout keep probability. |
| `leaky_relu_alpha` | Negative slope coefficient when `activation` is `leaky_relu`. |

---

## Training Configuration

Training hyperparameters are read from `data/train_config.json`:

```json
{
  "epochs": 28,
  "batch_size": 64,
  "validation_split": 0.12,
  "early_stopping_patience": 7,
  "base_lr": 0.0007,
  "optimizer": "adam",
  "weight_decay": 0.00015,
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "adam_epsilon": 1e-8,
  "scheduler": "cosine",
  "step_size": 8,
  "gamma": 0.92,
  "min_lr": 0.000005,
  "label_smoothing": 0.02,
  "use_class_weights": true,
  "class_weights_path": "../data/class_weights.txt"
}
```

---

## Class Weights

`data/class_weights.txt` assigns a scalar loss multiplier to each action token to compensate for class imbalance and to bias training toward or away from particular behaviours. Each line has the format `<action_name> <weight>`.

Higher weights cause the model to penalise mispredictions of that action more heavily, increasing its propensity to select those actions in ambiguous states. The currently configured weights are tuned for aggressive strategic posturing; escalatory actions such as `coup_attempt`, `tactical_nuke`, `strategic_nuke`, and `impose_embargo` carry elevated weights.

---

## Model Zoo

Trained model snapshots are stored under `data/model_zoo/` and tracked in `manifest.tsv`. Each entry records:

- Architecture hash (derived from layer dimensions and activation configuration)
- Training epoch and validation loss at the time of saving
- Timestamp and originating data directory

To point the trainer at a custom zoo location:

```bash
./pallas --model-zoo-dir /mnt/storage/pallas_models
```

---

## Distributed Mode

`pallas_battle` supports partitioning the country simulation across multiple nodes. Each node owns a disjoint subset of countries and communicates state deltas to its peers via UDP.

**Example — two-node cluster:**

Node 0:
```bash
./pallas_battle --distributed-node-id 0 --distributed-total-nodes 2 \
  --distributed-bind-host 0.0.0.0 --distributed-bind-port 19090 \
  --distributed-peers 192.168.1.11:19090
```

Node 1:
```bash
./pallas_battle --distributed-node-id 1 --distributed-total-nodes 2 \
  --distributed-bind-host 0.0.0.0 --distributed-bind-port 19090 \
  --distributed-peers 192.168.1.10:19090
```

All nodes must be started with identical scenario files and seeds to ensure deterministic world initialisation before the first synchronisation tick.

---

## Plugin System

Custom strategy tools can be loaded at runtime as shared libraries. Plugins must implement the `ToolRegistry` interface defined in `include/tool_registry.h`. The default plugin is built alongside the main targets:

```
build/default_tools_plugin.dir/  → libdefault_tools.so
```

To load a custom plugin, set the tool registry path in the scenario configuration or pass it programmatically before simulation start.

---

## Tournament Mode

Tournament mode runs a full round-robin contest among all models registered in a scenario, accumulating win/loss/draw statistics over one or more rounds.

```bash
./pallas_battle \
  --tournament \
  --tournament-rounds 5 \
  --scenario ../data/scenario_example.json \
  --tournament-output ../logs/results.json
```

Results are written to a JSON file containing a leaderboard sorted by win rate, along with per-matchup outcome matrices and aggregate action distribution statistics.

---

## Data Formats

| File | Format | Description |
|---|---|---|
| `battle_train.json` | JSON array | Training corpus; each entry is a `(state_vector, action_index)` pair |
| `battle_train.txt` | Text | Human-readable equivalent of the training corpus |
| `battle_replay.bin` | Binary | Tick-indexed replay log; each frame records full country state and model decisions |
| `model_zoo/manifest.tsv` | TSV | Index of saved model versions with metadata |
| `vocab.txt` | Text | Ordered list of the 26 action tokens (one per line) |
| `class_weights.txt` | Text | Per-action loss weight, space-separated: `<action> <weight>` |
| `scenario_example.json` | JSON | Reference scenario configuration |
| `model_config.json` | JSON | Neural network architecture definition |
| `train_config.json` | JSON | Training hyperparameters |
| `*.bin` model states | Binary | Serialised model weights + architecture hash + training metadata |

---

## Environment Variables

| Variable | Description |
|---|---|
| `PALLAS_DATA_DIR` | Overrides the default `../data` data directory path for all binaries |

---

## License

This project is released under the terms of the license found in the [LICENSE](LICENSE) file at the root of the repository.
