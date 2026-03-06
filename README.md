# Pallas

Pallas is a C++ battle-simulation and model-training platform. It supports:

1. Synthetic dataset generation
2. Neural policy training
3. Live browser-observable battle runtime
4. Replay logging and replay inspection

## Battle Interface

The canonical interface is defined in `include/battle_common.h`.

1. Input dimension: `90`
2. Output dimension: `26`

If a model state is not `90 x 26`, the battle runtime rejects it.

## Binaries

1. `pallas`
   Training and dataset pipeline.
2. `pallas_battle`
   Live battle HTTP server and web UI.
3. `pallas_replay`
   Replay reader/inspector.
4. `pallas_tests`
   Unit/integration test binary.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

## Run Training

```bash
cd build
./pallas --train-only --rebuild-battle-data
```

Optional path override:

```bash
./pallas --data-dir ../data
```

Or set:

```bash
export PALLAS_DATA_DIR=../data
```

## Run Battle Server

```bash
cd build
./pallas_battle --port 8080 --web-root ../web
```

Open `http://127.0.0.1:8080`.

### Security Environment Variables

1. `PALLAS_ALLOWED_ORIGIN`
   If set, mutable control/upload endpoints only accept this `Origin`.
2. `PALLAS_CONTROL_TOKEN`
   If set, mutable control/upload endpoints require token auth via:
   `Authorization: Bearer <token>`, or `X-Pallas-Token`, or `?token=`.
3. `PALLAS_MAX_UPLOAD_BYTES`
   Maximum upload body size for `POST /api/upload-model`.

### CORS

The server responds with CORS headers for accepted origins and supports `OPTIONS` preflight on API routes.

## Data Directory

`data/` typically contains:

1. `model_config.json`
2. `train_config.json`
3. `scenario_example.json`
4. `class_weights.txt`
5. `battle_train.json`
6. `battle_train.txt`
7. `vocab.txt`
8. `model_zoo/`

## Tests

```bash
cd build
ctest --output-on-failure
```

Test filtering is supported with:

```bash
PALLAS_TEST_FILTER=replay ./pallas_tests
```
