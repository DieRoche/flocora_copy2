# FlocoRa Implementation Audit

## 1. Scope

This README documents what the repository **actually executes** for `METHOD_NAME = FlocoRa`.

### What was treated as the active FlocoRa path

Based on repository naming and strategy wiring, the operational path for “FlocoRa” is the LoRA-based strategy path selected via:

- `--strategy fedlora` (or `--strategy fedloha` variant) in `main_ray.py`.
- `FedLora` strategy class in `strategies/fedlora.py`.
- LoRA injection and rank-pattern setup in `utils/lora.py`.

There is **no CLI literal `flocora` strategy value** in the switch logic; the run name uses the `flocora_*` prefix for experiment naming. So “FlocoRa” is represented in execution by the `fedlora`/`fedloha` branch, not by a separate strategy token.

---

## 2. High-Level Verdict

| Area | Verdict | Notes |
|---|---|---|
| Method activation | PARTIAL | FlocoRa name exists at run naming level, but runtime activation is `--strategy fedlora`/`fedloha` only. |
| Client compression path | PARTIAL | LoRA training restriction is active; optional fake quantization exists and is applied in-place before upload; no true encoded payload is sent. |
| Communication accounting | PARTIAL | Traffic is estimated from tensor size assumptions, not measured from actual serialized message bytes. |
| FLOPs accounting | PARTIAL | Local training FLOPs and estimated comp/decomp FLOPs are logged; server/eval FLOPs not counted; naming may overstate completeness. |
| Faithfulness clarity | UNCLEAR | Without paper-grounded stage list in repo, only code-structure-based expected stages can be compared. |

---

## 3. Actual Execution Pipeline

### End-to-end flow implemented

1. Parse arguments and build dataset partitions (Dirichlet split, with optional validation split).
2. Build server model and select strategy branch in `main_ray.py`.
3. For FlocoRa path (`fedlora`/`fedloha`):
   - Build base model.
   - Generate LoRA/LoHa target modules and rank pattern.
   - Inject PEFT adapter modules into server model.
   - Initialize Flower strategy `FedLora`.
4. Start Flower simulation.
5. Per round:
   - Server sends current parameters to sampled clients.
   - Each client instantiates its model, injects LoRA if configured, optionally applies fake quantization, sets received params, trains locally with SGD.
   - Client returns updated full state tensors as ndarray list plus metric dict.
   - Server aggregates returned tensors with weighted average.
   - Server evaluates global model on central test set.
   - W&B logs are emitted from both fit-metric aggregation and server evaluation paths.
6. End of run: history report and summary metrics are written and optionally logged.

### Active-code references

- Orchestration: `main_ray.py`
- Client training payload path: `client.py` -> `utils/mp_utils.py::mp_fit`
- Strategy aggregation: `strategies/fedlora.py::FedLora.aggregate_fit`
- Evaluation + traffic metrics: `utils/strats.py::EvaluateLora.__call__`
- W&B + metric aggregation: `utils/utils.py::aggregate_client_metrics`, `maybe_log_to_wandb`, `tell_history`

---

## 4. Method Activation and Required Flags

### Activation conditions

| Item | Status | Evidence |
|---|---|---|
| Explicit `flocora` strategy token | MISSING | `main_ray.py` strategy switch supports `fedavg`, `fedlora`/`fedloha`, `fedprox`. |
| FlocoRa operational branch | IMPLEMENTED DIFFERENTLY | FlocoRa behavior is represented by `--strategy fedlora` (and optionally `fedloha`) with LoRA config generation + injection. |
| Default strategy | `fedlora` | Parser default is `--strategy fedlora`. |

### Required/affecting flags for FlocoRa path

| Flag | Effect in active path |
|---|---|
| `--strategy fedlora` | Selects LoRA-based branch and `FedLora` strategy. |
| `--strategy fedloha` | Same branch, but LoHa config via `lora_type=args.strategy[3:]`. |
| `--lora_r` | Rank used in `gen_rank_pattern` and LoRA config. |
| `--lora_alpha` | Adapter scaling parameter in LoRA/LoHa config. |
| `--lora_ablation_mode` | Changes target_modules/modules_to_save/rank_pattern generation. |
| `--loha_ratio` | Used in rank calculation when ratio > 0. |
| `--apply_quant` + `--quant_bits` | Enables fake per-channel quantization pass in client before and after training. |
| `--prune`, `--prune_srv`, `--prate` | Optional pruning on uploaded params and/or received server params. |
| `--fedbn` | Excludes BN params from serialization/set path. |

### Generic FL vs method-specific vs helper-unused

| Category | Examples |
|---|---|
| Method-specific (FlocoRa path) | `main_ray.py` fedlora/fedloha branch; `strategies/fedlora.py`; LoRA injection/rank pattern in `utils/lora.py`. |
| Generic FL infrastructure | Flower client/server simulation, `strategies/fedavg.py` aggregation template, dataset partitioning, client process management. |
| Exists but not active for FlocoRa | `strategies/fedexp.py` not selected in `main_ray.py`; some utility helpers not called in active branch. |

---

## 5. Client-Side Processing After Local Training

### Object immediately after local training

After `train(...)` returns in `mp_fit`, the client model object `net` contains updated weights (LoRA-wrapped model if configured).

### Transformations before server uses payload

Ordered sequence in `mp_fit`:

1. Optional fake quantization pass `fakequant_trainable_channel(net, quant_bits)` if `apply_quant`.
2. Convert model state to list of NumPy arrays via `get_params(net, fedbn=...)`.
3. Optional pruning `prune(params, prate)` if `--prune`.
4. Place ndarray payload into `return_dict["params"]` for parent process.

### What is **not** done post-training

- No delta/update subtraction against global weights.
- No explicit normalization of updates.
- No sparse format encoding (COO/CSR/indices-values payload).
- No low-rank decomposition at transmission time (LoRA is architectural/training constraint, not a runtime decomposition of outgoing tensors).
- No explicit serialization/compression codec before Flower handoff.

---

## 6. Client-to-Server Payload and Transmission Logic

### Practical transmission mechanism

There is no custom network socket stack; Flower simulation handles transport. The repository-level “upload” handoff is:

- `FlowerClient.fit(...)` returns `(params, size, metrics)` where `params` came from child-process `return_dict["params"]`.
- `params` is a Python list of NumPy ndarrays from `get_params(...)`.

### Payload object type in active path

| Stage | Object |
|---|---|
| Child training output | `return_dict["params"]` (list of numpy ndarrays) |
| Client return | `params` from `_read_child_result` |
| Flower fit result | `fit_res.parameters` (Flower `Parameters` wrapping ndarray list) |
| Server aggregation input | `parameters_to_ndarrays(fit_res.parameters)` |

### Serialization status

- No explicit repository-side serialization/compression step before handoff.
- Transmission is effectively in-memory + Flower’s internal conversion for simulation.

---

## 7. Upload Traffic Validation

### `upload_traffic_per_client`

In active evaluation logging, `upload_traffic_per_client` is set to `model_size_bytes` derived by summing ndarray sizes of **server-evaluation parameters argument** (not measured wire bytes).

### `upload_traffic`

Computed as:

`upload_traffic = model_size_bytes * clients_per_round`

where `clients_per_round` resolves from config or `num_clients * samp_rate` fallback.

### Active-vs-total users check

- The multiplication uses estimated sampled clients per round (`clients_per_round`), not total users by default.
- However, this is still an estimate based on configuration, not actual per-round sampled client count from Flower callbacks.

### Reality check

Traffic metrics are **estimated analytical values**, not derived from actual transmitted payload representation/byte stream.

---

## 8. Server-Side Reconstruction / Decoding

| Operation | Status in active FlocoRa path | Notes |
|---|---|---|
| Decompression | MISSING | No decode of compressed bitstream. |
| Deserialization | PARTIAL/IMPLICIT | Flower converts `Parameters` to ndarrays; no custom codec logic in repo. |
| Dequantization | MISSING as transport step | Client uses fake quantization and sends dense float ndarrays, so no server dequant step. |
| Sparse reconstruction | MISSING | No sparse payload format is reconstructed. |
| Low-rank reconstruction | MISSING | No server-side reconstruction of factors into dense deltas. |
| Delta reconstruction | MISSING | Server aggregates full parameter tensors directly. |

---

## 9. Global Aggregation / Global Update Logic

Server update behavior (`FedLora.aggregate_fit`):

- Converts each client payload to ndarray list.
- Builds weighted tuples `(client_params, num_examples)`.
- Performs weighted average via Flower `aggregate(...)`.
- Returns aggregated parameters as new global model.

### Classification

- Aggregates **full weights** (state tensors), not deltas.
- No server optimizer step (no SGD/Adam on server model).
- Global “training” on server is aggregation + evaluation only.

---

## 10. Server-to-Client Payload and Download Logic

### Outgoing payload representation

The server strategy returns aggregated `Parameters`; next `configure_fit` uses those parameters in `FitIns` for sampled clients.

### Processing before send

| Processing type | Active status |
|---|---|
| Compression | Missing |
| Quantization | Missing |
| Sparsification/masking | Missing |
| Low-rank transform | Missing |
| Custom serialization | Missing |

No method-specific server-to-client payload transform is applied beyond Flower parameter container usage.

---

## 11. Download Traffic and Overall Traffic Validation

### Logged formulas

In traffic builder logic:

- `download_traffic = model_size_bytes * clients_per_round`
- `overall_traffic = upload_traffic + download_traffic`

### Validation

- Formula relation (`overall = upload + download`) is implemented.
- `download_traffic` is estimated from ndarray size of model parameters, not measured from true transmitted payload encoding.
- At run-end summary (`tell_history`), overall traffic mixes per-round and full-run values (upload/download reported as per-round; overall reported as total over rounds), which can be semantically inconsistent.

---

## 12. FLOPs Logging Validation

### Are training FLOPs logged to W&B?

Yes, via client metrics -> `aggregate_client_metrics` -> `maybe_log_to_wandb(step=server_round)`.

### `round_flops` and `total_flops`

- `round_flops` is emitted from summed client key `flops_by_epoch`.
- `total_flops` is a running cumulative sum of `round_total_flops` across rounds.

### Included components

| Component | Included in FLOPs accounting? | Where |
|---|---|---|
| Local training FLOPs | Yes (approx) | `FlopMeter` via `train()` -> `flops_by_epoch` |
| Client compression FLOPs | Estimated | `_estimate_lora_projection_flops` + `_estimate_quantization_flops` |
| Client decompression FLOPs | Estimated (mirrored same value) | Set equal to compression estimate |
| Server-side aggregation FLOPs | No | Not counted |
| Server evaluation FLOPs | No | Not counted |
| Real communication/serialization FLOPs | No | Not modeled |

### Naming caveat

`total_flops` accumulates training + estimated comp/decomp (through `round_total_flops`), while key name can be interpreted as all-system FLOPs; it is not full-system coverage.

---

## 13. Compression / Decompression FLOPs Validation

### `total_flops_compression`

- Reports running sum of `flops_compression` from clients.
- `flops_compression` is an estimate of:
  - LoRA projection FLOPs approximation.
  - Quantization FLOPs approximation (if `apply_quant`).

### Missing parts vs metric name implication

| Operation class | Counted? |
|---|---|
| Client compression (real encoding cost) | Partially (proxy estimate only) |
| Server decompression | No real decode path, not counted |
| Server compression | Not present |
| Client decompression | Counted only as mirrored estimate, not from real decode op |

So metric names suggest broader accounting than truly implemented.

---

## 14. Accuracy Logging Validation

### `acc_servers_highest`

There are two different uses:

1. In server evaluation metric builder (`_build_metrics`), `acc_servers_highest` is set equal to current round `test_acc`.
2. In run-summary (`tell_history`), `acc_servers_highest` is computed as `(mean + std)` over centralized accuracy history, not best-round max.

### Implications

- Name suggests “highest observed server accuracy”, but summary computation is statistical upper bound (`mean + std`), not max.
- Dataset/split for server accuracy is centralized test set (`test_set` loaded in `main_ray.py`, consumed by `Evaluate`/`EvaluateLora`).

---

## 15. Experiment Configuration Validation

Requested “standard setup” validation for FlocoRa:

| Item | Validation |
|---|---|
| Dirichlet alpha = 0.5 | Default `--alpha 0.5`; can be overridden; can be bypassed with `--alpha_inf`. Not enforced. |
| train/validation split = 80/20 | Default `--val_ratio 0.2`; applied in partitioning when `val_ratio > 0`. This corresponds to 80/20 by default, configurable. |
| optimizer = SGD | Enforced in active client training (`torch.optim.SGD`). |
| learning rate = 0.01 | Default `--cl_lr 0.01`; configurable; scheduler-like milestone adjustments can change it during rounds. |
| batch size = 128 | Default `--cl_bs 128`; configurable. |

Conclusion: these are mostly defaults, not immutable constraints.

---

## 16. WandB Metrics Audit

### Metrics logged from fit aggregation path

From `aggregate_client_metrics` and `maybe_log_to_wandb(step=server_round)`:

- FLOPs: `round_flops`, `round_flops_compression`, `round_flops_decompression`, `total_flops_including_compression`, `total_flops`, `total_flops_compression`, `total_flops_decompression`.
- Traffic/sparsity family if provided: `upload_traffic`, `download_traffic`, `overall_traffic`, `upload_traffic_per_client`, etc.

### Metrics logged from server evaluation path

From `Evaluate`/`EvaluateLora`:

- `distributed_loss`, `distributed_test_accuracy`, `acc_servers_highest`, `acc_server_highest`.
- plus traffic estimates (`upload_traffic`, `download_traffic`, `overall_traffic`, `upload_traffic_per_client`).

### End-of-run summary logging

`tell_history` logs summary report and optionally per-round traffic replay.

### Notes

- Same semantic field names can be produced by multiple paths with different meanings/timing.
- Some logged traffic values are estimated, not measured from true transport bytes.

---

## 17. Faithfulness to the Intended Method Structure

This section compares code against a **generic FlocoRa-like expectation inferred from repository organization** (LoRA-based compressed FL with communication accounting), not against external paper text.

| Expected stage (inferred) | Implemented status | Evidence-based comment |
|---|---|---|
| LoRA-constrained client fine-tuning | CORRECTLY IMPLEMENTED | Active `fedlora` branch injects low-rank adapters before training. |
| Client payload reduction due to LoRA trainable subset | PARTIALLY IMPLEMENTED | Trainable set is reduced by PEFT; however payload is still full ndarray state export path, not custom compact factor payload. |
| Optional quantization | IMPLEMENTED DIFFERENTLY | Fake quantization modifies model weights in-place; no true quantized transport object is sent. |
| Compression-aware server decode/reconstruction | MISSING | No explicit decode/dequant/reconstruction step on server. |
| Communication traffic tracking tied to real payload | PARTIAL | Metrics exist but are size estimates from model tensor sizes and configured clients/round. |
| Compression/decompression FLOPs accounting | PARTIAL | Proxy estimates logged; not derived from explicit codec operations. |
| Global aggregation for FL | CORRECTLY IMPLEMENTED | Weighted aggregation over client parameter tensors. |
| Server best-accuracy tracking semantics | PARTIAL | `acc_servers_highest` naming does not consistently mean true historical max. |

---

## 18. Mismatches, Risks, and Ambiguities

1. **Method naming ambiguity**: FlocoRa is not a direct strategy token; operational mapping relies on `fedlora`/`fedloha` branch interpretation.
2. **Payload realism gap**: Reported communication and compression semantics do not correspond to an explicitly encoded compressed payload pipeline.
3. **Traffic metric consistency risk**: In `tell_history`, `overall_traffic` uses total-over-rounds while `upload_traffic`/`download_traffic` are per-round.
4. **Accuracy naming mismatch**: `acc_servers_highest` can mean current-round accuracy in one place and mean+std in another.
5. **FLOPs completeness gap**: Server aggregation/evaluation FLOPs and real codec FLOPs are not covered.
6. **Config defaults vs enforcement**: “standard setup” values are defaults, not strictly enforced constraints.
7. **Unused helper code presence**: Some strategy/helper files exist but are not on active FlocoRa execution path.

---

## 19. Final Checklist

| Validation item | Status |
|---|---|
| 1. Method activation and actual pipeline identified | PASS |
| 2. Post-local-training preprocessing verified | PASS |
| 3. `round_flops` and `total_flops` wandb reporting validated | PASS (with scope caveats) |
| 4. Client-side compression logic classification | PASS |
| 5. Flags/config and wandb linkage validated | PASS |
| 6. Client-to-server payload object and handoff identified | PASS |
| 7. Upload traffic formula/active-user dependence validated | PARTIAL (estimated, config-based) |
| 8. Server-side decoding/reconstruction validated | PASS (mostly missing) |
| 9. Global update logic validated | PASS |
| 10. Server-to-client processing validated | PASS (no extra processing active) |
| 11. Download/overall traffic relation validated | PARTIAL (formula true, semantics mixed) |
| 12. Compression/decompression FLOPs coverage validated | PARTIAL |
| 13. `acc_servers_highest` behavior validated | PARTIAL (naming/semantics mismatch) |
| 14. Standard experiment config validated | PARTIAL (defaults, not enforced) |
| 15. Implementation-vs-intended-structure comparison | PASS (repo-inferred expectation only) |
