# Phase 2F.1 - Seed Protocol

| field | value |
| --- | --- |
| document_status | `CANDIDATE_PENDING_RESEARCHER_GPT_REVIEW` |
| phase_closure_status | `PENDING_RESEARCHER_GPT_REVIEW` |
| approval_authority | `RESEARCHER_AND_GPT_REVIEW` |
| source_config | `configs/protocol/phase2F1_seed_protocol.yaml` |
| source_config_sha256 | `ba5b1a1adce67c3f1cf9dd46657e3db89c9d29b85cc37a744462c55a617d3234` |
| generated_at_utc | `2026-08-18T08:54:43Z` |
| training_authorized | `false` |
| training_started | `false` |

This document is a CANDIDATE contract. A `validation_status` of `PASS` in `reports/02F1_seed_protocol_validation_report.json` does not change `phase_closure_status`, which remains `PENDING_RESEARCHER_GPT_REVIEW` until the researcher and GPT review decide.

## 1. Scope

Phase 2F.1 builds the seed contract that Phase 4-5 inherits. It excludes:

- `training`
- `model_initialization`
- `dataloader_runtime`
- `augmentation_runtime`
- `pseudo_labeling`
- `checkpointing`
- `model_evaluation`
- `hyperparameter_selection`
- `threshold_selection`
- `gpu_determinism_verification`
- `rng_state_capture_or_restore`
- `dataset_membership_modification`
- `rebuild_of_phase2E_or_phase2F`

## 2. Inherited from Phase 2F (immutable here)

- `protocol_identity` = `2F-C0-R11`
- `protocol_version` = `2.0.0`
- `train_size` = 3426
- `labeled_budgets` = ['1pct', '5pct', '10pct', '20pct']
- `labeled_sizes` = [34, 171, 343, 685]
- `unlabeled_sizes` = [3392, 3255, 3083, 2741]

| budget | labeled membership SHA-256 |
| --- | --- |
| 1pct | `c54e7d61e84b7cfce68c04a795783b5c5d01331d2aa9c754fb1ae1dbae4ba071` |
| 5pct | `c4db3b5f7a5b0f391ad883ef665c3d341af3ef6afb3fc553e71f1c4ef17ee50b` |
| 10pct | `fc008d31505227544087ba474613075a8cd077587df9d6b13146b378f5a3a7d6` |
| 20pct | `6f4aaba6be147983d56007c49ada234dfcabe7ec2f936f28f8cd240999b8417e` |

## 3. Partition seed

- `partition_seed` = 42
- `partition_seed_policy` = `PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH`
- `split_seed = legacy alias of partition_seed`
- `seed_search_performed` = false
- `partition_seed_equals_training_seed` = false
- The partition seed already fixed the train / validation / test split and the labeled / unlabeled membership. Phase 2F.1 does not regenerate them.

## 4. Training seeds

- `training_seed_count` = 10
- `deterministic_policy` = `CONTROLLED_BEST_EFFORT`
- `seed_domain` = [1, 2147483647]
- `rng_calls_used` = 0

Generation rule (public, recomputed at every run of this script):

```
namespace = ssl_detection_xray_v2|phase2F.1|training_seed
payload_i = namespace + |index=i
digest_i  = SHA256(UTF-8(payload_i))
seed_i    = 1 + (integer(first 8 hexadecimal characters of digest_i) mod (2^31 - 1))
i = 1,...,10
```

| index | training_seed | payload | digest prefix |
| --- | --- | --- | --- |
| 1 | 204886845 | `ssl_detection_xray_v2|phase2F.1|training_seed|index=1` | `0c36533c` |
| 2 | 1480646854 | `ssl_detection_xray_v2|phase2F.1|training_seed|index=2` | `5840e0c5` |
| 3 | 1798418854 | `ssl_detection_xray_v2|phase2F.1|training_seed|index=3` | `6b31b1a5` |
| 4 | 2045683682 | `ssl_detection_xray_v2|phase2F.1|training_seed|index=4` | `f9eea7e0` |
| 5 | 1814859839 | `ssl_detection_xray_v2|phase2F.1|training_seed|index=5` | `6c2c903e` |
| 6 | 1603952859 | `ssl_detection_xray_v2|phase2F.1|training_seed|index=6` | `df9a60d9` |
| 7 | 1878351743 | `ssl_detection_xray_v2|phase2F.1|training_seed|index=7` | `eff55f7d` |
| 8 | 875651179 | `ssl_detection_xray_v2|phase2F.1|training_seed|index=8` | `b4316069` |
| 9 | 477581743 | `ssl_detection_xray_v2|phase2F.1|training_seed|index=9` | `9c7751ad` |
| 10 | 869675675 | `ssl_detection_xray_v2|phase2F.1|training_seed|index=10` | `33d6329a` |

## 5. Pairing policy

- For every labeled budget and every training_seed_index k in 1..10, the supervised baseline run and the semi-supervised run use the same training_seed and the same partition_seed.
- `supervised_and_ssl_share_same_ordered_seed_list` = true
- `per_method_seed_lists_allowed` = false
- `reshuffling_allowed` = false
- `comparison_unit` = `(method, configuration_id, budget, training_seed_index)`

## 6. Retry policy

- `retry_allowed_reason` = `TECHNICAL_FAILURE_ONLY`
- `retry_must_reuse_same_training_seed` = true
- `retry_changes_training_seed` = false
- `silent_replacement_allowed` = false
- `failed_run_must_be_retained` = true, with `run_status` = `TECHNICAL_FAILURE` and a non-null `technical_failure_reason`
- the retry run must set `retry_of` to the `run_id` of the original attempt
- `result_based_retry_allowed` = false

## 7. Aggregation policy

- per-seed results are reported for every one of the 10 seeds
- the mean across seeds is reported
- the sample standard deviation is reported with `ddof=1`
- `mean = (1 / n) * sum_k x_k`
- `sd = sqrt( sum_k (x_k - mean)^2 / (n - 1) )`
- `seed_dropping_allowed` = false, `outlier_removal_allowed` = false, `best_seed_only_reporting_allowed` = false

## 8. Reproducibility policy

- `deterministic_policy` = `CONTROLLED_BEST_EFFORT`
- controllable sources to be seeded in Phase 4-5: `python_random`, `numpy_random`, `torch_cpu`, `torch_cuda`, `dataloader_workers`, `sampler`, `augmentation`
- `runtime_settings_must_be_recorded` = true
- limitation: Identical numerical output across different GPU models, CUDA versions, cuDNN versions, drivers, hardware or software versions is not asserted.
- `policy_verified_by_runtime_in_phase2F1` = false

## 9. Rationale

### 9.1 `training_seed_count_rationale`

Ten training seeds are fixed in advance of any training run so that run-to-run variation of the reported metrics can be reported, so that the conclusions of the study do not rest on a single run, and so that the supervised baseline and the semi-supervised method can be compared under the same training_seed. The value 10 is a pre-registered protocol decision of the researcher: no claim of `statistical_optimality_of_seed_count` is made, no `power_analysis_performed` is claimed, `stability_proven` is not asserted, and `variance_measured_in_phase2F1` is false because Phase 2F.1 executes no training run.

### 9.2 `controlled_best_effort_rationale`

Under `CONTROLLED_BEST_EFFORT`, Phase 4-5 must seed every controllable source of randomness, namely Python, NumPy, PyTorch on CPU and CUDA, DataLoader workers, the sampler and the augmentation pipeline, and must record the runtime settings actually applied by each run. Identical numerical output across different GPU models, CUDA versions, cuDNN versions, drivers, hardware or software versions is not asserted; `bitwise_reproducibility_across_environments` is listed under prohibited_claims. Phase 2F.1 locks this policy as a contract only and does not verify it with any training runtime.

## 10. Claims explicitly not made in Phase 2F.1

- `bitwise_reproducibility_across_environments`
- `power_analysis_performed`
- `self_declared_phase_closure`
- `stability_proven`
- `statistical_optimality_of_seed_count`
- `variance_measured_in_phase2F1`

## 11. Mandatory metadata schema for every future run

`runs_created_in_phase2F1` = 0. The schema below constrains future runs only.

| field | required | description |
| --- | --- | --- |
| `run_id` | yes | Unique identifier of one training run attempt. |
| `run_status` | yes | Lifecycle state of the run attempt, for example PLANNED, RUNNING, COMPLETED or TECHNICAL_FAILURE. |
| `method` | yes | Method under comparison, for example the supervised baseline or the semi-supervised method. |
| `configuration_id` | yes | Identifier of the frozen configuration used by the run. |
| `budget` | yes | Labeled budget of the run, one of the inherited labeled_budgets. |
| `partition_seed` | yes | Inherited partition seed. Must equal 42 for every run. |
| `training_seed` | yes | Training seed value taken from ordered_training_seed_values. |
| `training_seed_index` | yes | Position 1..10 of training_seed inside the ordered list. |
| `rng_state_id` | yes | Identifier of the RNG state record associated with the run. Mandatory field for future runs only. |
| `membership_checksum` | yes | SHA-256 checksum of the labeled membership used, must equal the inherited checksum of the budget. |
| `config_hash` | yes | Hash of the resolved configuration actually used by the run. |
| `code_revision` | yes | Source code revision identifier of the run. |
| `environment` | yes | Recorded execution environment: hardware, driver, CUDA, cuDNN and library versions. |
| `deterministic_runtime_settings` | yes | Determinism related runtime settings actually applied by the run. |
| `checkpoint` | yes | Reference to the checkpoint produced by the run. Null until a real run produces one. |
| `results` | yes | Recorded metrics of the run. Null until a real run produces them. |
| `retry_of` | yes | run_id of the original attempt when this run is a retry, null otherwise. |
| `technical_failure_reason` | yes | Reason recorded when run_status is TECHNICAL_FAILURE, null otherwise. |

rng_state_id is a mandatory field of the future-run record only. No RNG state is captured or restored in Phase 2F.1.

## 12. Authorization

- `training_authorized` = false
- `training_started` = false
- `state` of `data/manifests/seed_state_manifest.json` = `TEMPLATE_LOCKED_NO_RUNS` with `runs = []`
- unlock condition: Researcher and GPT review of the Phase 2F.1 artifacts and console output.

## 13. Commands

```
python scripts\02F1_build_seed_protocol.py --execute
python -m pytest tests\test_phase2F1_seed_protocol_guardrails.py -v --junitxml=reports\02F1_guardrails_junit.xml
python scripts\02F1_build_seed_protocol.py --validate-existing
```

