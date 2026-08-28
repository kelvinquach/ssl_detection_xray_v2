# GATE I REVIEW — Freeze Implementation Inputs + Canonical Protocol Layer

**Gate:** I  
**Decision:** `BLOCKED / NOT CLOSED`  
**Official training:** `NOT STARTED / NOT AUTHORIZED`  
**Test performance:** `NOT READ`  
**Scientific revision:** `NONE`

## 1. Authority application

The canonical layer follows this order: `vietluanvan.md` → FINAL MASTER → locked manifests/config/evidence → handoff/context/readme/log → checklist. A lower-authority artifact is never allowed to overwrite a locked scientific value.

## 2. Exact implementation inputs that Gate I must verify

### Fixed data / split identity
- `data/processed/coco/coco_master_jpg.json`
- `data/processed/coco/instances_train.json`
- `data/processed/coco/instances_val.json`
- `data/processed/coco/instances_test.json`
- `data/manifests/split_lock_manifest.json`
- `data/processed/canonical/canonical_class_mapping.csv`

### Labeled / unlabeled identity
- `configs/protocol/phase2F_labeled_unlabeled.yaml`
- `data/manifests/phase2F_lock_manifest.json`
- `data/manifests/phase2F_nested_split_check.json`
- `data/manifests/phase2F_leakage_check.json`
- `reports/02F_labeled_unlabeled_validation_report.json`
- `reports/02F_deterministic_reconstruction_check.json`
- `data/processed/coco/labeled_splits/instances_labeled_{1pct,5pct,10pct,20pct}.json`
- `data/processed/coco/unlabeled_splits/instances_unlabeled_{1pct,5pct,10pct,20pct}.json`

### Seed identity
- `configs/protocol/phase2F1_seed_protocol.yaml`
- `data/manifests/seed_manifest.json`

### Input representation
- `configs/protocol/phase2D1_jpg_representation.yaml`
- `data/processed/images_jpg/` inventory, checked only for existence/mode/dimensions

### Environment
- actual GPU training environment snapshot recorded in `configs/protocol/gateI_environment_baseline.yaml`

## 3. Canonical configs created / mapped

Created in this bundle:
- `configs/protocol/experimental_design.yaml`
- `configs/protocol/d4_training_protocol.yaml`
- `configs/protocol/d4_ssl_protocol.yaml`
- `configs/protocol/canonical_protocol_index.yaml`
- `configs/protocol/gateI_freeze_manifest.yaml`
- `configs/protocol/gateI_environment_baseline.yaml`

Reused, not re-created:
- `configs/protocol/phase2F_labeled_unlabeled.yaml`
- `configs/protocol/phase2F1_seed_protocol.yaml`
- `configs/protocol/phase2D1_jpg_representation.yaml`

Proposed for their later implementation gates:
- `configs/protocol/d5_metric_protocol.yaml`
- `configs/protocol/d6_statistical_protocol.yaml`

## 4. Frozen expected identities

- Fixed split: train/val/test = `3426 / 734 / 734`; No Finding = `350 / 75 / 75`.
- Split COCO SHA-256: train `0f3c37...bbebe3`; val `33064f...1762a`; test `e1a731...c00a4`.
- Split membership SHA-256: train `628b9b...fa11a`; val `87c23e...58c6`; test `1f7903...66b37`.
- Split lock manifest SHA-256: `b696ec...6012b`.
- Category mapping SHA-256: `443f79...6163`; COCO categories must be exactly 1..14 and No Finding must not be a detection category.
- Phase 2F protocol SHA-256: `fb75e3...9e10`; lock manifest `d8659d...3c88`.
- L sizes: `34 / 171 / 343 / 685`; U sizes: `3392 / 3255 / 3083 / 2741`; nested L required.
- Phase 2F labeled membership uses its own canonical numeric-sort checksum convention; it is intentionally different from Phase 2E's string-sort convention.
- Seed protocol SHA-256: `ba5b1a...3234`; seed manifest `fdc9db...a40b`; exact ordered list = 10 seeds from Phase 2F.1.
- Input representation semantics: JPEG grayscale `L`, 1 storage channel, identical 3-channel replication at model input, JPEG quality 95, no representation-level resize/crop, geometry preserved. **Active representation YAML byte hash is not yet frozen in this Gate-I bundle because multiple historical Library variants exist.**

## 5. Guardrail result

Static/config guardrails in the delivered bundle: **14/14 PASS**. Python compile: **PASS**.

The full current-repository guardrail is intentionally **NOT EXECUTED** here because the current working-tree split/L-U JSONs, manifests, JPG inventory and GPU environment are not mounted as one repository snapshot. Historical evidence is internally consistent, but historical consistency is not equivalent to proving the current working tree has not drifted.

The guardrail is read-only. It does not train, infer, evaluate test performance, regenerate splits, resample L/U, or alter a scientific protocol.

## 6. Differences / blocker classification

1. **Current repository byte/membership not recomputed** → implementation evidence blocker, not a scientific revision.
2. **GPU/MMDetection training environment not frozen** → implementation blocker, not a scientific revision.
3. **Multiple historical `phase2D1_jpg_representation.yaml` variants in Library** → provenance/version-control blocker. Semantics are locked; active byte identity still needs pinning.
4. **Checklist drift:** Phase 4 still contains wording equivalent to choosing a main detector, treating ViT/Swin as optional, and evaluating fixed test during Phase 4. Higher authority locks both R50 and Swin-T as official architectures and keeps test final-only. This is a lower-authority governance/implementation bug. Do not modify the scientific contract to match the checklist.

If a current repo file differs from a frozen value: default classification is **IMPLEMENTATION BUG**. If the researcher actually intends to change that locked scientific value, stop and open **CONTROLLED REVISION REQUIRED**; never auto-edit the protocol.

## 7. Gate-II authorization

**NO. Gate II — Shared SUP Infrastructure is not authorized yet.**

Gate II can open only after:
- `scripts/04I_gate1_freeze_inputs.py` is run on the actual repository and returns zero FAIL;
- the active representation YAML SHA is pinned after provenance resolution;
- `gateI_environment_baseline.yaml` contains the actual GPU stack and status `FROZEN`;
- no scientific mismatch is unresolved.

Only then may implementation proceed to shared SUP infrastructure. Official training remains separately unauthorized until its later gates.
