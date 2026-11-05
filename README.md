# Semi-Supervised Object Detection for Chest X-ray Abnormality Detection

**A controlled, paired evaluation of teacher–student semi-supervised object detection on VinDr-CXR**

This repository contains the implementation and research artifacts for a study comparing **teacher–student semi-supervised object detection (SSL)** with **supervised object detection (SUP)** under limited bounding-box annotation. The evaluation uses a fixed VinDr-CXR data realization, four labeled-data budgets, two detector architectures, and ten paired training seeds.

**Main finding:** Under the evaluated configuration and protocol, SSL did **not** improve detection performance over matched SUP. The mean paired difference (**SSL − SUP**) on final-test **bbox mAP@[0.50:0.95]** was **−1.11 AP points** (95% CI **[−1.18, −1.05]**). This is a **controlled empirical evaluation**, not a proposal of a new SSOD algorithm or evidence that SSOD is generally ineffective for chest X-rays.

> **Repository status:** The project includes completed official SUP/SSL experiments and final-result analyses. The previous README's “Phase 3A / training not started” statement is outdated. A public Git checkout alone is **not** a guarantee that the complete raw dataset, trained checkpoints, and externally stored execution artifacts are available.

## Research Questions

**Main research question.** In a controlled experimental design for multi-class chest X-ray detection, does teacher–student semi-supervised object detection improve detection performance over supervised learning when bounding-box annotation is limited?

Three subsidiary questions characterize the scope of the paired **SSL − SUP** difference:

1. **Label-budget dependence.** How does the difference in bbox mAP@[0.50:0.95] change across **1%, 5%, 10%, and 20%** labeled-image budgets?
2. **Effect characterization.** How is pseudo-label quality associated with the SSL − SUP difference, and how does the comparison manifest in the AP of low-support classes and false-positive burden on **No Finding** images?
3. **Architecture dependence.** After averaging equally across the four labeled-data budgets, does the SSL − SUP difference vary between **R50-FPN** and **Swin-T-FPN**?

The question organization follows the revised manuscript. Earlier internal analysis identifiers are implementation/history labels and **do not** constitute the current manuscript's research-question structure.

## Experimental Design

| Component | Protocol |
| --- | --- |
| Dataset | **VinDr-CXR**, using a controlled working subset of its original training portion |
| Working scope | **4,894 images**: 4,394 abnormal + 500 sampled No Finding images |
| Detection classes | **14** abnormality categories; No Finding is a zero-GT condition, not a 15th detection class |
| Train / validation / test | **3,426 / 734 / 734** images, one locked image-disjoint split |
| Labeled budgets | **1%, 5%, 10%, 20%**: **34 / 171 / 343 / 685** labeled training images |
| Unlabeled complement | **3,392 / 3,255 / 3,083 / 2,741** training images, respectively |
| Subset structure | Nested labeled subsets; fixed subset membership |
| Detector family | Faster R-CNN + FPN |
| Architecture conditions | ResNet-50-FPN (R50-FPN), Swin-T-FPN |
| Training conditions | Matched SUP vs. teacher–student SSL (Soft Teacher-based configuration) |
| Training repetitions | **10** prespecified paired seeds per architecture–budget condition |
| Main experiment | **160 runs**: 2 paradigms × 4 budgets × 2 architectures × 10 seeds |
| Additional reference | **20 SUP-100% runs**, excluded from primary SSL–SUP paired comparison |
| Primary endpoint | Final-test **bbox mAP@[0.50:0.95]** |
| Main estimand | SSL − SUP, aggregated equally across eight architecture–budget cells within each seed |

At a given budget, **SUP and SSL share the labeled subset, detector family, architecture condition, training seed, evaluation protocol and optimizer-update budget**. SSL additionally uses the unlabeled complement through teacher-generated pseudo-labels. Unlabeled-subset ground-truth annotations are not supplied to the detector after subset membership is locked.

The **1% budget uses 1,032 optimizer updates**, while the **5%, 10%, and 20% budgets use 2,064**. Therefore, cross-budget contrasts should **not** be interpreted as the isolated causal effect of annotation quantity.

### Training and Evaluation Flow

```text
VinDr-CXR original training portion (15,000 images)
          |
          v
Controlled working scope (4,894 images; 14 detection classes)
          |
          v
Fixed train / validation / test (3,426 / 734 / 734)
          |
          v
Nested labeled budgets: 1%, 5%, 10%, 20%
          |
          +---- L_b --> matched SUP training ------------+
          |                                              |
          +---- L_b + U_b --> teacher–student SSL -------+
                                                         |
                                     R50-FPN / Swin-T-FPN; 10 paired seeds
                                                         |
                                       Validation-based checkpoint selection
                                                         |
                                              Locked final-test evaluation
                                                         |
                                        Paired SSL − SUP statistical analysis
                                                         |
                             Main question + three subsidiary characterizations
```

Validation selects the **best SUP checkpoint** or the **best EMA Teacher checkpoint for SSL** using bbox mAP@[0.50:0.95]. The test partition is reserved for final evaluation rather than training or hyperparameter selection.

## Results

### Main question — Does SSL improve SUP?

| Primary paired comparison | Result |
| --- | ---: |
| Mean SSL − SUP difference | **−1.11 AP points** |
| Standard deviation across 10 seed-level aggregates | 0.09 AP points |
| Two-sided 95% confidence interval | **[−1.18, −1.05] AP points** |
| Prespecified one-sided test for positive SSL gain | **p = 1.000** |
| Seed-level aggregate differences | **10/10 negative** |
| Architecture–budget cell means | **8/8 negative** |

The prespecified **one-sided** confirmatory test does not support positive SSL gain. It should **not** be re-described after observing the results as a prespecified one-sided test proving that SSL is inferior.

### Subsidiary question 1 — Label-budget dependence

| Labeled budget | Mean SSL − SUP (AP points) |
| --- | ---: |
| 1% | −1.210 |
| 5% | −0.615 |
| 10% | −1.045 |
| 20% | −1.580 |

All four estimates are negative, and their magnitudes follow a **non-monotonic** pattern. The manuscript reports a Greenhouse–Geisser repeated-measures ANOVA and prespecified paired contrasts with Holm correction. These observations are conditional on a **single fixed split**, **one nested labeled-subset chain**, and the **budget-specific update schedules**.

### Subsidiary question 2 — Effect characterization

- **Pseudo-label quality:** The estimated association with SSL − SUP was positive but **not statistically supported** by the prespecified analysis (one-sided *p* = 0.138; Holm-adjusted *p* = 0.276). This does not establish a causal mechanism.
- **Low-support classes:** Exploratory SSL − SUP AP estimates were **−0.208 AP points** for Atelectasis and **−0.118 AP points** for Pneumothorax. These findings do not automatically generalize to all rare classes.
- **No Finding:** At the fixed evaluation threshold, false-positive burden decreased by **0.878 FP per negative image** on average under SSL relative to SUP. However, mean recall also decreased from **0.1820** (SUP) to **0.0912** (SSL). Fewer false positives must not be presented as an unqualified improvement.

### Subsidiary question 3 — Architecture dependence

| Architecture | Mean SSL − SUP (AP points) | Negative seeds |
| --- | ---: | ---: |
| R50-FPN | −1.153 | 10/10 |
| Swin-T-FPN | −1.073 | 10/10 |

The difference between architecture-specific SSL − SUP effects (Swin-T minus R50) was **+0.080 AP points**, with a two-sided 95% CI **[−0.046, +0.206]**. There is **insufficient evidence of a difference** between the two evaluated architecture conditions; this is not an equivalence finding or a comparison of their absolute mAP values.

## Repository Layout

The following paths are directories/files visible in the public repository. Some are historical staging, implementation, or intermediate research records rather than a curated release bundle.

| Path | Role |
| --- | --- |
| [`src/`](src/) | Core implementation |
| [`configs/`](configs/) | Model and experiment configurations |
| [`scripts/`](scripts/) | Dataset preparation, experiment and analysis utilities |
| [`data/`](data/) | Data-related code, metadata and/or local layout; **not a claim that the complete original VinDr-CXR dataset is redistributed** |
| [`experiments/`](experiments/) | Experiment definitions and execution-related material |
| [`analysis/`](analysis/) | Statistical analysis code and derived outputs |
| [`artifacts/`](artifacts/) | Execution/result artifacts and references |
| [`plots/`](plots/) | Plot-related files |
| [`reports/`](reports/) | Research reports and audits |
| [`tests/`](tests/) | Tests and protocol checks |
| [`docs/`](docs/) | Supporting research documentation |
| [`environment.yml`](environment.yml) | Conda environment specification |
| [`requirements.txt`](requirements.txt) | Python dependency specification |
| [`ARTIFACT_AND_EVIDENCE_CONTRACT.md`](ARTIFACT_AND_EVIDENCE_CONTRACT.md) | Artifact and evidence conventions |
| [`sn-article.tex`](sn-article.tex) | Repository manuscript source; **verify its revision before treating it as identical to the accompanying PDF** |

> **Audit note:** The repository also contains phase handoffs and staging directories. Their existence does not mean each is an official final output. Use the paper's current grouping for scientific interpretation, and trace actual outputs through the artifact/evidence records rather than equating old stage or RQ folder names with current research questions.

## Environment and Getting Started

The official implementation used a Python/MMDetection research environment, including **Python 3.10**, **PyTorch 2.1**, **MMCV 2.1**, **MMEngine 0.10**, and **MMDetection 3.3**. Consult the committed environment specifications and the exact run manifests before recreating an environment; GPU/CUDA and binary compatibility matter.

```bash
git clone https://github.com/kelvinquach/ssl_detection_xray_v2-main-backup.git
cd ssl_detection_xray_v2-main-backup

# Inspect the committed dependency specifications before installation:
cat environment.yml
cat requirements.txt
```

This deliberately stops short of advertising a one-command full reproduction: **a verified end-to-end public reproduction entry point, all required data placement instructions, and availability of full checkpoints have not been established by this README review**. Raw DICOM images must be obtained from the dataset's authorized distribution under its terms, and official model evaluation may require checkpoints/results stored outside Git.

To audit or reproduce the study, first locate the locked split and labeled-subset artifacts, then verify the exact configs and seeds, checkpoint availability, evaluator settings, and result tables before running the relevant scripts. Review [`ARTIFACT_AND_EVIDENCE_CONTRACT.md`](ARTIFACT_AND_EVIDENCE_CONTRACT.md) and [`reports/`](reports/) as starting points; these do not replace a verified executable reproduction guide.

## Reproducibility and Interpretation Boundaries

- The ten seeds capture **training stochasticity within a fixed data realization**, not uncertainty over alternative train/test splits or newly sampled labeled-subset chains.
- The controlled working scope retains all annotated abnormal images and samples only 500 No Finding images. Its composition **does not reproduce the source dataset's or clinical population's prevalence**.
- SSL is assessed through **one representative Soft Teacher-based configuration**, not an exhaustive evaluation of SSOD methods.
- Architecture comparisons include their prespecified architecture-specific training recipes; they do not isolate a pure backbone effect.
- Subset membership is fixed, and the test set is not a development/tuning resource.
- The study is an **object detection** study with bounding-box endpoints, not image-level classification, segmentation or clinical diagnosis.

## Data, Responsible Use, and Citation

**VinDr-CXR** is an externally sourced medical-imaging dataset. Please obtain it from its official distribution and comply with its applicable access and licensing requirements. Do not assume the full image dataset is included in this repository. The project is for **research evaluation**, not a validated clinical decision-support system.

When referencing this work, cite the corresponding research manuscript and the original VinDr-CXR dataset publication. Bibliographic publication details and a DOI for this manuscript are **not asserted here**, because they have not been verified as final.

**Study:** *Nghiên cứu học bán giám sát cho dò tìm bất thường trên X-quang phổi*  
**Authors:** Nguyen Thai Quach and Hai Thanh Nguyen  
**Repository:** https://github.com/kelvinquach/ssl_detection_xray_v2-main-backup
