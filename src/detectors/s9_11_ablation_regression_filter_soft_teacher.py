"""S9.11 OFAT ablation support for bbox-regression pseudo-label filtering.

Scientific scope
----------------
This detector exists ONLY for the prespecified S9.11 secondary mechanistic
ablation study.

Fixed ablation anchor:
    method        = SSL
    architecture  = Faster R-CNN R50-FPN
    labeled budget= 10%
    seeds         = all 10 locked training seeds

Only the pseudo-label selection rule used by the RCNN bbox-regression branch
is allowed to change in this component.

Prespecified variants:
    F0:
        initial pseudo score > 0.50
        no additional bbox-regression reliability filter

    F1:
        initial pseudo score > 0.50
        bbox-regression pseudo-labels additionally require score > 0.90

    F2 = frozen Main:
        initial pseudo score > 0.50
        bbox-regression pseudo-labels additionally require
        regression uncertainty < 0.02

OFAT invariant:
    Confidence filtering for RCNN classification, RPN filtering,
    strong photometric augmentation, EMA, L:U ratio, loss weights,
    optimizer, learning-rate schedule, update budget, validation evaluator,
    and every other scientific setting remain identical to frozen Main.

Important:
    F1 MUST NOT be implemented by setting reg_pseudo_thr = 0.90.
    reg_pseudo_thr is an uncertainty threshold, not a confidence score.

Evaluation:
    Non-Main F0/F1 variants are validation-only and are never eligible
    for final-test evaluation.

Thesis interpretation:
    The paired effect for each seed is measured against the frozen
    OFF_SSL_R50_B10 same-seed Main run.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
from mmdet.models.detectors.soft_teacher import SoftTeacher
from mmdet.registry import MODELS
from mmdet.structures import SampleList
from torch import Tensor

from src.detectors.empty_pseudo_safe_soft_teacher import (
    EmptyPseudoSafeSoftTeacher,
)


@MODELS.register_module()
class S911AblationRegressionFilterSoftTeacher(
    EmptyPseudoSafeSoftTeacher
):
    """SoftTeacher implementing the locked S9.11 F0/F1/F2 OFAT variants.

    This subclass deliberately changes only the selection of pseudo-labels
    used by the RCNN bbox-regression loss.

    F0
        Use every pseudo-label surviving the frozen initial score filter.

    F1
        Keep regression pseudo-labels with confidence score > 0.90.

    F2
        Frozen Main behaviour: keep regression pseudo-labels with
        reg_uncertainty < reg_pseudo_thr, where Main resolves to 0.02.

    F2 is implemented here as an explicit reference mode so that the
    ablation-specific code path can later be checked for behavioural
    equivalence with frozen Main before F0/F1 training is authorized.
    """

    VALID_ABLATION_VARIANTS = ("F0", "F1", "F2")
    F1_REGRESSION_SCORE_THRESHOLD = 0.90

    def __init__(
        self,
        ablation_regression_variant: str = "F2",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        variant = str(ablation_regression_variant).strip().upper()
        if variant not in self.VALID_ABLATION_VARIANTS:
            raise ValueError(
                "S9.11 regression ablation variant must be one of "
                f"{self.VALID_ABLATION_VARIANTS}; observed={variant!r}"
            )

        self.ablation_regression_variant = variant

    def _regression_keep_mask(self, instances):
        """Return the locked pseudo-label keep mask for F0/F1/F2.

        The input instances have already survived the frozen initial
        pseudo-label score filter in SoftTeacher.get_pseudo_instances().
        """

        if "scores" not in instances:
            raise RuntimeError(
                "S9_11_ABLATION_REGRESSION_SCORES_MISSING"
            )

        if "reg_uncs" not in instances:
            raise RuntimeError(
                "S9_11_ABLATION_REGRESSION_UNCERTAINTY_MISSING"
            )

        if self.ablation_regression_variant == "F0":
            # F0 scientific meaning:
            # no extra bbox-regression reliability filtering.
            return torch.ones_like(
                instances.scores,
                dtype=torch.bool,
            )

        if self.ablation_regression_variant == "F1":
            # F1 scientific meaning:
            # confidence-based filtering for bbox regression.
            return (
                instances.scores
                > self.F1_REGRESSION_SCORE_THRESHOLD
            )

        # F2 = frozen Main:
        # regression-uncertainty reliability filtering.
        return (
            instances.reg_uncs
            < self.semi_train_cfg.reg_pseudo_thr
        )

    def loss_by_pseudo_instances(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        batch_info: Optional[dict] = None,
    ) -> dict:
        """Preserve empty-pseudo safety and variant-correct evidence counts.

        EmptyPseudoSafeSoftTeacher records regression pseudo counts using the
        frozen Main uncertainty rule. That would be scientifically incorrect
        evidence for F0/F1.

        Therefore S9.11 reproduces the same safety contract here while
        computing regression_pseudo_count from the active ablation variant.
        """

        total_pseudo = sum(
            len(data_sample.gt_instances)
            for data_sample in batch_data_samples
        )

        classification_pseudo = 0
        regression_pseudo = 0

        for data_sample in batch_data_samples:
            instances = data_sample.gt_instances

            if "scores" not in instances or "reg_uncs" not in instances:
                raise RuntimeError(
                    "S9_11_ABLATION_PSEUDO_EVIDENCE_FIELDS_MISSING"
                )

            classification_pseudo += int(
                (
                    instances.scores
                    > self.semi_train_cfg.cls_pseudo_thr
                )
                .sum()
                .item()
            )

            regression_pseudo += int(
                self._regression_keep_mask(instances)
                .sum()
                .item()
            )

        if (
            classification_pseudo > total_pseudo
            or regression_pseudo > total_pseudo
        ):
            raise RuntimeError(
                "S9_11_ABLATION_PSEUDO_COUNT_EXCEEDS_BASE_SET"
            )

        # Keep exactly the official evidence-field schema expected by the
        # existing training-summary and closure machinery.
        self._official_last_pseudo_counts = {
            "pseudo_count": int(total_pseudo),
            "classification_pseudo_count": int(
                classification_pseudo
            ),
            "regression_pseudo_count": int(
                regression_pseudo
            ),
        }

        if total_pseudo == 0:
            # Preserve the official empty-pseudo safety behaviour:
            # an empty pseudo batch contributes exactly zero unsupervised loss.
            zero = next(
                parameter
                for parameter in self.student.parameters()
                if parameter.requires_grad
            ).sum() * 0.0

            return {
                "unsup_loss_rpn_cls": [
                    zero for _ in range(5)
                ],
                "unsup_loss_rpn_bbox": [
                    zero for _ in range(5)
                ],
                "unsup_loss_cls": zero,
                "unsup_loss_bbox": zero,
            }

        # Call the MMDetection SoftTeacher implementation directly.
        # Dynamic dispatch still routes the bbox-regression branch through
        # this class's rcnn_reg_loss_by_pseudo_instances() implementation.
        return SoftTeacher.loss_by_pseudo_instances(
            self,
            batch_inputs,
            batch_data_samples,
            batch_info,
        )

    def rcnn_reg_loss_by_pseudo_instances(
        self,
        x,
        unsup_rpn_results_list,
        batch_data_samples,
    ) -> dict:
        """Calculate RCNN bbox loss under the selected F0/F1/F2 rule."""

        rpn_results_list = copy.deepcopy(
            unsup_rpn_results_list
        )
        reg_data_samples = copy.deepcopy(
            batch_data_samples
        )

        for data_sample in reg_data_samples:
            instances = data_sample.gt_instances

            if len(instances) > 0:
                keep = self._regression_keep_mask(instances)
                data_sample.gt_instances = instances[keep]

        roi_losses = self.student.roi_head.loss(
            x,
            rpn_results_list,
            reg_data_samples,
        )

        return {
            "loss_bbox": roi_losses["loss_bbox"]
        }