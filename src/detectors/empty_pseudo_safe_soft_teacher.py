from typing import Optional

from mmdet.models.detectors.soft_teacher import SoftTeacher
from mmdet.registry import MODELS
from torch import Tensor

from mmdet.structures import SampleList


@MODELS.register_module()
class EmptyPseudoSafeSoftTeacher(SoftTeacher):
    """SoftTeacher with an explicit empty-pseudo guard.

    When no pseudo boxes survive filtering/projection for the whole
    unlabeled batch, the unsupervised branch must contribute exactly
    zero loss rather than background-only/fake supervision.
    """

    def loss_by_pseudo_instances(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
        batch_info: Optional[dict] = None,
    ) -> dict:
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
                    "OFFICIAL_PSEUDO_COUNT_EVIDENCE_FIELDS_MISSING"
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
                (
                    instances.reg_uncs
                    < self.semi_train_cfg.reg_pseudo_thr
                )
                .sum()
                .item()
            )

        if (
            classification_pseudo > total_pseudo
            or regression_pseudo > total_pseudo
        ):
            raise RuntimeError(
                "OFFICIAL_PSEUDO_COUNT_EVIDENCE_EXCEEDS_BASE_SET"
            )

        self._official_last_pseudo_counts = {
            "pseudo_count": int(total_pseudo),
            "classification_pseudo_count": int(classification_pseudo),
            "regression_pseudo_count": int(regression_pseudo),
        }

        if total_pseudo == 0:
            zero = next(
                parameter
                for parameter in self.student.parameters()
                if parameter.requires_grad
            ).sum() * 0.0

            return {
                "unsup_loss_rpn_cls": [zero for _ in range(5)],
                "unsup_loss_rpn_bbox": [zero for _ in range(5)],
                "unsup_loss_cls": zero,
                "unsup_loss_bbox": zero,
            }

        return super().loss_by_pseudo_instances(
            batch_inputs,
            batch_data_samples,
            batch_info,
        )