"""S7.P02 observation-only hook for the locked tiny SSL pilot.

This hook records runtime evidence only. It must not alter thresholds,
pseudo labels, losses, gradients, optimizer behavior, EMA behavior, or
validation/checkpoint semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import MethodType
from typing import Any, Dict

import torch
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmdet.registry import HOOKS


OBSERVATION_FILENAME = "pilot_runtime_observation.json"


def _state_dict_equal(left: torch.nn.Module, right: torch.nn.Module) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    if left_state.keys() != right_state.keys():
        return False
    return all(
        torch.equal(left_state[key], right_state[key])
        for key in left_state
    )


def _batch_size(samples: Any) -> int:
    try:
        return int(len(samples))
    except Exception:
        return -1


def _pseudo_counts(samples: Any) -> list[int]:
    counts = []
    for sample in samples:
        counts.append(int(len(sample.gt_instances)))
    return counts


def _all_zero_losses(losses: Dict[str, Any]) -> bool:
    tensors = []
    for value in losses.values():
        if isinstance(value, (list, tuple)):
            tensors.extend(item for item in value if torch.is_tensor(item))
        elif torch.is_tensor(value):
            tensors.append(value)
    return bool(tensors) and all(
        float(tensor.detach().abs().sum().item()) == 0.0
        for tensor in tensors
    )


@HOOKS.register_module()
class S7P02PilotObservationHook(Hook):
    """Observe the real SSL training path without changing its outputs."""

    priority = "NORMAL"

    def __init__(self) -> None:
        self._model = None
        self._original_instance_attrs = {}
        self._observation = None

    @staticmethod
    def _unwrap_model(runner: Any) -> Any:
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        return model

    def before_train(self, runner: Any) -> None:
        model = self._unwrap_model(runner)
        if not hasattr(model, "student") or not hasattr(model, "teacher"):
            raise RuntimeError(
                "S7P02_OBSERVER_REQUIRES_STUDENT_AND_TEACHER"
            )

        teacher_equal = _state_dict_equal(
            model.teacher,
            model.student,
        )

        self._model = model
        self._observation = {
            "schema_version": "1.0",
            "task": "S7.P02",
            "artifact_type": "PILOT_RUNTIME_OBSERVATION",
            "teacher_initialized_from_student_at_t0": teacher_equal,
            "model_type": type(model).__name__,
            "empty_pseudo_safe_model_active": (
                type(model).__name__ == "EmptyPseudoSafeSoftTeacher"
            ),
            "calls": {
                "loss_by_gt_instances": 0,
                "get_pseudo_instances": 0,
                "project_pseudo_instances": 0,
                "loss_by_pseudo_instances": 0,
            },
            "supervised_batch_sizes": [],
            "teacher_unlabeled_batch_sizes": [],
            "student_unlabeled_batch_sizes": [],
            "teacher_pseudo_counts_per_call": [],
            "projected_pseudo_counts_per_call": [],
            "loss_by_pseudo_total_pseudo": [],
            "empty_pseudo_call_count": 0,
            "nonempty_pseudo_call_count": 0,
            "empty_pseudo_zero_loss_verified": True,
        }

        self._patch_method(model, "loss_by_gt_instances", self._wrap_gt)
        self._patch_method(model, "get_pseudo_instances", self._wrap_get_pseudo)
        self._patch_method(
            model,
            "project_pseudo_instances",
            self._wrap_project_pseudo,
        )
        self._patch_method(
            model,
            "loss_by_pseudo_instances",
            self._wrap_pseudo_loss,
        )

    def _patch_method(self, model: Any, name: str, factory) -> None:
        if not hasattr(model, name):
            raise RuntimeError(f"S7P02_OBSERVER_MISSING_METHOD:{name}")
        existed = name in model.__dict__
        prior = model.__dict__.get(name)
        original = getattr(model, name)
        self._original_instance_attrs[name] = (existed, prior)
        setattr(model, name, MethodType(factory(original), model))

    def _wrap_gt(self, original):
        def wrapped(_model, batch_inputs, batch_data_samples, *args, **kwargs):
            obs = self._observation
            obs["calls"]["loss_by_gt_instances"] += 1
            obs["supervised_batch_sizes"].append(
                _batch_size(batch_data_samples)
            )
            return original(
                batch_inputs,
                batch_data_samples,
                *args,
                **kwargs,
            )
        return wrapped

    def _wrap_get_pseudo(self, original):
        def wrapped(_model, batch_inputs, batch_data_samples, *args, **kwargs):
            obs = self._observation
            obs["calls"]["get_pseudo_instances"] += 1
            obs["teacher_unlabeled_batch_sizes"].append(
                _batch_size(batch_data_samples)
            )
            output = original(
                batch_inputs,
                batch_data_samples,
                *args,
                **kwargs,
            )
            pseudo_samples = output[0]
            obs["teacher_pseudo_counts_per_call"].append(
                _pseudo_counts(pseudo_samples)
            )
            return output
        return wrapped

    def _wrap_project_pseudo(self, original):
        def wrapped(
            _model,
            pseudo_data_samples,
            target_data_samples,
            *args,
            **kwargs,
        ):
            obs = self._observation
            obs["calls"]["project_pseudo_instances"] += 1
            obs["student_unlabeled_batch_sizes"].append(
                _batch_size(target_data_samples)
            )
            output = original(
                pseudo_data_samples,
                target_data_samples,
                *args,
                **kwargs,
            )
            obs["projected_pseudo_counts_per_call"].append(
                _pseudo_counts(output)
            )
            return output
        return wrapped

    def _wrap_pseudo_loss(self, original):
        def wrapped(
            _model,
            batch_inputs,
            batch_data_samples,
            batch_info=None,
            *args,
            **kwargs,
        ):
            obs = self._observation
            obs["calls"]["loss_by_pseudo_instances"] += 1
            total_pseudo = int(sum(_pseudo_counts(batch_data_samples)))
            obs["loss_by_pseudo_total_pseudo"].append(total_pseudo)
            if total_pseudo == 0:
                obs["empty_pseudo_call_count"] += 1
            else:
                obs["nonempty_pseudo_call_count"] += 1

            losses = original(
                batch_inputs,
                batch_data_samples,
                batch_info,
                *args,
                **kwargs,
            )

            if total_pseudo == 0:
                obs["empty_pseudo_zero_loss_verified"] = bool(
                    obs["empty_pseudo_zero_loss_verified"]
                    and _all_zero_losses(losses)
                )
            return losses
        return wrapped

    def _restore(self) -> None:
        if self._model is None:
            return
        for name, (existed, prior) in self._original_instance_attrs.items():
            if existed:
                setattr(self._model, name, prior)
            elif name in self._model.__dict__:
                delattr(self._model, name)
        self._original_instance_attrs.clear()

    def after_train(self, runner: Any) -> None:
        try:
            obs = dict(self._observation or {})
            calls = obs.get("calls", {})
            obs["assertions"] = {
                "teacher_initialized_from_student_at_t0": bool(
                    obs.get("teacher_initialized_from_student_at_t0")
                ),
                "ssl_labeled_and_unlabeled_paths_exercised": bool(
                    calls.get("loss_by_gt_instances", 0) > 0
                    and calls.get("get_pseudo_instances", 0) > 0
                    and calls.get("project_pseudo_instances", 0) > 0
                    and calls.get("loss_by_pseudo_instances", 0) > 0
                ),
                "pseudo_label_generation_path_exercised": bool(
                    calls.get("get_pseudo_instances", 0) > 0
                    and calls.get("project_pseudo_instances", 0) > 0
                ),
                "empty_pseudo_safe_path_supported": bool(
                    obs.get("empty_pseudo_safe_model_active")
                    and (
                        obs.get("empty_pseudo_call_count", 0) == 0
                        or obs.get("empty_pseudo_zero_loss_verified")
                    )
                ),
            }

            path = Path(runner.work_dir) / OBSERVATION_FILENAME
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".json.tmp")
            temp.write_text(
                json.dumps(obs, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temp.replace(path)
        finally:
            self._restore()


__all__ = ["S7P02PilotObservationHook"]
