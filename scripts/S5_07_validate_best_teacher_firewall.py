"""S5.07 independent proof that BEST EMA Teacher is blocked for Q_pseudo."""

import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from src.evaluation.qpseudo import (
    CHECKPOINT_ROLE,
    MODEL_ROLE,
    SCIENTIFIC_CHECKPOINT_ROLE,
    SERIALIZATION_SCOPE,
    load_last_ema_teacher_checkpoint,
)


ROOT = Path("/workspace/ssod/project")
EVIDENCE = (
    ROOT
    / "artifacts/preflight/evaluation/qpseudo_path_golden_test.json"
)

QPSEUDO_IMPL = ROOT / "src/evaluation/qpseudo.py"
ACCEPTANCE_CONFIG = (
    ROOT / "configs/evaluation/s5_06_qpseudo_acceptance.py"
)
S5_06_VALIDATOR = ROOT / "scripts/S5_06_validate_qpseudo_path.py"

HISTORICAL_S5_06_EVIDENCE_SHA256 = (
    "71b9ac534117830ad559ca69645520adc49f9a154b642f56d850bb4b71723b87"
)

EXPECTED_SOURCE_SHA256 = {
    "src/evaluation/qpseudo.py":
        "7b0bfb96fb595be24288d083576fabfb2367d48ea1d870c1c099bbe9a889f788",
    "configs/evaluation/s5_06_qpseudo_acceptance.py":
        "bb748e521af2eaeefe197a06d75b3fff1b3200f032725d42af61cfd1e2cac1c1",
    "scripts/S5_06_validate_qpseudo_path.py":
        "63bb6393bbdb5e897a2dfd17ff947373aab08de2849fec756de3bdf7cccb3b51",
}

EXPECTED_CANONICAL_SHA256 = {
    "sn-article.tex":
        "8d97840cc546a58aef7354af0c856119b1aea16f8be1cb7bbc1ab1a324d40fcc",
    "IMPLEMENTATION_HANDOFF.md":
        "540c94746571612ff819fb62a1e9fac6123c4035bc2642518eaa7740770820d0",
    "ARTIFACT_AND_EVIDENCE_CONTRACT.md":
        "885d4eeb66857e99744f6e8a27083e5bb695bbcc0d172cf4ebff722c53f655a4",
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    checks = []

    def check(name, ok, observed=None):
        checks.append(
            {
                "name": name,
                "pass": bool(ok),
                "observed": observed,
            }
        )

    print("===== S5.07 BEST TEACHER Q_PSEUDO FIREWALL =====")

    historical_evidence_sha = sha256(EVIDENCE)
    check(
        "historical_s5_06_evidence_identity",
        historical_evidence_sha
        == HISTORICAL_S5_06_EVIDENCE_SHA256,
        historical_evidence_sha,
    )

    historical = json.loads(
        EVIDENCE.read_text(encoding="utf-8")
    )

    check(
        "historical_s5_06_task",
        historical.get("task") == "S5.06",
        historical.get("task"),
    )
    check(
        "historical_s5_06_status",
        historical.get("status") == "PASS",
        historical.get("status"),
    )
    check(
        "historical_s5_06_checks_64_of_64",
        historical.get("checks_passed") == 64
        and historical.get("checks_total") == 64
        and historical.get("failed_count") == 0,
        {
            "passed": historical.get("checks_passed"),
            "total": historical.get("checks_total"),
            "failed": historical.get("failed_count"),
        },
    )
    check(
        "historical_s5_06_best_teacher_used_false",
        historical["scientific_protocol"].get(
            "best_teacher_used"
        ) is False,
        historical["scientific_protocol"].get(
            "best_teacher_used"
        ),
    )

    historical_checks = {
        item["name"]: item
        for item in historical.get("checks", [])
    }

    check(
        "historical_best_role_rejection_pass",
        historical_checks.get(
            "loader::best_role_rejected", {}
        ).get("pass") is True,
        historical_checks.get(
            "loader::best_role_rejected"
        ),
    )
    check(
        "historical_best_path_absence_pass",
        historical_checks.get(
            "firewall::best_teacher_path_absent", {}
        ).get("pass") is True,
        historical_checks.get(
            "firewall::best_teacher_path_absent"
        ),
    )

    for rel, expected in EXPECTED_SOURCE_SHA256.items():
        actual = sha256(ROOT / rel)
        check(
            "source_identity::" + rel,
            actual == expected,
            actual,
        )

    for rel, expected in EXPECTED_CANONICAL_SHA256.items():
        actual = sha256(ROOT / rel)
        check(
            "canonical_identity::" + rel,
            actual == expected,
            actual,
        )

    check(
        "constant_checkpoint_role_last_ema_teacher",
        CHECKPOINT_ROLE == "LAST_EMA_TEACHER",
        CHECKPOINT_ROLE,
    )
    check(
        "constant_scientific_role_last",
        SCIENTIFIC_CHECKPOINT_ROLE == "LAST",
        SCIENTIFIC_CHECKPOINT_ROLE,
    )
    check(
        "constant_model_role_ema_teacher",
        MODEL_ROLE == "EMA_TEACHER",
        MODEL_ROLE,
    )
    check(
        "constant_scope_teacher_only",
        SERIALIZATION_SCOPE == "EMA_TEACHER_ONLY",
        SERIALIZATION_SCOPE,
    )

    impl_text = QPSEUDO_IMPL.read_text(encoding="utf-8")
    cfg_text = ACCEPTANCE_CONFIG.read_text(encoding="utf-8")
    handoff_text = (
        ROOT / "IMPLEMENTATION_HANDOFF.md"
    ).read_text(encoding="utf-8")
    artifact_text = (
        ROOT / "ARTIFACT_AND_EVIDENCE_CONTRACT.md"
    ).read_text(encoding="utf-8")

    check(
        "implementation_requires_last_role",
        'meta.get("scientific_checkpoint_role") '
        '!= SCIENTIFIC_CHECKPOINT_ROLE'
        in impl_text,
        True,
    )
    check(
        "acceptance_config_requires_last_ema_teacher",
        'checkpoint_role = "LAST_EMA_TEACHER"'
        in cfg_text,
        True,
    )
    check(
        "qpseudo_impl_has_no_best_checkpoint_path",
        "best_ema_teacher.pth" not in impl_text,
        True,
    )
    check(
        "acceptance_config_has_no_best_checkpoint_path",
        "best_ema_teacher.pth" not in cfg_text,
        True,
    )

    check(
        "handoff_best_is_official_detection_only",
        "BEST EMA Teacher → official detection evaluation"
        in handoff_text,
        True,
    )
    check(
        "handoff_last_is_qpseudo",
        "LAST EMA Teacher → Q_pseudo on fixed validation"
        in handoff_text,
        True,
    )
    check(
        "handoff_best_substitution_prohibited",
        "Không được dùng BEST Teacher thay LAST Teacher"
        in handoff_text,
        True,
    )
    check(
        "artifact_best_to_qpseudo_forbidden",
        "BEST Teacher → Q_pseudo" in artifact_text,
        True,
    )
    check(
        "artifact_last_qpseudo_hash_relation",
        "LAST Teacher ↔ Q_pseudo checkpoint hash match"
        in artifact_text,
        True,
    )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        teacher = torch.nn.Linear(3, 2)
        state = {
            key: value.detach().cpu().clone()
            for key, value in teacher.state_dict().items()
        }

        model = SimpleNamespace(
            teacher=teacher,
            semi_test_cfg={"predict_on": "teacher"},
        )

        # BEST and LAST use exactly the same valid Teacher state_dict.
        # The only scientific difference here is checkpoint role metadata.
        best_path = td / "synthetic_best_teacher.pth"
        last_path = td / "synthetic_last_teacher.pth"

        torch.save(
            {
                "meta": {
                    "scientific_checkpoint_role": "BEST",
                    "model_role": "EMA_TEACHER",
                    "serialization_scope":
                        "EMA_TEACHER_ONLY",
                    "optimizer_update": 100,
                },
                "state_dict": state,
            },
            best_path,
        )

        torch.save(
            {
                "meta": {
                    "scientific_checkpoint_role": "LAST",
                    "model_role": "EMA_TEACHER",
                    "serialization_scope":
                        "EMA_TEACHER_ONLY",
                    "optimizer_update": 100,
                },
                "state_dict": state,
            },
            last_path,
        )

        best_rejected = False
        best_error = ""

        try:
            load_last_ema_teacher_checkpoint(
                model,
                best_path,
            )
        except ValueError as exc:
            best_error = str(exc)
            best_rejected = (
                "scientific_checkpoint_role='LAST'"
                in best_error
            )

        check(
            "runtime_best_checkpoint_rejected_for_qpseudo",
            best_rejected,
            best_error,
        )

        last_info = load_last_ema_teacher_checkpoint(
            model,
            last_path,
        )

        check(
            "runtime_last_checkpoint_accepted_for_qpseudo",
            last_info.get("checkpoint_role")
            == "LAST_EMA_TEACHER",
            last_info,
        )
        check(
            "runtime_last_scientific_role_is_last",
            last_info.get("scientific_checkpoint_role")
            == "LAST",
            last_info.get("scientific_checkpoint_role"),
        )
        check(
            "runtime_last_model_role_is_ema_teacher",
            last_info.get("model_role") == "EMA_TEACHER",
            last_info.get("model_role"),
        )
        check(
            "runtime_last_scope_is_teacher_only",
            last_info.get("serialization_scope")
            == "EMA_TEACHER_ONLY",
            last_info.get("serialization_scope"),
        )

        check(
            "runtime_best_and_last_state_dicts_identical",
            all(
                torch.equal(
                    torch.load(
                        best_path,
                        map_location="cpu",
                    )["state_dict"][key],
                    torch.load(
                        last_path,
                        map_location="cpu",
                    )["state_dict"][key],
                )
                for key in state
            ),
            True,
        )

        check(
            "runtime_best_rejection_is_role_based",
            best_rejected
            and "scientific_checkpoint_role='LAST'"
            in best_error,
            best_error,
        )

    failed = [
        item["name"]
        for item in checks
        if not item["pass"]
    ]
    passed = len(checks) - len(failed)
    all_pass = not failed

    print("CHECKS_PASSED=" + str(passed))
    print("CHECKS_TOTAL=" + str(len(checks)))
    print("FAILED_COUNT=" + str(len(failed)))
    print(
        "FAILED_CHECKS="
        + (",".join(failed) if failed else "NONE")
    )

    if not all_pass:
        print("STATUS=FAIL")
        return 1

    if "s5_07_best_teacher_firewall" in historical:
        raise RuntimeError(
            "S5.07 evidence section already exists"
        )

    section = {
        "task": "S5.07",
        "status": "PASS",
        "tracker_action":
            "Block BEST Teacher for Q_pseudo",
        "canonical_rule": {
            "best_ema_teacher_role":
                "OFFICIAL_DETECTION_EVALUATION_ONLY",
            "last_ema_teacher_role":
                "Q_PSEUDO_FIXED_VALIDATION",
            "best_teacher_qpseudo_blocked": True,
            "last_teacher_qpseudo_required": True,
        },
        "runtime_firewall": {
            "best_checkpoint_role": "BEST",
            "best_checkpoint_rejected": True,
            "required_scientific_checkpoint_role":
                "LAST",
            "accepted_checkpoint_role":
                "LAST_EMA_TEACHER",
            "same_valid_teacher_state_used_for_best_and_last":
                True,
            "rejection_basis":
                "SCIENTIFIC_CHECKPOINT_ROLE_METADATA",
        },
        "scientific_boundaries": {
            "official_detection_model":
                "BEST_EMA_TEACHER",
            "qpseudo_model":
                "LAST_EMA_TEACHER",
            "scientific_protocol_changed": False,
            "hidden_u_gt_used": False,
            "test_used": False,
        },
        "historical_s5_06": {
            "evidence_sha256_before_s5_07_extension":
                historical_evidence_sha,
            "task": historical.get("task"),
            "status": historical.get("status"),
            "checks_passed":
                historical.get("checks_passed"),
            "checks_total":
                historical.get("checks_total"),
        },
        "source_sha256": {
            "src/evaluation/qpseudo.py":
                sha256(QPSEUDO_IMPL),
            "configs/evaluation/s5_06_qpseudo_acceptance.py":
                sha256(ACCEPTANCE_CONFIG),
            "scripts/S5_06_validate_qpseudo_path.py":
                sha256(S5_06_VALIDATOR),
            "scripts/S5_07_validate_best_teacher_firewall.py":
                sha256(Path(__file__)),
        },
        "canonical_sha256": {
            rel: sha256(ROOT / rel)
            for rel in EXPECTED_CANONICAL_SHA256
        },
        "checks_passed": passed,
        "checks_total": len(checks),
        "failed_count": 0,
        "failed_checks": [],
        "checks": checks,
    }

    historical["s5_07_best_teacher_firewall"] = section

    EVIDENCE.write_text(
        json.dumps(
            historical,
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )

    print("STATUS=PASS")
    print(
        "BEST_TEACHER_QPSEUDO_BLOCKED="
        + str(
            section["canonical_rule"][
                "best_teacher_qpseudo_blocked"
            ]
        )
    )
    print(
        "OFFICIAL_DETECTION_MODEL="
        + section["scientific_boundaries"][
            "official_detection_model"
        ]
    )
    print(
        "QPSEUDO_MODEL="
        + section["scientific_boundaries"][
            "qpseudo_model"
        ]
    )
    print(
        "HISTORICAL_S5_06_EVIDENCE_SHA256="
        + historical_evidence_sha
    )
    print(
        "UPDATED_EVIDENCE_SHA256="
        + sha256(EVIDENCE)
    )
    print("S5_07_BEST_TEACHER_FIREWALL=PASS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())