import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_required_canonical_files_exist():
    for name in ["canonical_protocol_index.yaml","experimental_design.yaml","d4_training_protocol.yaml","d4_ssl_protocol.yaml","gateI_freeze_manifest.yaml","gateI_environment_baseline.yaml","phase2F_labeled_unlabeled.yaml","phase2F1_seed_protocol.yaml","phase2D1_jpg_representation.yaml"]:
        assert (ROOT/"configs/protocol"/name).is_file()

def test_fail_closed_authorization_and_test_firewall():
    texts="\n".join((ROOT/"configs/protocol"/p).read_text(encoding="utf-8") for p in ["experimental_design.yaml","d4_training_protocol.yaml","d4_ssl_protocol.yaml","gateI_freeze_manifest.yaml"])
    assert "training_authorized: false" in texts
    assert "performance_read_allowed: false" in texts
    assert "test_performance_read: false" in texts

def test_guardrail_is_importable_and_non_training():
    p=ROOT/"scripts/04I_gate1_freeze_inputs.py"; spec=importlib.util.spec_from_file_location("gate_i",p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    text=p.read_text(encoding="utf-8").lower()
    for forbidden in ["model.train(","inference_detector(","cocoeval(","pseudo-label generation"]: assert forbidden not in text

def test_membership_hash_convention():
    p=ROOT/"scripts/04I_gate1_freeze_inputs.py"; spec=importlib.util.spec_from_file_location("gate_i_hash",p); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    h=__import__("hashlib")
    assert mod.phase2e_membership_hash([10,2,1]) == h.sha256(b"1\n10\n2").hexdigest()
    assert mod.phase2f_membership_hash([10,2,1]) == h.sha256(b"1\n2\n10").hexdigest()

def test_environment_guardrail_is_fail_closed():
    text=(ROOT/"scripts/04I_gate1_freeze_inputs.py").read_text(encoding="utf-8")
    for token in ["gateI_gpu_runtime.txt","gateI_official_gpu_pip_freeze.txt","gateI_system_packages.txt","OFFICIAL_GPU_ENVIRONMENT_PROVENANCE_MISMATCH","PYTORCH_CUDA_RUNTIME","NVIDIA_SMI_CUDA_MAX_SUPPORTED"]:
        assert token in text

def test_environment_baseline_separates_local_and_official():
    text=(ROOT/"configs/protocol/gateI_environment_baseline.yaml").read_text(encoding="utf-8")
    assert "local_gate_i_audit_environment:" in text
    assert "official_gpu_training_environment:" in text
    assert 'nvidia_smi_cuda_max_supported: "12.4"' in text
    assert 'cuda_runtime: "11.8"' in text
