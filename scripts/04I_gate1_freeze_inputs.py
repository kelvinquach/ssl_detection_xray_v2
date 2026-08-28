#!/usr/bin/env python3
"""Gate I structural/integrity audit. Never trains, infers, or computes metrics."""
from __future__ import annotations

import argparse, hashlib, json, os, platform, re, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = {"1pct": (34, 3392, 3), "5pct": (171, 3255, 17), "10pct": (343, 3083, 35), "20pct": (685, 2741, 70)}

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()

def load(path: Path):
    with path.open(encoding="utf-8") as f: return json.load(f)

def phase2e_membership_hash(ids) -> str:
    payload = "\n".join(str(v) for v in sorted(ids, key=lambda v: str(v))).encode()
    return hashlib.sha256(payload).hexdigest()

def phase2f_membership_hash(ids) -> str:
    payload = "\n".join(str(int(v)) for v in sorted(map(int, ids))).encode()
    return hashlib.sha256(payload).hexdigest()

def coco(path: Path):
    d = load(path); ids = {int(x["id"]) for x in d["images"]}; anns = d.get("annotations", [])
    annotated = {int(a["image_id"]) for a in anns}
    return {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path),
            "images": len(d["images"]), "annotations": len(anns), "zero_gt": len(ids-annotated),
            "ids": ids, "phase2e_membership_sha256": phase2e_membership_hash(ids),
            "phase2f_membership_sha256": phase2f_membership_hash(ids), "categories": d.get("categories", []),
            "dimensions_ok": all(int(i.get("width",0))>0 and int(i.get("height",0))>0 for i in d["images"])}

def check(name, ok, expected, actual, evidence, out):
    out.append({"id": name, "status": "PASS" if ok else "FAIL", "expected": expected, "actual": actual, "evidence": evidence})

def git(*args):
    try: return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as e: return f"UNAVAILABLE: {e}"

def key_values(path: Path):
    out={}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "=" in raw:
            k,v=raw.split("=",1); out[k.strip()]=v.strip()
    return out

def yaml_scalar(text: str, key: str):
    matches=re.findall(rf"(?m)^\s*{re.escape(key)}:\s*[\"']?([^\"'\n]+?)[\"']?\s*$",text)
    return matches[-1].strip() if matches else None

def yaml_provenance_hash(text: str, artifact_key: str):
    m=re.search(rf"(?ms)^\s{{4}}{re.escape(artifact_key)}:\s*\n\s+path:.*?\n\s+sha256:\s*[\"']?([0-9a-f]{{64}})[\"']?\s*$",text)
    return m.group(1) if m else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--json-output", default="reports/GATE_I_AUDIT.json"); ap.add_argument("--no-write", action="store_true"); args=ap.parse_args()
    checks=[]; blockers=[]
    required_authority=["MASTER_D1_D7_FINAL_IMPLEMENTATION_CONTRACT_VI.md","vietluanvan.md","PHASE_HANDOFF.md","PROJECT_CONTEXT.md","README.md","research_log.md","CHECKLIST_TRIEN_KHAI_FULL.xlsx"]
    authority={p:(ROOT/p).exists() for p in required_authority}
    for p,exists in authority.items(): check("authority:"+p, exists, True, exists, p, checks)
    if not authority[required_authority[0]]: blockers.append("MISSING_MASTER_D1_D7_FINAL_IMPLEMENTATION_CONTRACT_VI")
    if not authority[required_authority[1]]: blockers.append("MISSING_CONFIRMED_CURRENT_SCIENTIFIC_THESIS_SOURCE")

    split_lock=load(ROOT/"data/manifests/split_lock_manifest.json")
    split={n:coco(ROOT/f"data/processed/coco/instances_{n}.json") for n in ("train","val","test")}
    exp_n={"train":3426,"val":734,"test":734}; exp_nf={"train":350,"val":75,"test":75}
    for n,x in split.items():
        check(f"split:{n}:count",x["images"]==exp_n[n],exp_n[n],x["images"],x["path"],checks)
        check(f"split:{n}:zero_gt",x["zero_gt"]==exp_nf[n],exp_nf[n],x["zero_gt"],x["path"],checks)
        check(f"split:{n}:file_sha",x["sha256"]==split_lock["coco_json_sha256"][n],split_lock["coco_json_sha256"][n],x["sha256"],"split_lock_manifest.json",checks)
        check(f"split:{n}:membership",x["phase2e_membership_sha256"]==split_lock["image_id_sha256"][n],split_lock["image_id_sha256"][n],x["phase2e_membership_sha256"],"split_lock_manifest.json",checks)
    ids=[split[n]["ids"] for n in ("train","val","test")]
    check("split:pairwise_overlap",not(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]),0,[len(ids[0]&ids[1]),len(ids[0]&ids[2]),len(ids[1]&ids[2])],"actual COCO membership",checks)
    master=coco(ROOT/"data/processed/coco/coco_master_jpg.json")
    check("split:union",set.union(*ids)==master["ids"],master["images"],len(set.union(*ids)),"coco_master_jpg.json",checks)

    names=["Aortic enlargement","Atelectasis","Calcification","Cardiomegaly","Consolidation","ILD","Infiltration","Lung Opacity","Nodule/Mass","Other lesion","Pleural effusion","Pleural thickening","Pneumothorax","Pulmonary fibrosis"]
    expected_cats=[{"id":i+1,"name":n,"canonical_class_id":i} for i,n in enumerate(names)]
    def category_signature(cats): return [{"id":c.get("id"),"name":c.get("name"),"canonical_class_id":c.get("canonical_class_id")} for c in cats]
    for n,x in {"master":master,**split}.items(): check(f"categories:{n}",category_signature(x["categories"])==expected_cats,expected_cats,category_signature(x["categories"]),x["path"],checks)

    lock=load(ROOT/"data/manifests/phase2F_lock_manifest.json"); labeled={}; unlabeled={}
    for b,(nl,nu,nnf) in BUDGETS.items():
        l=coco(ROOT/f"data/processed/coco/labeled_splits/instances_labeled_{b}.json"); u=coco(ROOT/f"data/processed/coco/unlabeled_splits/instances_unlabeled_{b}.json"); labeled[b]=l; unlabeled[b]=u
        check(f"lu:{b}:sizes",(l["images"],u["images"],l["zero_gt"])==(nl,nu,nnf),(nl,nu,nnf),(l["images"],u["images"],l["zero_gt"]),"actual L/U JSON",checks)
        check(f"lu:{b}:partition",not(l["ids"]&u["ids"]) and l["ids"]|u["ids"]==split["train"]["ids"],"disjoint exact train partition",[len(l["ids"]&u["ids"]),len(l["ids"]|u["ids"])],"actual membership",checks)
        check(f"lu:{b}:hidden_gt",u["annotations"]==0,0,u["annotations"],u["path"],checks)
        check(f"lu:{b}:membership",l["phase2f_membership_sha256"]==lock["labeled_image_id_sha256"][b],lock["labeled_image_id_sha256"][b],l["phase2f_membership_sha256"],"phase2F_lock_manifest.json",checks)
        check(f"lu:{b}:file_sha",l["sha256"]==lock["coco_json_sha256"]["labeled"][b] and u["sha256"]==lock["coco_json_sha256"]["unlabeled"][b],"locked hashes",[l["sha256"],u["sha256"]],"phase2F_lock_manifest.json",checks)
        check(f"categories:labeled:{b}",category_signature(l["categories"])==expected_cats,expected_cats,category_signature(l["categories"]),l["path"],checks)
        check(f"categories:unlabeled:{b}",category_signature(u["categories"])==expected_cats,expected_cats,category_signature(u["categories"]),u["path"],checks)
    order=list(BUDGETS); check("lu:nested",all(labeled[order[i]]["ids"] < labeled[order[i+1]]["ids"] for i in range(3)),True,[len(labeled[b]["ids"]) for b in order],"actual labeled membership",checks)

    seed_text=(ROOT/"configs/protocol/phase2F1_seed_protocol.yaml").read_text(encoding="utf-8")
    seeds=[204886845,1480646854,1798418854,2045683682,1814859839,1603952859,1878351743,875651179,477581743,869675675]
    derived=[]
    for i in range(1,11): derived.append(1+(int(hashlib.sha256(f"ssl_detection_xray_v2|phase2F.1|training_seed|index={i}".encode()).hexdigest()[:8],16)%(2**31-1)))
    check("seed:ordered_derivation",derived==seeds,seeds,derived,"phase2F1_seed_protocol.yaml",checks)
    check("seed:policies",all(t in seed_text for t in ["partition_seed: 42","PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH","CONTROLLED_BEST_EFFORT","pairing_key: training_seed_index"]),True,"tokens scanned","phase2F1_seed_protocol.yaml",checks)

    rep=ROOT/"configs/protocol/phase2D1_jpg_representation.yaml"; jpg_root=ROOT/"data/processed/images_jpg/train"; jpg_count=sum(1 for _ in jpg_root.glob("*.jpg"))
    check("representation:active_protocol",rep.exists(),True,rep.exists(),str(rep.relative_to(ROOT)),checks)
    check("representation:inventory",jpg_count==4894,4894,jpg_count,str(jpg_root.relative_to(ROOT)),checks)
    check("representation:semantics",all(t in rep.read_text(encoding="utf-8") for t in ["final_quality: 95","jpeg_mode: L","replicate_grayscale_in_loader: true","resize: false","crop: false"]),True,"tokens scanned",str(rep.relative_to(ROOT)),checks)

    for p in ["canonical_protocol_index.yaml","experimental_design.yaml","d4_training_protocol.yaml","d4_ssl_protocol.yaml","gateI_freeze_manifest.yaml","gateI_environment_baseline.yaml"]: check("canonical:"+p,(ROOT/"configs/protocol"/p).exists(),True,(ROOT/"configs/protocol"/p).exists(),"configs/protocol",checks)

    env_start=len(checks); env_yaml_path=ROOT/"configs/protocol/gateI_environment_baseline.yaml"; env_text=env_yaml_path.read_text(encoding="utf-8")
    evidence={"gpu_runtime":ROOT/"reports/environment/gateI_gpu_runtime.txt","pip_freeze":ROOT/"reports/environment/gateI_official_gpu_pip_freeze.txt","system_packages":ROOT/"reports/environment/gateI_system_packages.txt"}
    expected_hash={name:yaml_provenance_hash(env_text,name) for name in evidence}
    for name,path in evidence.items():
        check(f"environment:evidence:{name}:exists",path.is_file(),True,path.is_file(),str(path.relative_to(ROOT)),checks)
        actual=sha(path) if path.is_file() else None
        check(f"environment:evidence:{name}:sha256",expected_hash[name] is not None and actual==expected_hash[name],expected_hash[name],actual,str(env_yaml_path.relative_to(ROOT)),checks)
    runtime=key_values(evidence["gpu_runtime"]) if evidence["gpu_runtime"].is_file() else {}; pip_text=evidence["pip_freeze"].read_text(encoding="utf-8") if evidence["pip_freeze"].is_file() else ""
    runtime_expected={"GPU":"NVIDIA A800 80GB PCIe","GPU_COUNT":"1","NVIDIA_DRIVER":"550.90.07","PYTHON":"3.10.13","PYTORCH":"2.1.0","TORCHVISION":"0.16.0","PYTORCH_CUDA_RUNTIME":"11.8","CUDNN":"8.7.0","NUMPY":"1.26.4","OPENCV_RUNTIME":"4.10.0","MMCV":"2.1.0","MMENGINE":"0.10.7","MMDETECTION":"3.3.0","DOCKER_IMAGE":"pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime","CUDA_AVAILABLE":"true","CUDA_TENSOR_SMOKE_TEST":"PASS","MMCV_CUDA_NMS":"PASS","PIP_CHECK":"PASS","OFFICIAL_TRAINING_EXECUTED":"false","TEST_PERFORMANCE_READ":"false","NVIDIA_SMI_CUDA_MAX_SUPPORTED":"12.4"}
    for key,expected in runtime_expected.items(): check(f"environment:runtime:{key}",runtime.get(key)==expected,expected,runtime.get(key),str(evidence["gpu_runtime"].relative_to(ROOT)),checks)
    pip_expected={"torch":"2.1.0","torchvision":"0.16.0","numpy":"1.26.4","opencv-python":"4.10.0.84","mmcv":"2.1.0","mmengine":"0.10.7","mmdet":"3.3.0"}
    for pkg,ver in pip_expected.items(): check(f"environment:pip:{pkg}",re.search(rf"(?m)^{re.escape(pkg)}=={re.escape(ver)}$",pip_text) is not None,ver,"present" if re.search(rf"(?m)^{re.escape(pkg)}=={re.escape(ver)}$",pip_text) else "missing",str(evidence["pip_freeze"].relative_to(ROOT)),checks)
    yaml_required={"status":"FROZEN","provider":"Vast.ai","model":"NVIDIA A800 80GB PCIe","device_count":"1","driver_version":"550.90.07","nvidia_smi_cuda_max_supported":"12.4","cuda_runtime":"11.8","cudnn":"8.7.0","opencv_python":"4.10.0.84","image":"pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime","cuda_available":"true","official_training_executed":"false","test_performance_read":"false","training_authorized":"false"}
    for key,expected in yaml_required.items(): check(f"environment:yaml:{key}",yaml_scalar(env_text,key)==expected,expected,yaml_scalar(env_text,key),str(env_yaml_path.relative_to(ROOT)),checks)
    yaml_profile_tokens=['python:\n    version: "3.10.13"','pytorch:\n    version: "2.1.0"','torchvision: "0.16.0"','numpy: "1.26.4"','mmcv: "2.1.0"','mmengine: "0.10.7"','mmdetection: "3.3.0"','cuda_tensor_smoke_test: PASS','mmcv_cuda_nms: PASS','pip_check: PASS']
    check("environment:yaml:complete_profile",all(t in env_text for t in yaml_profile_tokens),yaml_profile_tokens,[t for t in yaml_profile_tokens if t not in env_text],str(env_yaml_path.relative_to(ROOT)),checks)
    env_failed=any(c["status"]=="FAIL" for c in checks[env_start:])
    if env_failed: blockers.append("OFFICIAL_GPU_ENVIRONMENT_PROVENANCE_MISMATCH")
    failed=[c for c in checks if c["status"]=="FAIL"]
    result={"schema_version":"1.0.0","timestamp_utc":datetime.now(timezone.utc).isoformat(),"repository":{"path":str(ROOT),"branch":git("branch","--show-current"),"head":git("rev-parse","HEAD"),"status":git("status","--short")},"authority":authority,"representation_protocol_sha256":sha(rep),"environment":{"local_audit":{"os":platform.platform(),"python":sys.version.split()[0]},"official_gpu":{"provider":"Vast.ai","gpu":runtime.get("GPU"),"pytorch_cuda_runtime":runtime.get("PYTORCH_CUDA_RUNTIME"),"nvidia_smi_cuda_max_supported":runtime.get("NVIDIA_SMI_CUDA_MAX_SUPPORTED"),"provenance_sha256":expected_hash}},"checks":checks,"summary":{"passed":len(checks)-len(failed),"failed":len(failed),"total":len(checks)},"blockers":sorted(set(blockers)),"gate_i_status":"BLOCKED / FAIL" if failed or blockers else "CLOSED / PASS","gate_ii_authorized":False if failed or blockers else True,"official_training_authorized":False,"test_status":"CLOSED / STRUCTURAL-INTEGRITY-ONLY","test_performance_was_read":False,"official_training_was_executed":False}
    if not args.no_write:
        out=ROOT/args.json_output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2,ensure_ascii=False)); return 1 if failed or blockers else 0
if __name__=="__main__": raise SystemExit(main())
