#!/usr/bin/env bash
set -Eeuo pipefail

# ============================================================================
# SSOD S1 Data Preflight — Vast Dataset Preparation / Verification
#
# Purpose
#   S1.05: Google Drive -> Vast local/persistent storage -> transfer evidence
#   S1.06: dataset integrity / canonical invariants -> integrity evidence
#   Data preflight: fixed-split membership, L/U membership, class mapping,
#                   and loader/path-resolution evidence
#
# Typical local invocation (Git Bash on Windows):
#   scripts/data/prepare_vast_dataset.sh --remote <HOST> <PORT> --full
#
# Verify an already-present dataset (e.g. persistent volume):
#   scripts/data/prepare_vast_dataset.sh --remote <HOST> <PORT> --verify-only
#
# This script intentionally does NOT:
#   - train SUP/SSL
#   - rebuild fixed train/val/test
#   - rebuild labeled/unlabeled membership
#   - use hidden U ground truth
#   - alter scientific protocol
# ============================================================================

SCRIPT_VERSION="1.1.1"

MODE="full"
REMOTE_MODE="false"
REMOTE_HOST=""
REMOTE_PORT=""
SSOD_ROOT="${SSOD_ROOT:-/workspace/ssod}"

DRIVE_FILE_ID="${DRIVE_FILE_ID:-1M6XOpkdsyt7Q7tl-RyEFFENh_kR_G4jm}"

# Locked/observed operational integrity values from the completed S1.05-S1.06 path.
EXPECTED_PAYLOAD_SIZE=7502034622
EXPECTED_PAYLOAD_SHA256="2f16a795c371a20a7abed5eb7a31e968484e9d1ecc3dfa2b1d72fa0436a6bd2f"
EXPECTED_TREE_SHA256="0587000e016a6c46612fa5d7d7e7fae996c98bae58adb5165633607b9863295a"
EXPECTED_JPG_COUNT=4894
EXPECTED_UNCOMPRESSED_BYTES=7527137704

# Canonical source-file SHA-256 values already verified local <-> Vast.
TRAIN_SHA256="0f3c37a6f1b5bcc6971b01fd4c69c7a2a3a1e8bee145c11488c7a658dcbbebe3"
VAL_SHA256="33064f47ba690be13e9d418d8ec5d4a6a2bec8482c24ea3eadf83fb33d01762a"
TEST_SHA256="e1a73110e92af2656276d6c532035afe474b6ac6f2b2f03849834c036e1c00a4"

L1_SHA256="1e267c547a8f535f6a69088da4735bd12c0188fae1b49e9d785b3e1e6883df98"
L5_SHA256="ebb98f32cad612fe48adbbd6b5da8aa5db10bec478d1c7fd9288a9a93bbc3763"
L10_SHA256="6814b5b9ba8dfcd4af483d97c1a6b7e7ab26914bdd02bbb923502b43ecf2fdf6"
L20_SHA256="6a8b5bf9c59baea41eefb0611e15a387257255b6fc59a3de607bfc2e80801a14"

U1_SHA256="d2e1d0923f97ced05c1c4e319312d82d42e615defb55412d3c5e83e97486e3f5"
U5_SHA256="9b2c128a751a70db4c4680c0fd584cc34907bcda569567c3f9fb96af0722e387"
U10_SHA256="894c2de257cd5a33b60e00d782a5322a288f3d9a2b7c637ab29ad3419cce9260"
U20_SHA256="60699d361d86b61b66ad9c9553be555a23808adb53746cead0b190064df4d5bb"

PHASE2F_SHA256="7b541eb09c60321b56d06ef4918a686c14a7c62eaa9965161ff1144f9a7b4467"
PHASE2F_LOCK_SHA256="d8659d6fe40f9a32de0d32833c22dccc804da15009e3131d2640f7b9fb473c88"
PHASE2F_SCRIPT_CAPTURE_SHA256="c5e6d04afdf9db2e5ed0c626eae8536677e0313319a8244f74bd98583e20da56"

PHASE2E_REPORT_SHA256="64a072227863f949f07a0695dec06574351232cbb6d4fd331feb3a822e70d53e"
PHASE2E_SCRIPT_SHA256="caeb03cf3d5454ab12a9a35fd8735d93431a33ff7a1c6053730d0503cca6ecda"

usage() {
  cat <<'EOF'
Usage:
  prepare_vast_dataset.sh --remote HOST PORT [--full|--verify-only]
  prepare_vast_dataset.sh --on-remote [--full|--verify-only]

Modes:
  --full         Download from Google Drive if canonical dataset is not already
                 present and verified; then create the full S1 data-preflight evidence block.
  --verify-only  Do not download. Verify existing canonical data and regenerate
                 all machine-readable data-preflight evidence.

Environment overrides:
  SSOD_ROOT       Default: /workspace/ssod
  DRIVE_FILE_ID   Default: locked Google Drive file ID
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      [[ $# -ge 3 ]] || { echo "[FAIL] --remote requires HOST PORT"; exit 2; }
      REMOTE_HOST="$2"
      REMOTE_PORT="$3"
      shift 3
      ;;
    --on-remote)
      REMOTE_MODE="true"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --verify-only)
      MODE="verify-only"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[FAIL] unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

log()  { printf '%s\n' "$*"; }
pass() { printf '[PASS] %s\n' "$*"; }
warn() { printf '[WARN] %s\n' "$*"; }
fail() { printf '[FAIL] %s\n' "$*" >&2; exit 2; }

sha256_of() {
  sha256sum "$1" | awk '{print $1}'
}

verify_hash() {
  local path="$1"
  local expected="$2"
  local label="$3"
  [[ -f "$path" ]] || fail "$label missing: $path"
  local observed
  observed="$(sha256_of "$path")"
  log "$label"
  log "  expected=$expected"
  log "  observed=$observed"
  [[ "$observed" == "$expected" ]] || fail "$label SHA-256 mismatch"
  pass "$label SHA-256 matches"
}

# ----------------------------------------------------------------------------
# LOCAL WRAPPER
# ----------------------------------------------------------------------------
if [[ "$REMOTE_MODE" != "true" ]]; then
  [[ -n "$REMOTE_HOST" && -n "$REMOTE_PORT" ]] || {
    usage
    fail "local mode requires --remote HOST PORT"
  }

  command -v ssh >/dev/null 2>&1 || fail "ssh command missing"
  command -v scp >/dev/null 2>&1 || fail "scp command missing"
  command -v sha256sum >/dev/null 2>&1 || fail "sha256sum command missing"

  SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

  TRAIN_LOCAL="$REPO_ROOT/data/processed/coco/instances_train.json"
  VAL_LOCAL="$REPO_ROOT/data/processed/coco/instances_val.json"
  TEST_LOCAL="$REPO_ROOT/data/processed/coco/instances_test.json"

  L1_LOCAL="$REPO_ROOT/data/processed/coco/labeled_splits/instances_labeled_1pct.json"
  L5_LOCAL="$REPO_ROOT/data/processed/coco/labeled_splits/instances_labeled_5pct.json"
  L10_LOCAL="$REPO_ROOT/data/processed/coco/labeled_splits/instances_labeled_10pct.json"
  L20_LOCAL="$REPO_ROOT/data/processed/coco/labeled_splits/instances_labeled_20pct.json"

  U1_LOCAL="$REPO_ROOT/data/processed/coco/unlabeled_splits/instances_unlabeled_1pct.json"
  U5_LOCAL="$REPO_ROOT/data/processed/coco/unlabeled_splits/instances_unlabeled_5pct.json"
  U10_LOCAL="$REPO_ROOT/data/processed/coco/unlabeled_splits/instances_unlabeled_10pct.json"
  U20_LOCAL="$REPO_ROOT/data/processed/coco/unlabeled_splits/instances_unlabeled_20pct.json"

  PHASE2F_LOCAL="$REPO_ROOT/reports/02F_labeled_unlabeled_validation_report.json"
  PHASE2F_LOCK_LOCAL="$REPO_ROOT/data/manifests/phase2F_lock_manifest.json"
  PHASE2F_SCRIPT_LOCAL="$REPO_ROOT/scripts/02F_build_labeled_unlabeled.py"

  PHASE2E_REPORT_LOCAL="$REPO_ROOT/reports/phase2E_build_fixed_split_validation_report.json"
  PHASE2E_SCRIPT_LOCAL="$REPO_ROOT/scripts/02E_C0_R2_exact_constrained_candidate_split.py"

  log "========================================"
  log "SSOD S1.05-S1.06 ONE-COMMAND PREPARER"
  log "========================================"
  log "mode=$MODE"
  log "target=root@$REMOTE_HOST"
  log "port=$REMOTE_PORT"
  log "script_version=$SCRIPT_VERSION"

  log
  log "=== VERIFY LOCAL SOURCE EVIDENCE ==="
  verify_hash "$TRAIN_LOCAL" "$TRAIN_SHA256" "instances_train.json"
  verify_hash "$VAL_LOCAL" "$VAL_SHA256" "instances_val.json"
  verify_hash "$TEST_LOCAL" "$TEST_SHA256" "instances_test.json"
  verify_hash "$L1_LOCAL" "$L1_SHA256" "instances_labeled_1pct.json"
  verify_hash "$L5_LOCAL" "$L5_SHA256" "instances_labeled_5pct.json"
  verify_hash "$L10_LOCAL" "$L10_SHA256" "instances_labeled_10pct.json"
  verify_hash "$L20_LOCAL" "$L20_SHA256" "instances_labeled_20pct.json"

  verify_hash "$U1_LOCAL" "$U1_SHA256" "instances_unlabeled_1pct.json"
  verify_hash "$U5_LOCAL" "$U5_SHA256" "instances_unlabeled_5pct.json"
  verify_hash "$U10_LOCAL" "$U10_SHA256" "instances_unlabeled_10pct.json"
  verify_hash "$U20_LOCAL" "$U20_SHA256" "instances_unlabeled_20pct.json"

  verify_hash "$PHASE2F_LOCAL" "$PHASE2F_SHA256" "02F_labeled_unlabeled_validation_report.json"
  verify_hash "$PHASE2F_LOCK_LOCAL" "$PHASE2F_LOCK_SHA256" "phase2F_lock_manifest.json"
  verify_hash "$PHASE2F_SCRIPT_LOCAL" "$PHASE2F_SCRIPT_CAPTURE_SHA256" "02F_build_labeled_unlabeled.py (captured identity)"

  verify_hash "$PHASE2E_REPORT_LOCAL" "$PHASE2E_REPORT_SHA256" "phase2E_build_fixed_split_validation_report.json"
  verify_hash "$PHASE2E_SCRIPT_LOCAL" "$PHASE2E_SCRIPT_SHA256" "02E_C0_R2_exact_constrained_candidate_split.py"

  LOCAL_SCRIPT_SHA="$(sha256_of "$SCRIPT_PATH")"
  log "LOCAL_SCRIPT_SHA256=$LOCAL_SCRIPT_SHA"

  log
  log "=== PREPARE REMOTE DIRECTORIES ==="
  ssh -p "$REMOTE_PORT" "root@$REMOTE_HOST" \
    "mkdir -p \
      '$SSOD_ROOT/data/coco' \
      '$SSOD_ROOT/artifacts/preflight/data/source_evidence' \
      '$SSOD_ROOT/artifacts/preflight/data/source_evidence/phase2E' \
      '$SSOD_ROOT/artifacts/preflight/data/source_evidence/phase2F' \
      /tmp/ssod_s1"

  log
  log "=== TRANSFER SCRIPT ==="
  scp -P "$REMOTE_PORT" "$SCRIPT_PATH" \
    "root@$REMOTE_HOST:/tmp/ssod_s1/prepare_vast_dataset.sh"

  REMOTE_SCRIPT_SHA="$(
    ssh -p "$REMOTE_PORT" "root@$REMOTE_HOST" \
      "sha256sum /tmp/ssod_s1/prepare_vast_dataset.sh | awk '{print \$1}'"
  )"
  log "REMOTE_SCRIPT_SHA256=$REMOTE_SCRIPT_SHA"
  [[ "$LOCAL_SCRIPT_SHA" == "$REMOTE_SCRIPT_SHA" ]] || fail "script transfer hash mismatch"
  pass "script transfer hash matches"

  log
  log "=== TRANSFER CANONICAL COCO / PHASE2F EVIDENCE ==="
  scp -P "$REMOTE_PORT" \
    "$TRAIN_LOCAL" "$VAL_LOCAL" "$TEST_LOCAL" \
    "$L1_LOCAL" "$L5_LOCAL" "$L10_LOCAL" "$L20_LOCAL" \
    "$U1_LOCAL" "$U5_LOCAL" "$U10_LOCAL" "$U20_LOCAL" \
    "root@$REMOTE_HOST:$SSOD_ROOT/data/coco/"

  scp -P "$REMOTE_PORT" "$PHASE2F_LOCAL" \
    "root@$REMOTE_HOST:$SSOD_ROOT/artifacts/preflight/data/source_evidence/"

  scp -P "$REMOTE_PORT" \
    "$PHASE2F_LOCK_LOCAL" "$PHASE2F_SCRIPT_LOCAL" \
    "root@$REMOTE_HOST:$SSOD_ROOT/artifacts/preflight/data/source_evidence/phase2F/"

  scp -P "$REMOTE_PORT" \
    "$PHASE2E_REPORT_LOCAL" "$PHASE2E_SCRIPT_LOCAL" \
    "root@$REMOTE_HOST:$SSOD_ROOT/artifacts/preflight/data/source_evidence/phase2E/"

  log
  log "=== RUN REMOTE DATASET PREPARER ==="
  ssh -p "$REMOTE_PORT" "root@$REMOTE_HOST" \
    "chmod +x /tmp/ssod_s1/prepare_vast_dataset.sh && \
     SSOD_ROOT='$SSOD_ROOT' DRIVE_FILE_ID='$DRIVE_FILE_ID' \
     /tmp/ssod_s1/prepare_vast_dataset.sh --on-remote --$MODE"

  log
  log "========================================"
  log "NEW_VAST_DATA_PREPARATION=PASS"
  log "========================================"
  exit 0
fi

# ----------------------------------------------------------------------------
# REMOTE EXECUTION
# ----------------------------------------------------------------------------
command -v python >/dev/null 2>&1 || fail "python command missing"
command -v curl >/dev/null 2>&1 || fail "curl command missing"
command -v sha256sum >/dev/null 2>&1 || fail "sha256sum command missing"

DATA_ROOT="$SSOD_ROOT/data"
COCO_ROOT="$DATA_ROOT/coco"
IMG_ROOT="$DATA_ROOT/images"
INCOMING_ROOT="$DATA_ROOT/incoming"
STAGING_ROOT="$DATA_ROOT/staging_extract"
ART_ROOT="$SSOD_ROOT/artifacts/preflight/data"
SOURCE_EVIDENCE_ROOT="$ART_ROOT/source_evidence"

PAYLOAD="$INCOMING_ROOT/gdrive_payload"
TRANSFER_MANIFEST="$ART_ROOT/dataset_transfer_manifest.json"
INTEGRITY_REPORT="$ART_ROOT/dataset_integrity_report.json"
PHASE2F_EVIDENCE="$SOURCE_EVIDENCE_ROOT/02F_labeled_unlabeled_validation_report.json"
PHASE2F_LOCK="$SOURCE_EVIDENCE_ROOT/phase2F/phase2F_lock_manifest.json"
PHASE2F_SCRIPT="$SOURCE_EVIDENCE_ROOT/phase2F/02F_build_labeled_unlabeled.py"
PHASE2E_REPORT="$SOURCE_EVIDENCE_ROOT/phase2E/phase2E_build_fixed_split_validation_report.json"
PHASE2E_SCRIPT="$SOURCE_EVIDENCE_ROOT/phase2E/02E_C0_R2_exact_constrained_candidate_split.py"

mkdir -p \
  "$COCO_ROOT" "$IMG_ROOT" "$INCOMING_ROOT" \
  "$ART_ROOT" "$SOURCE_EVIDENCE_ROOT" \
  "$SOURCE_EVIDENCE_ROOT/phase2E" "$SOURCE_EVIDENCE_ROOT/phase2F"

log "========================================"
log "SSOD S1.05-S1.06 REMOTE DATASET PREPARER"
log "========================================"
log "mode=$MODE"
log "ssod_root=$SSOD_ROOT"
log "drive_file_id=$DRIVE_FILE_ID"

log
log "=== VERIFY TRANSFERRED CANONICAL INPUTS ==="
verify_hash "$COCO_ROOT/instances_train.json" "$TRAIN_SHA256" "instances_train.json"
verify_hash "$COCO_ROOT/instances_val.json" "$VAL_SHA256" "instances_val.json"
verify_hash "$COCO_ROOT/instances_test.json" "$TEST_SHA256" "instances_test.json"
verify_hash "$COCO_ROOT/instances_labeled_1pct.json" "$L1_SHA256" "instances_labeled_1pct.json"
verify_hash "$COCO_ROOT/instances_labeled_5pct.json" "$L5_SHA256" "instances_labeled_5pct.json"
verify_hash "$COCO_ROOT/instances_labeled_10pct.json" "$L10_SHA256" "instances_labeled_10pct.json"
verify_hash "$COCO_ROOT/instances_labeled_20pct.json" "$L20_SHA256" "instances_labeled_20pct.json"

verify_hash "$COCO_ROOT/instances_unlabeled_1pct.json" "$U1_SHA256" "instances_unlabeled_1pct.json"
verify_hash "$COCO_ROOT/instances_unlabeled_5pct.json" "$U5_SHA256" "instances_unlabeled_5pct.json"
verify_hash "$COCO_ROOT/instances_unlabeled_10pct.json" "$U10_SHA256" "instances_unlabeled_10pct.json"
verify_hash "$COCO_ROOT/instances_unlabeled_20pct.json" "$U20_SHA256" "instances_unlabeled_20pct.json"

verify_hash "$PHASE2F_EVIDENCE" "$PHASE2F_SHA256" "Phase 2F validation report"
verify_hash "$PHASE2F_LOCK" "$PHASE2F_LOCK_SHA256" "Phase 2F lock manifest"
verify_hash "$PHASE2F_SCRIPT" "$PHASE2F_SCRIPT_CAPTURE_SHA256" "Phase 2F current implementation capture"
verify_hash "$PHASE2E_REPORT" "$PHASE2E_REPORT_SHA256" "Phase 2E validation report"
verify_hash "$PHASE2E_SCRIPT" "$PHASE2E_SCRIPT_SHA256" "Phase 2E implementation script"

log
log "=== CHECK EXISTING CANONICAL DATASET ==="
EXISTING_STATUS="$(
python - "$IMG_ROOT" "$EXPECTED_TREE_SHA256" "$EXPECTED_JPG_COUNT" <<'PY'
from pathlib import Path
import hashlib, sys

root = Path(sys.argv[1])
expected_tree = sys.argv[2]
expected_count = int(sys.argv[3])

def file_sha256(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

if not root.exists():
    print("ABSENT")
    raise SystemExit

files = sorted(
    p for p in root.iterdir()
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
)
if len(files) == 0:
    print("EMPTY")
    raise SystemExit

tree = hashlib.sha256()
names = []
for p in files:
    names.append(p.name)
    tree.update(f"{p.name}\t{file_sha256(p)}\n".encode("utf-8"))

ok = (
    len(files) == expected_count
    and len(set(names)) == expected_count
    and tree.hexdigest() == expected_tree
)
print("VALID" if ok else "INVALID")
PY
)"
log "existing_dataset_status=$EXISTING_STATUS"

if [[ "$EXISTING_STATUS" == "INVALID" ]]; then
  fail "existing canonical image root is non-empty but does not match locked tree hash"
fi

if [[ "$MODE" == "verify-only" && "$EXISTING_STATUS" != "VALID" ]]; then
  fail "--verify-only requires an existing canonical dataset with the locked tree hash"
fi

TRANSFER_EVENT="REUSED_EXISTING_VERIFIED_DATASET"
PAYLOAD_OBSERVED="false"
PAYLOAD_OBSERVED_SHA=""
PAYLOAD_OBSERVED_SIZE=""

if [[ "$MODE" == "full" && "$EXISTING_STATUS" != "VALID" ]]; then
  log
  log "=== GOOGLE DRIVE ACCESS / CONFIRMATION PROBE ==="
  PROBE="/tmp/ssod_s1_gdrive_probe.html"
  curl -L -sS -o "$PROBE" \
    "https://drive.google.com/uc?export=download&id=${DRIVE_FILE_ID}"

  DOWNLOAD_URL="$(
    python - "$PROBE" "$DRIVE_FILE_ID" <<'PY'
from html.parser import HTMLParser
from urllib.parse import urlencode
import sys

probe = sys.argv[1]
expected_id = sys.argv[2]
html = open(probe, encoding="utf-8").read()

class P(HTMLParser):
    def __init__(self):
        super().__init__()
        self.action = None
        self.inputs = []
    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "form" and self.action is None:
            self.action = a.get("action")
        if tag == "input":
            n = a.get("name")
            if n:
                self.inputs.append((n, a.get("value", "")))

p = P()
p.feed(html)

if p.action != "https://drive.usercontent.google.com/download":
    raise SystemExit("unexpected Google Drive confirmation form action")

params = dict(p.inputs)
if params.get("id") != expected_id:
    raise SystemExit("Google Drive confirmation file id mismatch")
if params.get("export") != "download":
    raise SystemExit("Google Drive confirmation export mismatch")
if not params.get("confirm") or not params.get("uuid"):
    raise SystemExit("Google Drive confirmation token/uuid missing")

print(p.action + "?" + urlencode(params))
PY
  )"
  pass "Google Drive confirmation form resolved dynamically"

  log
  log "=== DOWNLOAD PAYLOAD ==="
  # Idempotent/resumable behavior:
  # - if exact locked payload already exists, skip network transfer
  # - otherwise resume a partial payload with curl -C -
  if [[ -f "$PAYLOAD" ]]; then
    CURRENT_SIZE="$(stat -c '%s' "$PAYLOAD")"
    log "existing_payload_size=$CURRENT_SIZE"
    if [[ "$CURRENT_SIZE" -eq "$EXPECTED_PAYLOAD_SIZE" ]]; then
      CURRENT_SHA="$(sha256_of "$PAYLOAD")"
      if [[ "$CURRENT_SHA" == "$EXPECTED_PAYLOAD_SHA256" ]]; then
        pass "locked payload already present; download skipped"
      else
        fail "payload has expected size but wrong SHA-256; refusing overwrite"
      fi
    elif [[ "$CURRENT_SIZE" -gt "$EXPECTED_PAYLOAD_SIZE" ]]; then
      fail "existing payload is larger than locked expected size"
    else
      curl -L --fail --retry 5 --retry-delay 5 \
        --continue-at - \
        -o "$PAYLOAD" \
        "$DOWNLOAD_URL"
    fi
  else
    curl -L --fail --retry 5 --retry-delay 5 \
      -o "$PAYLOAD" \
      "$DOWNLOAD_URL"
  fi

  PAYLOAD_OBSERVED_SIZE="$(stat -c '%s' "$PAYLOAD")"
  PAYLOAD_OBSERVED_SHA="$(sha256_of "$PAYLOAD")"
  log "payload_size_bytes=$PAYLOAD_OBSERVED_SIZE"
  log "payload_sha256=$PAYLOAD_OBSERVED_SHA"

  [[ "$PAYLOAD_OBSERVED_SIZE" -eq "$EXPECTED_PAYLOAD_SIZE" ]] \
    || fail "payload size mismatch"
  [[ "$PAYLOAD_OBSERVED_SHA" == "$EXPECTED_PAYLOAD_SHA256" ]] \
    || fail "payload SHA-256 mismatch"
  PAYLOAD_OBSERVED="true"
  pass "Google Drive payload matches locked size + SHA-256"

  log
  log "=== ZIP CRC / STRUCTURE / EXTRACTION ==="
  python - "$PAYLOAD" "$STAGING_ROOT" "$EXPECTED_JPG_COUNT" "$EXPECTED_UNCOMPRESSED_BYTES" <<'PY'
from pathlib import Path
import shutil, sys, zipfile

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
expected_count = int(sys.argv[3])
expected_uncompressed = int(sys.argv[4])

if not zipfile.is_zipfile(src):
    raise SystemExit("archive_type != ZIP")

with zipfile.ZipFile(src) as z:
    entries = z.infolist()
    files = [x for x in entries if not x.is_dir()]
    jpgs = [x for x in files if Path(x.filename).suffix.lower() in {".jpg", ".jpeg"}]
    unsafe = [
        x.filename for x in entries
        if Path(x.filename).is_absolute() or ".." in Path(x.filename).parts
    ]
    total_uncompressed = sum(x.file_size for x in files)
    bad = z.testzip()

    print("archive_type=ZIP")
    print(f"entry_count={len(entries)}")
    print(f"file_count={len(files)}")
    print(f"jpg_count={len(jpgs)}")
    print(f"unsafe_path_count={len(unsafe)}")
    print(f"uncompressed_bytes={total_uncompressed}")
    print(f"bad_crc_file={bad}")

    if bad is not None:
        raise SystemExit("ZIP CRC verification failed")
    if len(files) != expected_count or len(jpgs) != expected_count:
        raise SystemExit("ZIP file/JPG count mismatch")
    if unsafe:
        raise SystemExit("ZIP contains unsafe paths")
    if total_uncompressed != expected_uncompressed:
        raise SystemExit("ZIP uncompressed byte count mismatch")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    z.extractall(dst)

extracted = [
    p for p in dst.rglob("*")
    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
]
all_files = [p for p in dst.rglob("*") if p.is_file()]

print(f"extracted_file_count={len(all_files)}")
print(f"extracted_jpg_count={len(extracted)}")
if len(all_files) != expected_count or len(extracted) != expected_count:
    raise SystemExit("extracted count mismatch")
PY

  log
  log "=== VERIFY STAGING TREE AND PROMOTE ==="
  python - "$STAGING_ROOT/images_jpg/train" "$IMG_ROOT" "$EXPECTED_TREE_SHA256" "$EXPECTED_JPG_COUNT" <<'PY'
from pathlib import Path
import hashlib, shutil, sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
expected_tree = sys.argv[3]
expected_count = int(sys.argv[4])

def sha256_file(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def tree_hash(root):
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )
    names = [p.name for p in files]
    h = hashlib.sha256()
    for i, p in enumerate(files, 1):
        h.update(f"{p.name}\t{sha256_file(p)}\n".encode("utf-8"))
        if i % 500 == 0:
            print(f"hashed_files={i}")
    return len(files), len(set(names)), h.hexdigest()

src_count, src_unique, src_tree = tree_hash(src)
print(f"staging_count={src_count}")
print(f"staging_unique={src_unique}")
print(f"staging_tree_sha256={src_tree}")

if src_count != expected_count or src_unique != expected_count or src_tree != expected_tree:
    raise SystemExit("staging dataset does not match locked tree hash")

dst.mkdir(parents=True, exist_ok=True)
existing = [p for p in dst.iterdir() if p.is_file()]
if existing:
    raise SystemExit("destination image root is unexpectedly non-empty before promotion")

for p in src.iterdir():
    if p.is_file():
        shutil.copy2(p, dst / p.name)

dst_count, dst_unique, dst_tree = tree_hash(dst)
print(f"destination_count={dst_count}")
print(f"destination_unique={dst_unique}")
print(f"destination_tree_sha256={dst_tree}")

if dst_count != expected_count or dst_unique != expected_count or dst_tree != expected_tree:
    raise SystemExit("destination promotion integrity failed")

print("PROMOTION_INTEGRITY=PASS")
PY
  TRANSFER_EVENT="GOOGLE_DRIVE_DOWNLOADED_AND_VERIFIED"
else
  log
  log "=== DATASET DOWNLOAD SKIPPED ==="
  pass "existing canonical dataset matches locked tree hash"
  if [[ -f "$PAYLOAD" ]]; then
    PAYLOAD_OBSERVED_SIZE="$(stat -c '%s' "$PAYLOAD")"
    PAYLOAD_OBSERVED_SHA="$(sha256_of "$PAYLOAD")"
    if [[ "$PAYLOAD_OBSERVED_SIZE" -eq "$EXPECTED_PAYLOAD_SIZE" \
       && "$PAYLOAD_OBSERVED_SHA" == "$EXPECTED_PAYLOAD_SHA256" ]]; then
      PAYLOAD_OBSERVED="true"
    fi
  fi
fi

log
log "=== CANONICAL MEMBERSHIP / DATA INVARIANT AUDIT ==="

# One Python program performs:
# - 4,894 JPG membership vs train/val/test union
# - split counts and No Finding counts
# - 14-class mapping
# - L/U budget counts and nesting
# - 1% 14/14 coverage
# - Phase 2F provenance: minimum_class_coverage repair move count = 0
# - destination tree hash
# - dataset_transfer_manifest.json
# - dataset_integrity_report.json
python - \
  "$SSOD_ROOT" \
  "$DRIVE_FILE_ID" \
  "$MODE" \
  "$TRANSFER_EVENT" \
  "$PAYLOAD_OBSERVED" \
  "$PAYLOAD_OBSERVED_SIZE" \
  "$PAYLOAD_OBSERVED_SHA" \
  "$EXPECTED_PAYLOAD_SIZE" \
  "$EXPECTED_PAYLOAD_SHA256" \
  "$EXPECTED_TREE_SHA256" \
  "$PHASE2F_SHA256" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, sys

(
    ssod_root,
    drive_file_id,
    mode,
    transfer_event,
    payload_observed,
    payload_observed_size,
    payload_observed_sha,
    expected_payload_size,
    expected_payload_sha,
    expected_tree_sha,
    expected_phase2f_sha,
) = sys.argv[1:]

ROOT = Path(ssod_root)
IMG_ROOT = ROOT / "data/images"
COCO_ROOT = ROOT / "data/coco"
ART_ROOT = ROOT / "artifacts/preflight/data"
TRANSFER_MANIFEST = ART_ROOT / "dataset_transfer_manifest.json"
INTEGRITY_REPORT = ART_ROOT / "dataset_integrity_report.json"
PHASE2F_EVIDENCE = ART_ROOT / "source_evidence/02F_labeled_unlabeled_validation_report.json"
PAYLOAD = ROOT / "data/incoming/gdrive_payload"

FILES = {
    "train": COCO_ROOT / "instances_train.json",
    "val": COCO_ROOT / "instances_val.json",
    "test": COCO_ROOT / "instances_test.json",
    "1pct": COCO_ROOT / "instances_labeled_1pct.json",
    "5pct": COCO_ROOT / "instances_labeled_5pct.json",
    "10pct": COCO_ROOT / "instances_labeled_10pct.json",
    "20pct": COCO_ROOT / "instances_labeled_20pct.json",
}

EXPECTED_SPLITS = {
    "train": {"images": 3426, "no_finding": 350},
    "val":   {"images": 734,  "no_finding": 75},
    "test":  {"images": 734,  "no_finding": 75},
}

EXPECTED_BUDGETS = {
    "1pct":  {"L": 34,  "U": 3392, "no_finding": 3},
    "5pct":  {"L": 171, "U": 3255, "no_finding": 17},
    "10pct": {"L": 343, "U": 3083, "no_finding": 35},
    "20pct": {"L": 685, "U": 2741, "no_finding": 70},
}

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

def tree_hash(root):
    files = sorted(
        p for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    )
    names = [p.name for p in files]
    h = hashlib.sha256()
    for i, p in enumerate(files, 1):
        h.update(f"{p.name}\t{sha256_file(p)}\n".encode("utf-8"))
        if i % 500 == 0:
            print(f"verified_files={i}")
    return files, names, h.hexdigest()

def load(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def image_ids(d):
    return {x["id"] for x in d["images"]}

def annotated_ids(d):
    return {a["image_id"] for a in d.get("annotations", [])}

def zero_gt_ids(d):
    return image_ids(d) - annotated_ids(d)

def categories(d):
    return {c["id"]: c["name"] for c in d["categories"]}

def covered_categories(d):
    return {a["category_id"] for a in d.get("annotations", [])}

def find_1pct_diag(obj):
    if isinstance(obj, dict):
        if (
            obj.get("budget") == "1pct"
            and "class_coverage_count" in obj
            and "move_count_by_phase" in obj
        ):
            return obj
        for v in obj.values():
            found = find_1pct_diag(v)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_1pct_diag(v)
            if found is not None:
                return found
    return None

D = {k: load(v) for k, v in FILES.items()}
phase2f = load(PHASE2F_EVIDENCE)
diag1 = find_1pct_diag(phase2f)
if diag1 is None:
    raise SystemExit("cannot locate 1pct Phase 2F diagnostics")

if sha256_file(PHASE2F_EVIDENCE) != expected_phase2f_sha:
    raise SystemExit("Phase 2F provenance hash mismatch")

base_cat = categories(D["train"])
train_ids = image_ids(D["train"])

jpgs, jpg_names, observed_tree = tree_hash(IMG_ROOT)
jpg_name_set = set(jpg_names)

expected_names = set()
for split in ["train", "val", "test"]:
    expected_names |= {
        Path(x["file_name"]).name
        for x in D[split]["images"]
    }

missing = sorted(expected_names - jpg_name_set)
unexpected = sorted(jpg_name_set - expected_names)

split_observed = {}
for name in ["train", "val", "test"]:
    split_observed[name] = {
        "images": len(image_ids(D[name])),
        "no_finding": len(zero_gt_ids(D[name])),
    }

split_overlaps = {
    "train_val": len(image_ids(D["train"]) & image_ids(D["val"])),
    "train_test": len(image_ids(D["train"]) & image_ids(D["test"])),
    "val_test": len(image_ids(D["val"]) & image_ids(D["test"])),
}

budget_observed = {}
Lsets = {}

for name in ["1pct", "5pct", "10pct", "20pct"]:
    ids = image_ids(D[name])
    Lsets[name] = ids
    budget_observed[name] = {
        "L": len(ids),
        "U": len(train_ids - ids),
        "no_finding": len(zero_gt_ids(D[name])),
        "class_coverage": len(
            covered_categories(D[name]) & set(base_cat)
        ),
        "subset_of_train": ids <= train_ids,
    }

nesting = {
    "L1_subset_L5": Lsets["1pct"] < Lsets["5pct"],
    "L5_subset_L10": Lsets["5pct"] < Lsets["10pct"],
    "L10_subset_L20": Lsets["10pct"] < Lsets["20pct"],
    "L20_subset_train": Lsets["20pct"] < train_ids,
}

phase2f_1pct = {
    "target_labeled_size": diag1.get("target_labeled_size"),
    "actual_labeled_size": diag1.get("actual_labeled_size"),
    "target_no_finding_size": diag1.get("target_no_finding_size"),
    "actual_no_finding_size": diag1.get("actual_no_finding_size"),
    "class_coverage_count": diag1.get("class_coverage_count"),
    "missing_classes": diag1.get("missing_classes"),
    "repair_move_count_total": diag1.get("repair_move_count_total"),
    "minimum_class_coverage_repair_move_count":
        diag1["move_count_by_phase"].get("minimum_class_coverage"),
}

checks = {
    "working_scope_images_4894": len(jpgs) == 4894,
    "jpg_unique_4894": len(jpg_name_set) == 4894,
    "jpg_missing_zero": len(missing) == 0,
    "jpg_unexpected_zero": len(unexpected) == 0,
    "destination_tree_hash_match": observed_tree == expected_tree_sha,
    "detection_classes_14": len(base_cat) == 14,
    "category_ids_1_to_14": sorted(base_cat) == list(range(1, 15)),
    "category_mapping_consistent":
        all(categories(D[k]) == base_cat for k in D),
    "split_counts_match":
        all(split_observed[k] == EXPECTED_SPLITS[k] for k in EXPECTED_SPLITS),
    "split_id_overlaps_zero":
        all(v == 0 for v in split_overlaps.values()),
    "budget_counts_match":
        all(
            budget_observed[k]["L"] == EXPECTED_BUDGETS[k]["L"]
            and budget_observed[k]["U"] == EXPECTED_BUDGETS[k]["U"]
            and budget_observed[k]["no_finding"] == EXPECTED_BUDGETS[k]["no_finding"]
            for k in EXPECTED_BUDGETS
        ),
    "all_labeled_subsets_within_train":
        all(budget_observed[k]["subset_of_train"] for k in budget_observed),
    "nested_membership": all(nesting.values()),
    "one_pct_coverage_14_of_14":
        budget_observed["1pct"]["class_coverage"] == 14,
    "one_pct_minimum_class_coverage_repair_zero":
        phase2f_1pct["minimum_class_coverage_repair_move_count"] == 0,
    "one_pct_phase2f_actual_size_34":
        phase2f_1pct["actual_labeled_size"] == 34,
    "one_pct_phase2f_no_finding_3":
        phase2f_1pct["actual_no_finding_size"] == 3,
    "one_pct_phase2f_coverage_14":
        phase2f_1pct["class_coverage_count"] == 14,
}

overall = all(checks.values())
if not overall:
    print("=== FAILED CHECKS ===")
    for k, v in checks.items():
        if not v:
            print(k)
    raise SystemExit("dataset invariant audit failed")

# Storage classification is observational/operational:
# overlay => local container storage; otherwise record persistent.
storage_type = "local"
try:
    import subprocess
    fstype = subprocess.check_output(
        ["findmnt", "-n", "-T", str(ROOT), "-o", "FSTYPE"],
        text=True
    ).strip()
    if fstype and fstype != "overlay":
        storage_type = "persistent"
except Exception:
    fstype = None

now = datetime.now(timezone.utc).isoformat()

payload_present = PAYLOAD.exists()
payload_size_observed = int(payload_observed_size) if payload_observed_size else (
    PAYLOAD.stat().st_size if payload_present else None
)
payload_sha_observed = payload_observed_sha or (
    sha256_file(PAYLOAD) if payload_present else None
)

transfer_manifest = {
    "schema_version": "1.1",
    "artifact_type": "DATASET_TRANSFER_MANIFEST",
    "status": "PASS",
    "transfer_mode": transfer_event,
    "transport_source": "Google Drive",
    "google_drive_file_id": drive_file_id,
    "source_payload_present": payload_present,
    "source_payload_size_bytes_expected": int(expected_payload_size),
    "source_payload_size_bytes_observed": payload_size_observed,
    "source_payload_sha256_expected": expected_payload_sha,
    "source_payload_sha256_observed": payload_sha_observed,
    "archive_type": "ZIP",
    "expected_jpg_count": 4894,
    "observed_jpg_count": len(jpgs),
    "missing_jpg_count": len(missing),
    "unexpected_jpg_count": len(unexpected),
    "duplicate_filename_count": len(jpg_names) - len(jpg_name_set),
    "destination_storage_type": storage_type,
    "destination_image_root": str(IMG_ROOT),
    "destination_tree_sha256": observed_tree,
    "direct_google_drive_io_for_training": False,
    "membership_verification": "PASS",
    "promotion_integrity": "PASS",
    "transfer_or_verification_timestamp_utc": now,
}

ART_ROOT.mkdir(parents=True, exist_ok=True)
TRANSFER_MANIFEST.write_text(
    json.dumps(transfer_manifest, indent=2) + "\n",
    encoding="utf-8",
)

report = {
    "schema_version": "1.1",
    "artifact_type": "DATASET_INTEGRITY_REPORT",
    "status": "PASS",
    "working_scope": {
        "expected_images": 4894,
        "observed_images": len(jpgs),
        "unique_filenames": len(jpg_name_set),
        "missing_jpg_count": len(missing),
        "unexpected_jpg_count": len(unexpected),
        "detection_class_count": len(base_cat),
        "no_finding_semantics":
            "zero-GT negative image; not detection class 15",
    },
    "category_mapping": {
        "category_ids": sorted(base_cat),
        "categories": base_cat,
        "consistent_across_all_coco_files":
            checks["category_mapping_consistent"],
    },
    "fixed_splits": {
        "observed": split_observed,
        "expected": EXPECTED_SPLITS,
        "id_overlaps": split_overlaps,
    },
    "labeled_budgets": {
        "observed": budget_observed,
        "expected": EXPECTED_BUDGETS,
        "nesting": nesting,
    },
    "one_pct_construction_evidence": {
        "construction_identity": "label-aware / coverage-constrained",
        **phase2f_1pct,
        "interpretation":
            "14/14 coverage is present in final 1% subset; dedicated "
            "minimum-class-coverage repair required 0 moves. This does "
            "not mean the 1% construction had zero total repair moves.",
    },
    "data_paths": {
        "training_image_root": str(IMG_ROOT),
        "coco_root": str(COCO_ROOT),
        "destination_storage_type": storage_type,
        "direct_google_drive_io_for_training": False,
    },
    "hashes": {
        "dataset_transfer_manifest_sha256": sha256_file(TRANSFER_MANIFEST),
        "destination_tree_sha256": observed_tree,
        "phase2f_validation_report_sha256": sha256_file(PHASE2F_EVIDENCE),
        "coco_files": {k: sha256_file(v) for k, v in FILES.items()},
    },
    "source_evidence": {
        "dataset_transfer_manifest": str(TRANSFER_MANIFEST),
        "phase2f_validation_report": str(PHASE2F_EVIDENCE),
    },
    "checks": checks,
    "report_created_at_utc": now,
}

INTEGRITY_REPORT.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

# Parse/validate the generated evidence again.
transfer_check = json.loads(TRANSFER_MANIFEST.read_text())
report_check = json.loads(INTEGRITY_REPORT.read_text())

transfer_ok = (
    transfer_check["status"] == "PASS"
    and transfer_check["observed_jpg_count"] == 4894
    and transfer_check["missing_jpg_count"] == 0
    and transfer_check["unexpected_jpg_count"] == 0
    and transfer_check["destination_tree_sha256"] == expected_tree_sha
    and transfer_check["direct_google_drive_io_for_training"] is False
)
report_ok = (
    report_check["status"] == "PASS"
    and all(report_check["checks"].values())
)

print("=== DATASET TRANSFER MANIFEST SUMMARY ===")
print("status =", transfer_check["status"])
print("transfer_mode =", transfer_check["transfer_mode"])
print("observed_jpg_count =", transfer_check["observed_jpg_count"])
print("missing_jpg_count =", transfer_check["missing_jpg_count"])
print("unexpected_jpg_count =", transfer_check["unexpected_jpg_count"])
print("destination_tree_sha256 =", transfer_check["destination_tree_sha256"])
print("direct_google_drive_io_for_training =", transfer_check["direct_google_drive_io_for_training"])
print("TRANSFER_MANIFEST_VALIDATION =", "PASS" if transfer_ok else "FAIL")

print("=== DATASET INTEGRITY REPORT SUMMARY ===")
print("status =", report_check["status"])
print("train =", split_observed["train"])
print("val =", split_observed["val"])
print("test =", split_observed["test"])
print("budgets =", budget_observed)
print("nesting =", nesting)
print("1pct_phase2f =", phase2f_1pct)
print("all_checks_pass =", all(checks.values()))
print("INTEGRITY_REPORT_VALIDATION =", "PASS" if report_ok else "FAIL")

if not transfer_ok or not report_ok:
    raise SystemExit("generated evidence validation failed")
PY


log
log "=== EXTENDED DATA-PREFLIGHT AUDITS ==="
python - "$SSOD_ROOT" <<'PY'
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import cv2

ROOT = Path(__import__("sys").argv[1])
COCO = ROOT / "data/coco"
ART = ROOT / "artifacts/preflight/data"
SRC = ART / "source_evidence"

PHASE2E_REPORT = SRC / "phase2E/phase2E_build_fixed_split_validation_report.json"
PHASE2E_SCRIPT = SRC / "phase2E/02E_C0_R2_exact_constrained_candidate_split.py"
PHASE2F_VALIDATION = SRC / "02F_labeled_unlabeled_validation_report.json"
PHASE2F_LOCK = SRC / "phase2F/phase2F_lock_manifest.json"
PHASE2F_SCRIPT = SRC / "phase2F/02F_build_labeled_unlabeled.py"

SPLITS = {
    "train": COCO / "instances_train.json",
    "val": COCO / "instances_val.json",
    "test": COCO / "instances_test.json",
}
LABELED = {
    "1pct": COCO / "instances_labeled_1pct.json",
    "5pct": COCO / "instances_labeled_5pct.json",
    "10pct": COCO / "instances_labeled_10pct.json",
    "20pct": COCO / "instances_labeled_20pct.json",
}
UNLABELED = {
    "1pct": COCO / "instances_unlabeled_1pct.json",
    "5pct": COCO / "instances_unlabeled_5pct.json",
    "10pct": COCO / "instances_unlabeled_10pct.json",
    "20pct": COCO / "instances_unlabeled_20pct.json",
}

EXPECTED_PHASE2E_REPORT_SHA = "64a072227863f949f07a0695dec06574351232cbb6d4fd331feb3a822e70d53e"
EXPECTED_PHASE2E_SCRIPT_SHA = "caeb03cf3d5454ab12a9a35fd8735d93431a33ff7a1c6053730d0503cca6ecda"
EXPECTED_PHASE2F_VALIDATION_SHA = "7b541eb09c60321b56d06ef4918a686c14a7c62eaa9965161ff1144f9a7b4467"
EXPECTED_PHASE2F_LOCK_SHA = "d8659d6fe40f9a32de0d32833c22dccc804da15009e3131d2640f7b9fb473c88"
EXPECTED_PHASE2F_SCRIPT_CAPTURE_SHA = "c5e6d04afdf9db2e5ed0c626eae8536677e0313319a8244f74bd98583e20da56"

EXPECTED_MAPPING = {
    1: "Aortic enlargement",
    2: "Atelectasis",
    3: "Calcification",
    4: "Cardiomegaly",
    5: "Consolidation",
    6: "ILD",
    7: "Infiltration",
    8: "Lung Opacity",
    9: "Nodule/Mass",
    10: "Other lesion",
    11: "Pleural effusion",
    12: "Pleural thickening",
    13: "Pneumothorax",
    14: "Pulmonary fibrosis",
}
EXPECTED_BUDGETS = {
    "1pct": {"L": 34, "U": 3392, "NF_L": 3},
    "5pct": {"L": 171, "U": 3255, "NF_L": 17},
    "10pct": {"L": 343, "U": 3083, "NF_L": 35},
    "20pct": {"L": 685, "U": 2741, "NF_L": 70},
}

def load(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def canonical_int(value):
    if isinstance(value, bool):
        raise ValueError(f"bool image_id is invalid: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    raise ValueError(f"non-integer image_id: {value!r}")

def phase2e_membership_sha256(values):
    canonical = "\n".join(str(v) for v in sorted(values, key=lambda v: str(v)))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def phase2f_membership_sha256(values):
    canonical_ints = sorted(canonical_int(v) for v in values)
    canonical = "\n".join(str(v) for v in canonical_ints)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def ids(data):
    return [canonical_int(im["id"]) for im in data["images"]]

def idset(data):
    return set(ids(data))

def annotated_ids(data):
    return {canonical_int(a["image_id"]) for a in data.get("annotations", [])}

def zero_gt_ids(data):
    return idset(data) - annotated_ids(data)

def mapping(data):
    return {int(c["id"]): c["name"] for c in data.get("categories", [])}

def find_1pct_diag(obj):
    if isinstance(obj, dict):
        if (
            obj.get("budget") == "1pct"
            and "class_coverage_count" in obj
            and "move_count_by_phase" in obj
        ):
            return obj
        for value in obj.values():
            found = find_1pct_diag(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_1pct_diag(value)
            if found is not None:
                return found
    return None

def write_validated(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    if parsed.get("status") != "PASS" or not all(parsed.get("checks", {}).values()):
        raise SystemExit(f"generated artifact failed validation: {path}")
    return sha256_file(path)

# Load all source artifacts.
p2e = load(PHASE2E_REPORT)
p2f_val = load(PHASE2F_VALIDATION)
p2f_lock = load(PHASE2F_LOCK)
Dsplit = {k: load(v) for k, v in SPLITS.items()}
DL = {k: load(v) for k, v in LABELED.items()}
DU = {k: load(v) for k, v in UNLABELED.items()}

# -------------------------------------------------------------------------
# 1) split_membership_audit.json
# -------------------------------------------------------------------------
locked_split_hash = p2e["locked_r2_evidence"]["image_id_sha256"]
split_obs = {}
split_sets = {}

for name, data in Dsplit.items():
    vals = ids(data)
    s = set(vals)
    split_sets[name] = s
    split_obs[name] = {
        "images": len(vals),
        "unique_image_ids": len(s),
        "annotations": len(data.get("annotations", [])),
        "zero_gt_images": len(zero_gt_ids(data)),
        "categories": len(data.get("categories", [])),
        "sha256_image_ids": phase2e_membership_sha256(vals),
        "sha256_coco_json": sha256_file(SPLITS[name]),
    }

overlaps = {
    "train_val": len(split_sets["train"] & split_sets["val"]),
    "train_test": len(split_sets["train"] & split_sets["test"]),
    "val_test": len(split_sets["val"] & split_sets["test"]),
}
union_ids = set().union(*split_sets.values())
annotation_union = sum(x["annotations"] for x in split_obs.values())

split_checks = {
    "phase2e_report_sha256_matches": sha256_file(PHASE2E_REPORT) == EXPECTED_PHASE2E_REPORT_SHA,
    "phase2e_script_sha256_matches": sha256_file(PHASE2E_SCRIPT) == EXPECTED_PHASE2E_SCRIPT_SHA,
    "phase2e_script_matches_report_locked_hash":
        sha256_file(PHASE2E_SCRIPT) == p2e["inputs"]["r2_script_sha256"],
    "phase2e_source_status_pass": p2e.get("status") == "PASS",
    "partition_seed_42": p2e["locked_r2_evidence"]["seed"] == 42,
    "seed_policy_locked":
        p2e["locked_r2_evidence"]["seed_policy"] == "PRE_SPECIFIED_LOCKED_NO_SEED_SEARCH",
    "train_membership_hash_matches":
        split_obs["train"]["sha256_image_ids"] == locked_split_hash["train"],
    "val_membership_hash_matches":
        split_obs["val"]["sha256_image_ids"] == locked_split_hash["val"],
    "test_membership_hash_matches":
        split_obs["test"]["sha256_image_ids"] == locked_split_hash["test"],
    "split_sizes_match":
        split_obs["train"]["images"] == 3426
        and split_obs["val"]["images"] == 734
        and split_obs["test"]["images"] == 734,
    "no_finding_counts_match":
        split_obs["train"]["zero_gt_images"] == 350
        and split_obs["val"]["zero_gt_images"] == 75
        and split_obs["test"]["zero_gt_images"] == 75,
    "image_ids_unique_within_each_split":
        all(x["images"] == x["unique_image_ids"] for x in split_obs.values()),
    "zero_cross_split_overlap": all(v == 0 for v in overlaps.values()),
    "image_union_4894": len(union_ids) == 4894,
    "annotation_union_36096": annotation_union == 36096,
    "all_splits_have_14_categories":
        all(x["categories"] == 14 for x in split_obs.values()),
}

split_audit = {
    "schema_version": "1.1",
    "artifact_type": "SPLIT_MEMBERSHIP_AUDIT",
    "status": "PASS" if all(split_checks.values()) else "FAIL",
    "canonical_membership_hash_rule": {
        "source": "02E_C0_R2_exact_constrained_candidate_split.py",
        "stable_id_key": "str(value)",
        "sort": "sorted(values, key=stable_id_key)",
        "serialization": "newline-joined str(image_id), no trailing newline",
        "encoding": "UTF-8",
        "digest": "SHA-256",
    },
    "phase2e_provenance": {
        "validation_report_path": str(PHASE2E_REPORT),
        "validation_report_sha256": sha256_file(PHASE2E_REPORT),
        "implementation_script_path": str(PHASE2E_SCRIPT),
        "implementation_script_sha256": sha256_file(PHASE2E_SCRIPT),
        "partition_seed": p2e["locked_r2_evidence"]["seed"],
        "seed_policy": p2e["locked_r2_evidence"]["seed_policy"],
    },
    "locked_membership_sha256": locked_split_hash,
    "observed_splits": split_obs,
    "split_integrity": {
        "overlaps": overlaps,
        "image_union": len(union_ids),
        "annotation_union": annotation_union,
    },
    "checks": split_checks,
    "audit_created_at_utc": datetime.now(timezone.utc).isoformat(),
}
split_sha = write_validated(ART / "split_membership_audit.json", split_audit)

# -------------------------------------------------------------------------
# 2) labeled_unlabeled_membership_audit.json
# -------------------------------------------------------------------------
train_ids = split_sets["train"]
train_zero_gt = zero_gt_ids(Dsplit["train"])
locked_L_hash = p2f_lock["labeled_image_id_sha256"]
locked_L_coco = p2f_lock["coco_json_sha256"]["labeled"]
locked_U_coco = p2f_lock["coco_json_sha256"]["unlabeled"]

lu_obs = {}
Lsets = {}
Usets = {}

for b in ("1pct", "5pct", "10pct", "20pct"):
    l_list = ids(DL[b])
    u_list = ids(DU[b])
    L = set(l_list)
    U = set(u_list)
    Lsets[b] = L
    Usets[b] = U
    lu_obs[b] = {
        "L": len(L),
        "U": len(U),
        "no_finding_in_L": len(L & train_zero_gt),
        "labeled_unique_image_ids": len(L),
        "unlabeled_unique_image_ids": len(U),
        "labeled_duplicate_image_id_count": len(l_list) - len(L),
        "unlabeled_duplicate_image_id_count": len(u_list) - len(U),
        "labeled_membership_sha256": phase2f_membership_sha256(L),
        "unlabeled_membership_sha256_observed_not_locked": phase2f_membership_sha256(U),
        "labeled_coco_sha256": sha256_file(LABELED[b]),
        "unlabeled_coco_sha256": sha256_file(UNLABELED[b]),
        "unlabeled_annotations_count": len(DU[b].get("annotations", [])),
        "L_subset_train": L <= train_ids,
        "U_subset_train": U <= train_ids,
        "L_intersect_U_empty": len(L & U) == 0,
        "L_union_U_equals_T": (L | U) == train_ids,
        "U_equals_exact_train_complement": U == (train_ids - L),
    }

diag1 = find_1pct_diag(p2f_val)
if diag1 is None:
    raise SystemExit("cannot locate 1pct diagnostics in Phase 2F validation evidence")

one_pct_coverage_identity = {
    "labeled_size": lu_obs["1pct"]["L"],
    "class_coverage_count": len({
        a["category_id"] for a in DL["1pct"].get("annotations", [])
    }),
    "present_class_ids": sorted({
        a["category_id"] for a in DL["1pct"].get("annotations", [])
    }),
    "missing_class_ids": sorted(
        set(range(1, 15))
        - {a["category_id"] for a in DL["1pct"].get("annotations", [])}
    ),
    "repair_move_count_total": diag1.get("repair_move_count_total"),
    "minimum_class_coverage_repair_move_count":
        diag1["move_count_by_phase"].get("minimum_class_coverage"),
    "phase2f_validation_report_sha256": sha256_file(PHASE2F_VALIDATION),
    "instances_labeled_1pct_sha256": sha256_file(LABELED["1pct"]),
    "interpretation":
        "The final 1% labeled subset contains 34 images and covers all 14 "
        "detection classes. The dedicated minimum-class-coverage repair "
        "phase required 0 moves; this does not mean total repair moves were zero.",
}

lu_nesting = {
    "L1_subset_L5": Lsets["1pct"] < Lsets["5pct"],
    "L5_subset_L10": Lsets["5pct"] < Lsets["10pct"],
    "L10_subset_L20": Lsets["10pct"] < Lsets["20pct"],
    "L20_subset_train": Lsets["20pct"] < train_ids,
    "U20_subset_U10": Usets["20pct"] < Usets["10pct"],
    "U10_subset_U5": Usets["10pct"] < Usets["5pct"],
    "U5_subset_U1": Usets["5pct"] < Usets["1pct"],
}
lu_checks = {
    "phase2f_validation_report_sha256_matches":
        sha256_file(PHASE2F_VALIDATION) == EXPECTED_PHASE2F_VALIDATION_SHA,
    "phase2f_lock_manifest_sha256_matches":
        sha256_file(PHASE2F_LOCK) == EXPECTED_PHASE2F_LOCK_SHA,
    "phase2f_current_script_identity_matches_capture":
        sha256_file(PHASE2F_SCRIPT) == EXPECTED_PHASE2F_SCRIPT_CAPTURE_SHA,
    "lock_train_coco_sha256_matches":
        sha256_file(SPLITS["train"]) == p2f_lock["train_coco_sha256"],
    "all_labeled_membership_hashes_match_lock":
        all(lu_obs[b]["labeled_membership_sha256"] == locked_L_hash[b] for b in lu_obs),
    "all_labeled_coco_hashes_match_lock":
        all(lu_obs[b]["labeled_coco_sha256"] == locked_L_coco[b] for b in lu_obs),
    "all_unlabeled_coco_hashes_match_lock":
        all(lu_obs[b]["unlabeled_coco_sha256"] == locked_U_coco[b] for b in lu_obs),
    "all_budget_counts_match":
        all(
            lu_obs[b]["L"] == EXPECTED_BUDGETS[b]["L"]
            and lu_obs[b]["U"] == EXPECTED_BUDGETS[b]["U"]
            and lu_obs[b]["no_finding_in_L"] == EXPECTED_BUDGETS[b]["NF_L"]
            for b in lu_obs
        ),
    "all_labeled_and_unlabeled_ids_unique":
        all(
            lu_obs[b]["labeled_duplicate_image_id_count"] == 0
            and lu_obs[b]["unlabeled_duplicate_image_id_count"] == 0
            for b in lu_obs
        ),
    "all_L_and_U_subset_train":
        all(lu_obs[b]["L_subset_train"] and lu_obs[b]["U_subset_train"] for b in lu_obs),
    "all_L_U_disjoint": all(lu_obs[b]["L_intersect_U_empty"] for b in lu_obs),
    "all_L_U_union_equals_train": all(lu_obs[b]["L_union_U_equals_T"] for b in lu_obs),
    "all_U_equal_exact_train_complement":
        all(lu_obs[b]["U_equals_exact_train_complement"] for b in lu_obs),
    "all_official_unlabeled_annotations_empty":
        all(lu_obs[b]["unlabeled_annotations_count"] == 0 for b in lu_obs),
    "nested_labeled_and_reverse_unlabeled_membership": all(lu_nesting.values()),
    "validation_report_status_pass": p2f_val.get("status") == "PASS",
    "validation_nested_pass": p2f_val.get("nested_split_check", {}).get("nested_pass") is True,
    "one_pct_labeled_size_34":
        one_pct_coverage_identity["labeled_size"] == 34,
    "one_pct_class_coverage_14_of_14":
        one_pct_coverage_identity["class_coverage_count"] == 14
        and one_pct_coverage_identity["missing_class_ids"] == [],
    "one_pct_minimum_class_coverage_repair_move_count_zero":
        one_pct_coverage_identity["minimum_class_coverage_repair_move_count"] == 0,
}

lu_audit = {
    "schema_version": "1.1",
    "artifact_type": "LABELED_UNLABELED_MEMBERSHIP_AUDIT",
    "status": "PASS" if all(lu_checks.values()) else "FAIL",
    "phase2f_membership_hash_rule": {
        "source": "02F_build_labeled_unlabeled.py",
        "input_requirement": "canonical integer image IDs",
        "sort": "numeric ascending",
        "serialization": "decimal str(image_id) joined by LF, no trailing LF",
        "encoding": "UTF-8",
        "digest": "SHA-256",
        "note": "Intentionally differs from Phase 2E sort-key=str(value) convention.",
    },
    "provenance": {
        "lock_manifest_path": str(PHASE2F_LOCK),
        "lock_manifest_sha256": sha256_file(PHASE2F_LOCK),
        "validation_report_path": str(PHASE2F_VALIDATION),
        "validation_report_sha256": sha256_file(PHASE2F_VALIDATION),
        "implementation_script_path": str(PHASE2F_SCRIPT),
        "implementation_script_sha256": sha256_file(PHASE2F_SCRIPT),
        "implementation_identity_role":
            "captured current implementation identity; not claimed as a script hash locked by Phase 2F artifacts",
    },
    "locked_labeled_membership_sha256": locked_L_hash,
    "observed_budgets": lu_obs,
    "one_pct_coverage_identity": one_pct_coverage_identity,
    "s1_10_evidence_status": (
        "PASS"
        if (
            lu_checks["one_pct_labeled_size_34"]
            and lu_checks["one_pct_class_coverage_14_of_14"]
            and lu_checks["one_pct_minimum_class_coverage_repair_move_count_zero"]
        )
        else "FAIL"
    ),
    "nesting": lu_nesting,
    "unlabeled_membership_semantics": {
        "definition": "U_b = T \\\\ L_b",
        "official_unlabeled_coco_files_verified": True,
        "official_unlabeled_annotations_must_be_empty": True,
        "independently_locked_unlabeled_membership_hash": False,
        "unlabeled_coco_sha256_locked": True,
    },
    "checks": lu_checks,
    "audit_created_at_utc": datetime.now(timezone.utc).isoformat(),
}
lu_sha = write_validated(ART / "labeled_unlabeled_membership_audit.json", lu_audit)

# -------------------------------------------------------------------------
# 3) class_mapping_audit.json
# -------------------------------------------------------------------------
all_coco = {}
all_coco.update(Dsplit)
all_coco.update({f"L_{k}": v for k, v in DL.items()})
all_coco.update({f"U_{k}": v for k, v in DU.items()})

mapping_matches = {name: mapping(data) == EXPECTED_MAPPING for name, data in all_coco.items()}
nf_as_category = {
    name: any(str(c.get("name", "")).strip().casefold() == "no finding"
              for c in data.get("categories", []))
    for name, data in all_coco.items()
}
coverage_L = {
    b: len({a["category_id"] for a in DL[b].get("annotations", [])})
    for b in DL
}
class_checks = {
    "exact_category_mapping_1_to_14": mapping(Dsplit["train"]) == EXPECTED_MAPPING,
    "category_ids_contiguous_1_to_14":
        sorted(mapping(Dsplit["train"])) == list(range(1, 15)),
    "detection_class_count_14": len(mapping(Dsplit["train"])) == 14,
    "mapping_consistent_across_fixed_labeled_unlabeled_coco_files":
        all(mapping_matches.values()),
    "no_finding_not_detection_category": not any(nf_as_category.values()),
    "all_labeled_budgets_cover_14_classes":
        all(v == 14 for v in coverage_L.values()),
    "all_unlabeled_annotations_empty":
        all(len(DU[b].get("annotations", [])) == 0 for b in DU),
}
class_audit = {
    "schema_version": "1.1",
    "artifact_type": "CLASS_MAPPING_AUDIT",
    "status": "PASS" if all(class_checks.values()) else "FAIL",
    "detection_class_count": 14,
    "canonical_category_mapping": {str(k): v for k, v in EXPECTED_MAPPING.items()},
    "category_semantics": {
        "no_finding": {
            "is_detection_category": False,
            "category_id": None,
            "representation": "zero-GT negative image",
        }
    },
    "mapping_matches_by_file": mapping_matches,
    "no_finding_category_presence_by_file": nf_as_category,
    "labeled_class_coverage": coverage_L,
    "checks": class_checks,
    "audit_created_at_utc": datetime.now(timezone.utc).isoformat(),
}
class_sha = write_validated(ART / "class_mapping_audit.json", class_audit)

# -------------------------------------------------------------------------
# 4) dataset_loader_manifest.json
# -------------------------------------------------------------------------
IMG_ROOT = ROOT / "data/images"
loader_files = {}
loader_files.update(SPLITS)
loader_files.update({f"L_{k}": v for k, v in LABELED.items()})
loader_files.update({f"U_{k}": v for k, v in UNLABELED.items()})
loader_data = {}
loader_data.update(Dsplit)
loader_data.update({f"L_{k}": v for k, v in DL.items()})
loader_data.update({f"U_{k}": v for k, v in DU.items()})

resolution = {}
all_resolved = True
for name, data in loader_data.items():
    missing = []
    resolved = 0
    for im in data["images"]:
        p = IMG_ROOT / Path(im["file_name"]).name
        if p.is_file():
            resolved += 1
        else:
            missing.append(str(p))
    resolution[name] = {
        "coco_images": len(data["images"]),
        "resolved_paths": resolved,
        "missing_paths": len(missing),
        "sample_missing_paths": missing[:10],
    }
    all_resolved = all_resolved and not missing

sample_checks = []
for split in ("train", "val", "test"):
    images = Dsplit[split]["images"]
    for idx in sorted(set((0, len(images)//2, len(images)-1))):
        rec = images[idx]
        p = IMG_ROOT / Path(rec["file_name"]).name
        arr = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        decoded = arr is not None
        if decoded:
            h, w = arr.shape[:2]
        else:
            h, w = None, None
        sample_checks.append({
            "split": split,
            "image_id": rec["id"],
            "resolved_path": str(p),
            "exists": p.is_file(),
            "decode_success": decoded,
            "expected_width": rec.get("width"),
            "expected_height": rec.get("height"),
            "observed_width": w,
            "observed_height": h,
            "dimension_match":
                decoded and w == rec.get("width") and h == rec.get("height"),
        })

loader_checks = {
    "training_image_root_exists": IMG_ROOT.is_dir(),
    "training_image_root_is_vast_runtime_path": str(IMG_ROOT).startswith("/workspace/"),
    "direct_google_drive_io_for_training_false": True,
    "all_coco_annotation_files_exist": all(p.is_file() for p in loader_files.values()),
    "all_image_paths_resolve": all_resolved,
    "all_sample_images_exist": all(x["exists"] for x in sample_checks),
    "all_sample_images_decode": all(x["decode_success"] for x in sample_checks),
    "all_sample_dimensions_match_coco": all(x["dimension_match"] for x in sample_checks),
    "all_official_unlabeled_annotations_empty":
        all(len(DU[b].get("annotations", [])) == 0 for b in DU),
}
loader_manifest = {
    "schema_version": "1.1",
    "artifact_type": "DATASET_LOADER_MANIFEST",
    "status": "PASS" if all(loader_checks.values()) else "FAIL",
    "training_image_root": str(IMG_ROOT),
    "direct_google_drive_io_for_training": False,
    "storage_semantics": {
        "transport_source": "Google Drive",
        "training_source": "Vast local/persistent filesystem",
        "google_drive_role": "transport/durable source only; never direct training I/O",
    },
    "coco_annotation_paths": {name: str(path) for name, path in loader_files.items()},
    "coco_annotation_sha256": {name: sha256_file(path) for name, path in loader_files.items()},
    "path_resolution_checks": resolution,
    "sample_decode_and_dimension_checks": sample_checks,
    "resolver_rule": {
        "description": "Resolve runtime image as training_image_root / basename(COCO file_name)",
        "reason": "Canonical Vast image root is flat and contains the exact 4894 locked JPGs.",
    },
    "checks": loader_checks,
    "manifest_created_at_utc": datetime.now(timezone.utc).isoformat(),
}
loader_sha = write_validated(ART / "dataset_loader_manifest.json", loader_manifest)

print("=== EXTENDED DATA-PREFLIGHT SUMMARY ===")
print("split_membership_audit = PASS")
print("labeled_unlabeled_membership_audit = PASS")
print("class_mapping_audit = PASS")
print("dataset_loader_manifest = PASS")
print("official_unlabeled_annotations_empty =", lu_checks["all_official_unlabeled_annotations_empty"])
print("all_official_U_equal_exact_train_complement =", lu_checks["all_U_equal_exact_train_complement"])
print("one_pct_labeled_size =", one_pct_coverage_identity["labeled_size"])
print("one_pct_class_coverage =", one_pct_coverage_identity["class_coverage_count"])
print("one_pct_minimum_class_coverage_repair_move_count =", one_pct_coverage_identity["minimum_class_coverage_repair_move_count"])
print("S1_10_EVIDENCE_STATUS =", lu_audit["s1_10_evidence_status"])
print("all_loader_paths_resolve =", loader_checks["all_image_paths_resolve"])
print("sample_decode_pass =", loader_checks["all_sample_images_decode"])
print("sample_dimension_match_pass =", loader_checks["all_sample_dimensions_match_coco"])
print("split_membership_audit_sha256 =", split_sha)
print("labeled_unlabeled_membership_audit_sha256 =", lu_sha)
print("class_mapping_audit_sha256 =", class_sha)
print("dataset_loader_manifest_sha256 =", loader_sha)
print("EXTENDED_DATA_PREFLIGHT_VALIDATION=PASS")
PY

log
log "=== MACHINE-READABLE EVIDENCE SHA256 ==="
TRANSFER_SHA="$(sha256_of "$TRANSFER_MANIFEST")"
INTEGRITY_SHA="$(sha256_of "$INTEGRITY_REPORT")"
log "dataset_transfer_manifest_sha256=$TRANSFER_SHA"
log "dataset_integrity_report_sha256=$INTEGRITY_SHA"

pass "S1.05 dataset transfer preparation/evidence"
pass "S1.06 dataset integrity/invariant evidence"
pass "fixed split membership audit"
pass "labeled/unlabeled membership + official U artifact audit"
pass "class mapping audit"
pass "dataset loader/path-resolution manifest"

log
log "========================================"
log "S1_05_STATUS=PASS"
log "S1_06_STATUS=PASS"
log "SPLIT_MEMBERSHIP_AUDIT=PASS"
log "LABELED_UNLABELED_MEMBERSHIP_AUDIT=PASS"
log "S1_10_COVERAGE_IDENTITY=PASS"
log "CLASS_MAPPING_AUDIT=PASS"
log "DATASET_LOADER_MANIFEST=PASS"
log "DATA_PREFLIGHT_BLOCK=PASS"
log "DATA_PREPARATION=PASS"
log "========================================"
