set -euo pipefail

ROOTFS=/workspace/rq4_frozen_bundle/rootfs
HOSTOUT=/workspace/rq4_outputs/s10_1_13_raw_v1
ROOTOUT="$ROOTFS/tmp/s10_1_13_rq4_raw_v1"

echo '=== S10.1.13 RQ4 FROZEN RAW MATERIALIZATION ==='

if [ -e "$HOSTOUT" ] || [ -e "$ROOTOUT" ]; then
  echo 'OUTPUT_DIR_ALREADY_EXISTS'
  echo "HOSTOUT_EXISTS=$([ -e "$HOSTOUT" ] && echo TRUE || echo FALSE)"
  echo "ROOTOUT_EXISTS=$([ -e "$ROOTOUT" ] && echo TRUE || echo FALSE)"
  if [ -d "$HOSTOUT" ]; then
    echo '=== EXISTING HOST OUTPUT INVENTORY ==='
    find "$HOSTOUT" -maxdepth 1 -type f -printf '%f\n' | sort
  fi
  exit 2
fi

sha() {
  sha256sum "$1" | awk '{print $1}'
}

test "$(sha /workspace/rq4_inputs/S10_1_13_compute_rq4.R)" = '551b371cf8f53e4430640cb4f520b42b2d28842634029b4a943e863075128bfe'
test "$(sha /workspace/rq4_inputs/rq4_input_for_frozen_prepare.csv)" = 'e9fc962382aea76682f8750780504ac67e89cac9165d08b233ab65c5682a593f'
test "$(sha /workspace/rq4_inputs/rq4_mixed_model.R)" = '6c92eaf928c203cd68917b1a1ebc7cd3fd0b6977e541e9e3df7962a3c08ff1d4'
test "$(sha /workspace/rq4_inputs/rq4_cr2_sensitivity.R)" = '9e9a21faf87c851ed9d74151d4b6c2860d70f2b143fb32ad4662dd2b0c049668'

cp /workspace/rq4_inputs/S10_1_13_compute_rq4.R "$ROOTFS/tmp/S10_1_13_compute_rq4.R"
cp /workspace/rq4_inputs/rq4_input_for_frozen_prepare.csv "$ROOTFS/tmp/rq4_input_for_frozen_prepare.csv"
cp /workspace/rq4_inputs/rq4_mixed_model.R "$ROOTFS/tmp/rq4_mixed_model.R"
cp /workspace/rq4_inputs/rq4_cr2_sensitivity.R "$ROOTFS/tmp/rq4_cr2_sensitivity.R"

chroot "$ROOTFS" /usr/local/bin/Rscript \
  /tmp/S10_1_13_compute_rq4.R \
  /tmp/rq4_input_for_frozen_prepare.csv \
  /tmp/s10_1_13_rq4_raw_v1 \
  /tmp/rq4_mixed_model.R \
  /tmp/rq4_cr2_sensitivity.R

mkdir -p "$HOSTOUT"
cp "$ROOTOUT"/* "$HOSTOUT"/

echo '=== RAW FILE INVENTORY ==='
find "$HOSTOUT" -maxdepth 1 -type f -printf '%f\n' | sort

echo '=== RAW FILE SHA256 ==='
sha256sum "$HOSTOUT"/* | sort -k2

echo '=== ROW AUDIT ==='
echo "ANALYSIS_DATASET_LINES=$(wc -l < "$HOSTOUT/rq4_analysis_dataset.raw.csv")"
echo "SEED_INFLUENCE_LINES=$(wc -l < "$HOSTOUT/rq4_seed_influence.raw.csv")"

test "$(find "$HOSTOUT" -maxdepth 1 -type f | wc -l)" -eq 6
test "$(wc -l < "$HOSTOUT/rq4_analysis_dataset.raw.csv")" -eq 81
test "$(wc -l < "$HOSTOUT/rq4_seed_influence.raw.csv")" -eq 11

echo 'S10_1_13_RQ4_FROZEN_RAW_MATERIALIZATION=PASS'
