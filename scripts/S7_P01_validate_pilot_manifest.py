#!/usr/bin/env python3
import ast
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'artifacts/preflight/pilot/pilot_end_to_end_manifest.json'

EXPECTED = {
    'base_git_head': '81b9096629151b60853dc206f29935f994ab775d',
    'scientific_source_sha256': 'b6881ce90da9194f88fd9664b51dc96bdf112145949e3a89705f16b06853c681',
    'implementation_contract_sha256': 'd274f98402e0492039a9da0bdfed538102fc6ca155f4f7870525c55c46aa0e1d',
    'artifact_contract_sha256': '222270bfe62f64074ec1cc9c0436a68bd3b0766b92900a1457511ba4535d4ccc',
    'shared_validation_sha256': '14232e84c4e36b1627ee7bd74c8346075c63ef3d8a9a347ccb57adbf63469f10',
    'r50_config_sha256': 'b1e8408cea2ab8c95c5d841f6b862bc7e7e83c8e64347d4f9b84ece463a53041',
    'swin_config_sha256': '9d6b3a8298692c2af30c7e6bd3b00ea864527e484cbd0b9dd53864d9aa2c3dd6',
    'r50_parent_sha256': 'd76c4b66025369e6f2ee9b90ba0671a9c5ae31d62a1001033571497221f0539a',
    'swin_parent_sha256': '3fce5bf62d2864088abfb143acc1287205cdc259495df45d2ea5ab71f476ef56',
}

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

checks = {}
try:
    m = json.loads(MANIFEST.read_text(encoding='utf-8'))
    checks['manifest_exists'] = MANIFEST.is_file()
    checks['schema_version'] = m.get('schema_version') == '1.1'
    checks['artifact_type'] = m.get('artifact_type') == 'PILOT_END_TO_END_MANIFEST'
    checks['task'] = m.get('task') == 'S7.P01'
    checks['status'] = m.get('status') == 'LOCKED_PRE_EXECUTION'
    checks['execution_not_started'] = m.get('execution_status') == 'NOT_STARTED'
    checks['scientific_protocol_change_false'] = m.get('scientific_protocol_change') is False
    checks['base_git_head'] = m.get('base_git_head') == EXPECTED['base_git_head']
    g = m['governance']
    checks['scientific_source_identity'] = g['scientific_source_sha256'] == EXPECTED['scientific_source_sha256']
    checks['implementation_contract_identity'] = g['implementation_contract_sha256'] == EXPECTED['implementation_contract_sha256']
    checks['artifact_contract_identity'] = g['artifact_contract_sha256'] == EXPECTED['artifact_contract_sha256']
    iso = m['isolation']
    checks['pilot_role'] = iso['run_type'] == 'PILOT' and iso['condition_role'] == 'PILOT'
    checks['pilot_official_roots_separate'] = iso['runtime_root'] == 'artifacts/runs/pilot' and iso['official_runtime_root'] == 'artifacts/runs/official'
    checks['rename_prohibited'] = iso['pilot_to_official_rename_prohibited'] is True
    checks['final_test_ineligible'] = iso['eligible_for_final_test'] is False
    checks['official_training_unauthorized'] = iso['official_training_authorized'] is False
    checks['final_evaluation_not_authorized'] = iso['final_evaluation_status'] == 'NOT_AUTHORIZED'
    checks['test_unauthorized'] = iso['test_access_authorized'] is False
    checks['test_not_accessed'] = iso['test_accessed'] is False
    checks['hidden_u_not_accessed'] = iso['hidden_unlabeled_gt_accessed'] is False
    checks['performance_tuning_prohibited'] = iso['performance_driven_tuning_prohibited'] is True
    op = m['operational_shortening']
    checks['pilot_one_update'] = op['pilot_expected_optimizer_updates'] == 1 and op['pilot_max_iters'] == 1 and op['pilot_val_interval'] == 1
    checks['official_budget_unchanged'] = op['official_scientific_update_budget_unchanged'] is True and op['official_configs_modified'] is False
    sv = m['shared_validation_config']
    sv_path = ROOT / sv['path']
    checks['shared_validation_hash'] = sv_path.is_file() and sha256(sv_path) == EXPECTED['shared_validation_sha256'] == sv['sha256']
    checks['fixed_validation_only'] = sv['validation_ann_file'] == '/workspace/ssod/data/coco/instances_val.json' and sv['test_dataloader_attached'] is False and sv['test_evaluator_attached'] is False
    conditions = m['conditions']
    checks['condition_count'] = len(conditions) == 2
    expected_conditions = [('PILOT_SSL_R50_E2E_001','R50','configs/pilot/s7_p01_ssl_r50_tiny_e2e.py','configs/ssl/s5_06_soft_teacher_r50_fpn_scientific_eval_1pct.py',EXPECTED['r50_config_sha256'],EXPECTED['r50_parent_sha256']),('PILOT_SSL_SWIN_E2E_001','SWIN','configs/pilot/s7_p01_ssl_swin_t_tiny_e2e.py','configs/ssl/s5_06_soft_teacher_swin_t_fpn_scientific_eval_1pct.py',EXPECTED['swin_config_sha256'],EXPECTED['swin_parent_sha256'])]
    for c, exp in zip(conditions, expected_conditions):
        run_id, arch, cfg_rel, parent_rel, cfg_hash, parent_hash = exp
        cfg = ROOT / cfg_rel
        parent = ROOT / parent_rel
        prefix = arch.lower()
        checks[prefix + '_identity'] = c['pilot_run_id'] == run_id and c['method'] == 'SSL' and c['architecture'] == arch and c['label_budget'] == '1pct'
        checks[prefix + '_not_started'] = c['execution_status'] == 'NOT_STARTED' and c['expected_optimizer_updates'] == 1
        checks[prefix + '_config_hash'] = cfg.is_file() and sha256(cfg) == cfg_hash == c['pilot_config_sha256']
        checks[prefix + '_parent_hash'] = parent.is_file() and sha256(parent) == parent_hash == c['parent_official_family_config_sha256']
        checks[prefix + '_runtime_absent'] = not (ROOT / c['runtime_path']).exists()
        checks[prefix + '_config_no_bom'] = cfg.read_bytes()[:3] != b'\xef\xbb\xbf'
        ast.parse(cfg.read_text(encoding='utf-8'), filename=str(cfg))
    required = set(m['required_runtime_assertions'])
    checks['runtime_assertions_locked'] = {'teacher_initialized_from_student_at_t0','ssl_labeled_and_unlabeled_paths_exercised','pseudo_label_generation_path_exercised','ema_update_count_matches_optimizer_updates','test_access_count_equals_0','pilot_runtime_isolated_from_official_runtime'}.issubset(required)
except Exception as exc:
    print('S7_P01_VALIDATOR_EXCEPTION=' + repr(exc))
    sys.exit(2)

print('=== S7.P01 PILOT MANIFEST VALIDATOR ===')
for name, ok in checks.items():
    print(name + '=' + ('PASS' if ok else 'FAIL'))
failed = [name for name, ok in checks.items() if not ok]
print('TOTAL_CHECKS=' + str(len(checks)))
print('FAILED_CHECKS=' + str(len(failed)))
if failed:
    print('FAILED_NAMES=' + ','.join(failed))
    print('S7_P01_PILOT_MANIFEST=FAIL')
    sys.exit(1)
print('S7_P01_PILOT_MANIFEST=PASS')
