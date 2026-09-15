'''S5.04 golden tests for FP/image at the locked operating point.'''

import hashlib
import json
import runpy
import sys
import tempfile
from pathlib import Path

import numpy as np

from src.metrics.operating_point_metric import OperatingPointMetric

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'artifacts' / 'preflight' / 'evaluation' / 'operating_point_golden_test.json'
S5_03_VALIDATOR = ROOT / 'scripts' / 'S5_03_validate_operating_point_metric.py'
S5_03_METRIC_SHA256 = 'e1bd577f8aaa613e9b31910f0df029792036bdb03790796a0394d13c1eac03a1'
S5_03_VALIDATOR_SHA256 = '0c9bda89417f6356d50385340aa78f6185cb7ec188afd7b539993c20c0273c74'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run():
    historical = json.loads(EVIDENCE.read_text(encoding='utf-8'))
    checks = []

    def check(name, ok, observed):
        checks.append({'name': name, 'pass': bool(ok), 'observed': observed})

    check('historical_s5_03_identity',
          historical.get('task') == 'S5.03' and historical.get('status') == 'PASS'
          and historical.get('checks_passed') == 15 and historical.get('checks_total') == 15,
          {'task': historical.get('task'), 'status': historical.get('status'),
           'checks_passed': historical.get('checks_passed'), 'checks_total': historical.get('checks_total')})
    check('historical_s5_03_checks_preserved',
          len(historical.get('checks', [])) == 15
          and all(c.get('pass') for c in historical.get('checks', [])),
          len(historical.get('checks', [])))
    historical_sources = historical.get('source_sha256', {})
    check('historical_s5_03_source_hashes_preserved',
          historical_sources.get('src/metrics/operating_point_metric.py') == S5_03_METRIC_SHA256
          and historical_sources.get('scripts/S5_03_validate_operating_point_metric.py') == S5_03_VALIDATOR_SHA256,
          historical_sources)

    helpers = runpy.run_path(str(S5_03_VALIDATOR))
    build_fixture = helpers['build_fixture']
    boxes = helpers['boxes']

    with tempfile.TemporaryDirectory() as td:
        ann = Path(td) / 'fixture.json'
        build_fixture(ann)
        metric = OperatingPointMetric(str(ann), tau_eval=0.50, iou_match_threshold=0.50,
                                      max_dets_per_image=100, prefix='operating')

        results = [
            {'img_id': 1,
             'bboxes': boxes([[0, 0, 10, 10], [0, 0, 10, 10]]),
             'scores': np.array([0.90, 0.80]), 'labels': np.array([0, 0])},
            {'img_id': 8,
             'bboxes': boxes([[0, 0, 10, 10]]),
             'scores': np.array([0.10]), 'labels': np.array([0])},
            {'img_id': 9,
             'bboxes': boxes([[0, 0, 10, 10]]),
             'scores': np.array([0.90]), 'labels': np.array([0])},
        ]

        matched = [metric.match_image(r['img_id'], r['bboxes'], r['scores'], r['labels'])
                   for r in results]
        fp_total = sum(r['FP'] for r in matched)
        evaluated_image_count = len(results)
        agg = metric.compute_metrics(results)
        expected_fp_per_image = 2.0 / 3.0

        check('same_operating_point_locks',
              metric.tau_eval == 0.5 and metric.iou_match_threshold == 0.5
              and metric.max_dets_per_image == 100 and metric.cat_ids == tuple(range(1, 15)),
              {'tau': metric.tau_eval, 'iou': metric.iou_match_threshold,
               'max': metric.max_dets_per_image, 'cats': list(metric.cat_ids)})
        check('fp_total', fp_total == 2, {'FP_total': fp_total, 'per_image': matched})
        check('evaluated_image_count', evaluated_image_count == 3, evaluated_image_count)
        check('fp_per_image',
              abs(agg['FP_per_image'] - expected_fp_per_image) < 1e-12, agg)
        check('recall_preserved', abs(agg['Recall_tau_eval'] - 0.5) < 1e-12, agg)
        check('output_keys_exact',
              set(agg) == {'Recall_tau_eval', 'FP_per_image'}, sorted(agg))
        check('zero_gt_image_contributes_global_fp',
              matched[2]['GT'] == 0 and matched[2]['FP'] == 1 and matched[2]['FN'] == 0,
              matched[2])

    all_pass = all(c['pass'] for c in checks)
    section = {
        'schema_version': '1.0.0',
        'task': 'S5.04',
        'status': 'PASS' if all_pass else 'FAIL',
        'metric': 'FP_per_image',
        'definition': 'SUM_FP_OVER_ALL_EVALUATED_IMAGES_DIVIDED_BY_EVALUATED_IMAGE_COUNT',
        'tau_eval': 0.50,
        'iou_match_threshold': 0.50,
        'max_dets_per_image': 100,
        'after_detector_native_nms_max100': True,
        'evaluator_runs_second_nms': False,
        'evaluator_truncates_top100': False,
        'matching': 'CATEGORY_AWARE_ONE_TO_ONE',
        'matching_algorithm': 'DETERMINISTIC_SCORE_DESCENDING_GREEDY',
        'prediction_order': 'DESCENDING_CONFIDENCE_STABLE_DETECTOR_ORDER',
        'gt_candidate_policy': 'UNMATCHED_SAME_CLASS_ONLY',
        'gt_selection': 'MAX_IOU',
        'equal_score_tie_break': 'PRESERVE_DETECTOR_OUTPUT_ORDER',
        'equal_iou_tie_break': 'EARLIEST_FIXED_COCO_ANNOTATION_ORDER',
        'fixed_validation_only': True,
        'test_access': False,
        'fixture': 'SYNTHETIC_COCO_14_CLASS_REUSED_FROM_S5_03',
        'evaluated_image_count': evaluated_image_count,
        'FP_total': fp_total,
        'FP_per_image': float(agg['FP_per_image']),
        'Recall_tau_eval': float(agg['Recall_tau_eval']),
        'checks_passed': sum(c['pass'] for c in checks),
        'checks_total': len(checks),
        'checks': checks,
        'source_sha256': {
            'src/metrics/operating_point_metric.py': sha256(ROOT / 'src/metrics/operating_point_metric.py'),
            'src/metrics/__init__.py': sha256(ROOT / 'src/metrics/__init__.py'),
            'configs/evaluation/s5_03_operating_point_evaluator.py': sha256(ROOT / 'configs/evaluation/s5_03_operating_point_evaluator.py'),
            'scripts/S5_03_validate_operating_point_metric.py': sha256(S5_03_VALIDATOR),
            'scripts/S5_04_validate_fp_per_image.py': sha256(Path(__file__)),
        },
    }

    historical['s5_04_fp_per_image'] = section
    EVIDENCE.write_text(json.dumps(historical, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    print('===== S5.04 FP/IMAGE GOLDEN TEST =====')
    for c in checks:
        print('{}={}'.format(c['name'], c['pass']))
    print('FP_TOTAL={}'.format(section['FP_total']))
    print('EVALUATED_IMAGE_COUNT={}'.format(section['evaluated_image_count']))
    print('FP_PER_IMAGE={}'.format(section['FP_per_image']))
    print('CHECKS_PASSED={}'.format(section['checks_passed']))
    print('CHECKS_TOTAL={}'.format(section['checks_total']))
    print('STATUS={}'.format(section['status']))
    print('EVIDENCE={}'.format(EVIDENCE))
    print('EVIDENCE_SHA256={}'.format(sha256(EVIDENCE)))
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(run())
