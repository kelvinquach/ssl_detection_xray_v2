'''S5.05 golden tests for FP/negative and negative-image FAR.'''

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
S5_04_METRIC_SHA256 = '03e1a6800e9cb9150e7da05d6e3406dc50a6892337d0b0e943267633dfc73112'
S5_04_VALIDATOR_SHA256 = '407beee0b2aa88917996a7fca3256185ca7fc0e0946dd17b1376317b16b64fb7'


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

    check(
        'historical_s5_03_identity',
        historical.get('task') == 'S5.03'
        and historical.get('status') == 'PASS'
        and historical.get('checks_passed') == 15
        and historical.get('checks_total') == 15,
        {
            'task': historical.get('task'),
            'status': historical.get('status'),
            'checks_passed': historical.get('checks_passed'),
            'checks_total': historical.get('checks_total'),
        },
    )
    check(
        'historical_s5_03_checks_preserved',
        len(historical.get('checks', [])) == 15
        and all(c.get('pass') for c in historical.get('checks', [])),
        len(historical.get('checks', [])),
    )
    historical_sources = historical.get('source_sha256', {})
    check(
        'historical_s5_03_source_hashes_preserved',
        historical_sources.get('src/metrics/operating_point_metric.py')
        == S5_03_METRIC_SHA256
        and historical_sources.get('scripts/S5_03_validate_operating_point_metric.py')
        == S5_03_VALIDATOR_SHA256,
        historical_sources,
    )

    s5_04 = historical.get('s5_04_fp_per_image')
    check(
        'historical_s5_04_identity',
        isinstance(s5_04, dict)
        and s5_04.get('task') == 'S5.04'
        and s5_04.get('status') == 'PASS'
        and s5_04.get('checks_passed') == 10
        and s5_04.get('checks_total') == 10,
        None if not isinstance(s5_04, dict) else {
            'task': s5_04.get('task'),
            'status': s5_04.get('status'),
            'checks_passed': s5_04.get('checks_passed'),
            'checks_total': s5_04.get('checks_total'),
        },
    )
    check(
        'historical_s5_04_checks_preserved',
        isinstance(s5_04, dict)
        and len(s5_04.get('checks', [])) == 10
        and all(c.get('pass') for c in s5_04.get('checks', [])),
        0 if not isinstance(s5_04, dict) else len(s5_04.get('checks', [])),
    )
    s5_04_sources = {} if not isinstance(s5_04, dict) else s5_04.get('source_sha256', {})
    check(
        'historical_s5_04_source_hashes_preserved',
        s5_04_sources.get('src/metrics/operating_point_metric.py')
        == S5_04_METRIC_SHA256
        and s5_04_sources.get('scripts/S5_04_validate_fp_per_image.py')
        == S5_04_VALIDATOR_SHA256,
        s5_04_sources,
    )

    helpers = runpy.run_path(str(S5_03_VALIDATOR))
    build_fixture = helpers['build_fixture']
    boxes = helpers['boxes']

    with tempfile.TemporaryDirectory() as td:
        ann = Path(td) / 'fixture.json'
        build_fixture(ann)

        fixture = json.loads(ann.read_text(encoding='utf-8'))
        fixture['images'].append(
            {'id': 10, 'file_name': '10.jpg', 'width': 100, 'height': 100}
        )
        ann.write_text(
            json.dumps(fixture, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )

        metric = OperatingPointMetric(
            str(ann),
            tau_eval=0.50,
            iou_match_threshold=0.50,
            max_dets_per_image=100,
            prefix='operating',
        )

        results = [
            {
                'img_id': 1,
                'bboxes': boxes([[0, 0, 10, 10], [0, 0, 10, 10]]),
                'scores': np.array([0.90, 0.80]),
                'labels': np.array([0, 0]),
            },
            {
                'img_id': 8,
                'bboxes': boxes([[0, 0, 10, 10]]),
                'scores': np.array([0.10]),
                'labels': np.array([0]),
            },
            {
                'img_id': 9,
                'bboxes': boxes([[0, 0, 10, 10], [20, 20, 30, 30]]),
                'scores': np.array([0.90, 0.80]),
                'labels': np.array([0, 1]),
            },
            {
                'img_id': 10,
                'bboxes': boxes([[0, 0, 10, 10]]),
                'scores': np.array([0.10]),
                'labels': np.array([0]),
            },
        ]

        matched = [
            metric.match_image(
                r['img_id'],
                r['bboxes'],
                r['scores'],
                r['labels'],
            )
            for r in results
        ]
        agg = metric.compute_metrics(results)

        negative_rows = [m for m in matched if m['GT'] == 0]
        negative_image_count = len(negative_rows)
        fp_total_on_negative = sum(m['FP'] for m in negative_rows)
        negative_images_with_fp = sum(m['FP'] >= 1 for m in negative_rows)

        expected_fp_per_image = 3.0 / 4.0
        expected_fp_per_negative = 2.0 / 2.0
        expected_negative_far = 1.0 / 2.0

        check(
            'same_operating_point_locks',
            metric.tau_eval == 0.5
            and metric.iou_match_threshold == 0.5
            and metric.max_dets_per_image == 100
            and metric.cat_ids == tuple(range(1, 15)),
            {
                'tau': metric.tau_eval,
                'iou': metric.iou_match_threshold,
                'max': metric.max_dets_per_image,
                'cats': list(metric.cat_ids),
            },
        )
        check(
            'zero_gt_images_identified',
            negative_image_count == 2
            and matched[2]['GT'] == 0
            and matched[3]['GT'] == 0,
            {'negative_image_count': negative_image_count, 'per_image': matched},
        )
        check(
            'all_retained_detections_on_zero_gt_are_fp',
            matched[2]['retained_detections'] == 2
            and matched[2]['FP'] == 2
            and matched[2]['TP'] == 0
            and matched[2]['FN'] == 0
            and matched[3]['retained_detections'] == 0
            and matched[3]['FP'] == 0
            and matched[3]['TP'] == 0
            and matched[3]['FN'] == 0,
            {'negative_rows': negative_rows},
        )
        check(
            'negative_image_count',
            negative_image_count == 2,
            negative_image_count,
        )
        check(
            'fp_total_on_negative',
            fp_total_on_negative == 2,
            fp_total_on_negative,
        )
        check(
            'fp_per_negative',
            abs(agg['FP_per_negative'] - expected_fp_per_negative) < 1e-12,
            agg,
        )
        check(
            'negative_image_far',
            abs(agg['negative_image_FAR'] - expected_negative_far) < 1e-12,
            {
                'negative_images_with_fp': negative_images_with_fp,
                'negative_image_count': negative_image_count,
                'negative_image_FAR': agg['negative_image_FAR'],
            },
        )
        check(
            'fp_per_negative_and_far_are_distinct',
            abs(agg['FP_per_negative'] - agg['negative_image_FAR']) > 1e-12,
            {
                'FP_per_negative': agg['FP_per_negative'],
                'negative_image_FAR': agg['negative_image_FAR'],
            },
        )
        check(
            'fp_per_image_preserved',
            abs(agg['FP_per_image'] - expected_fp_per_image) < 1e-12,
            agg,
        )
        check(
            'recall_preserved',
            abs(agg['Recall_tau_eval'] - 0.5) < 1e-12,
            agg,
        )
        check(
            'output_keys_exact',
            set(agg)
            == {
                'Recall_tau_eval',
                'FP_per_image',
                'FP_per_negative',
                'negative_image_FAR',
            },
            sorted(agg),
        )

    all_pass = all(c['pass'] for c in checks)
    section = {
        'schema_version': '1.0.0',
        'task': 'S5.05',
        'status': 'PASS' if all_pass else 'FAIL',
        'metrics': ['FP_per_negative', 'negative_image_FAR'],
        'fp_per_negative_definition':
            'SUM_FP_ON_ZERO_GT_NEGATIVES_DIVIDED_BY_NEGATIVE_IMAGE_COUNT',
        'negative_image_far_definition':
            'NEGATIVE_IMAGES_WITH_AT_LEAST_ONE_FP_DIVIDED_BY_NEGATIVE_IMAGE_COUNT',
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
        'negative_definition': 'ZERO_GT_NO_FINDING_NOT_DETECTION_CLASS',
        'all_retained_detections_on_zero_gt_are_fp': True,
        'fixed_validation_only': True,
        'test_access': False,
        'fixture': 'SYNTHETIC_COCO_14_CLASS_WITH_TWO_ZERO_GT_NEGATIVES',
        'evaluated_image_count': len(results),
        'negative_image_count': negative_image_count,
        'FP_total_on_negative': fp_total_on_negative,
        'negative_images_with_fp': negative_images_with_fp,
        'FP_per_negative': float(agg['FP_per_negative']),
        'negative_image_FAR': float(agg['negative_image_FAR']),
        'FP_per_image': float(agg['FP_per_image']),
        'Recall_tau_eval': float(agg['Recall_tau_eval']),
        'checks_passed': sum(c['pass'] for c in checks),
        'checks_total': len(checks),
        'checks': checks,
        'source_sha256': {
            'src/metrics/operating_point_metric.py':
                sha256(ROOT / 'src/metrics/operating_point_metric.py'),
            'src/metrics/__init__.py':
                sha256(ROOT / 'src/metrics/__init__.py'),
            'configs/evaluation/s5_03_operating_point_evaluator.py':
                sha256(ROOT / 'configs/evaluation/s5_03_operating_point_evaluator.py'),
            'scripts/S5_03_validate_operating_point_metric.py':
                sha256(S5_03_VALIDATOR),
            'scripts/S5_04_validate_fp_per_image.py':
                sha256(ROOT / 'scripts/S5_04_validate_fp_per_image.py'),
            'scripts/S5_05_validate_negative_metrics.py':
                sha256(Path(__file__)),
        },
    }

    historical['s5_05_negative_metrics'] = section
    EVIDENCE.write_text(
        json.dumps(historical, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    print('===== S5.05 NEGATIVE METRICS GOLDEN TEST =====')
    for c in checks:
        print('{}={}'.format(c['name'], c['pass']))
    print('NEGATIVE_IMAGE_COUNT={}'.format(section['negative_image_count']))
    print('FP_TOTAL_ON_NEGATIVE={}'.format(section['FP_total_on_negative']))
    print('NEGATIVE_IMAGES_WITH_FP={}'.format(section['negative_images_with_fp']))
    print('FP_PER_NEGATIVE={}'.format(section['FP_per_negative']))
    print('NEGATIVE_IMAGE_FAR={}'.format(section['negative_image_FAR']))
    print('FP_PER_IMAGE={}'.format(section['FP_per_image']))
    print('RECALL_TAU_EVAL={}'.format(section['Recall_tau_eval']))
    print('CHECKS_PASSED={}'.format(section['checks_passed']))
    print('CHECKS_TOTAL={}'.format(section['checks_total']))
    print('STATUS={}'.format(section['status']))
    print('EVIDENCE={}'.format(EVIDENCE))
    print('EVIDENCE_SHA256={}'.format(sha256(EVIDENCE)))
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(run())
