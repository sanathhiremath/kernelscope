"""Compare detector outputs without claiming annotated task accuracy."""
import math


def iou(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/union if union > 0 else 0.


def agreement(reference, candidate, overlap=.9):
    """Greedy same-class IoU match, both sides score-filtered by caller."""
    matched = reference_count = candidate_count = 0
    overlaps, score_drifts = [], []
    finite = True
    if len(reference) != len(candidate):
        raise ValueError('Reference and candidate frame counts differ')
    for expected, actual in zip(reference, candidate):
        reference_count += len(expected)
        candidate_count += len(actual)
        used = set()
        for prediction in actual:
            finite &= all(math.isfinite(float(v)) for v in prediction['box'] + [prediction['score']])
        for detection in sorted(expected, key=lambda row: row['score'], reverse=True):
            possibilities = [(iou(detection['box'], row['box']), index) for index, row in enumerate(actual)
                             if index not in used and row['label'] == detection['label']]
            value, index = max(possibilities, default=(0., -1))
            if value >= overlap:
                used.add(index)
                matched += 1
                overlaps.append(value)
                score_drifts.append(abs(detection['score']-actual[index]['score']))
    denominator = reference_count + candidate_count
    f1 = 2*matched/denominator if denominator else 1.
    return dict(reference_detections=reference_count, candidate_detections=candidate_count,
                matched_detections=matched, preservation_f1=f1,
                mean_matched_iou=sum(overlaps)/len(overlaps) if overlaps else None,
                max_matched_score_drift=max(score_drifts, default=0.), finite=finite,
                passed=finite and reference_count > 0 and f1 >= .95)
