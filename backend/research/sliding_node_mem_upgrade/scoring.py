"""Benign-only relation calibration and repeated-evidence node scoring.

No labels are accepted by these components. The chronological first half of
validation fits reference distributions; the second half sets the threshold.
"""
from collections import defaultdict
import heapq
import math
import numpy as np


class RelationCalibrator:
    def fit(self, relations, losses):
        self.samples = {}
        for r in np.unique(relations):
            values = np.sort(np.asarray(losses)[np.asarray(relations) == r])
            self.samples[int(r)] = values
        self.fallback = np.sort(np.asarray(losses))
        if not len(self.fallback):
            raise ValueError('Empty benign calibration set')
        return self

    def transform(self, relations, losses):
        relations, losses = np.asarray(relations), np.asarray(losses)
        result = np.zeros_like(losses, dtype=float)
        for r in np.unique(relations):
            select = relations == r
            values = self.samples.get(int(r), self.fallback)
            if len(values) < 32:
                values = self.fallback
            tail = (len(values) - np.searchsorted(values, losses[select], side='left') + 1) / (len(values) + 1)
            # Extend beyond the reference maximum to avoid saturated tail ties.
            scale = max(float(np.quantile(values, .75) - np.quantile(values, .25)), .1)
            extra = np.log1p(np.maximum(losses[select] - values[-1], 0.) / scale)
            result[select] = -np.log(tail) + extra
        return result


class NodeEvidence:
    def __init__(self, k=3):
        self.k, self.heaps, self.counts = k, defaultdict(list), defaultdict(int)

    def update(self, src, dst, scores):
        for u, v, value in zip(src, dst, scores):
            for node in {int(u), int(v)}:
                self.counts[node] += 1
                heap = self.heaps[node]
                if len(heap) < self.k:
                    heapq.heappush(heap, float(value))
                elif value > heap[0]:
                    heapq.heapreplace(heap, float(value))

    def scores(self):
        # Missing top-k evidence is zero. Bonferroni-style degree penalty reduces
        # the multiple-opportunity advantage of high-activity benign entities.
        return {node: max(0., sum(heap) / self.k - math.log(max(1, self.counts[node])))
                for node, heap in self.heaps.items()}


def calibrated_scores(validation_records, test_records, benign_quantile=.9999):
    val = np.asarray(validation_records, dtype=float)
    test = np.asarray(test_records, dtype=float)
    cut = len(val) // 2
    if cut < 32 or not len(test):
        raise ValueError('Insufficient calibration or test events')
    calibrator = RelationCalibrator().fit(val[:cut, 2], val[:cut, 3])
    outputs = []
    for data in (val[cut:], test):
        aggregator = NodeEvidence()
        scores = calibrator.transform(data[:, 2], data[:, 3])
        aggregator.update(data[:, 0], data[:, 1], scores)
        outputs.append(aggregator.scores())
    threshold = float(np.quantile(list(outputs[0].values()), benign_quantile, method='higher'))
    return threshold, outputs[1]
