"""Benign-calibrated anomaly evidence without a top-30 alert cap.

Scores preserve a single extreme event and repeated evidence in fixed time
windows. Peaks never fall as harmless traffic arrives. Calibration is empirical;
its false-alert budget is not a guarantee under distribution shift.
"""
import heapq
import math
from pathlib import Path
import numpy as np
from sliding_node_mem_upgrade.scoring import RelationCalibrator

EDGE_DTYPE = np.dtype([('src', '<i8'), ('dst', '<i8'), ('time', '<f8'),
                       ('relation', '<i4'), ('loss', '<f4')])


class EdgeWriter:
    """One contiguous binary write per batch; scores can be memory-mapped later."""
    def __init__(self, path):
        self.path = Path(path)
        self.file = self.path.open('wb', buffering=1024*1024)
        self.count = 0

    def append(self, src, dst, times, relations, losses):
        rows = np.empty(len(src), dtype=EDGE_DTYPE)
        for name, values in zip(EDGE_DTYPE.names, (src, dst, times, relations, losses)):
            rows[name] = values
        self.file.write(rows.tobytes())
        self.count += len(rows)

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_edges(path):
    if Path(path).stat().st_size % EDGE_DTYPE.itemsize:
        raise ValueError('Incomplete edge score file')
    if not Path(path).stat().st_size:
        return np.empty(0, dtype=EDGE_DTYPE)
    return np.memmap(path, dtype=EDGE_DTYPE, mode='r')


class NodeEvidence:
    def __init__(self, window_seconds=900., k=3):
        if window_seconds <= 0 or k < 1:
            raise ValueError('Positive evidence window and k required')
        self.window_seconds, self.k = window_seconds, k
        self.heaps, self.windows, self.peaks = {}, {}, {}
        self.last_time = -math.inf
        self.completed = []

    def update(self, src, dst, times, scores):
        if not (len(src) == len(dst) == len(times) == len(scores)):
            raise ValueError('One endpoint pair, time and score per event required')
        for u, v, when, value in zip(src, dst, times, scores):
            if not math.isfinite(when) or when < self.last_time or not math.isfinite(value) or value < 0:
                raise ValueError('Finite chronological timestamps and nonnegative scores required')
            self.last_time = when
            window = math.floor(when/self.window_seconds)
            for node in {int(u), int(v)}:
                if node in self.windows and self.windows[node] != window:
                    self.completed.append(self._score(self.heaps.pop(node)))
                self.windows[node] = window
                heap = self.heaps.setdefault(node, [])
                if len(heap) < self.k:
                    heapq.heappush(heap, float(value))
                elif value > heap[0]:
                    heapq.heapreplace(heap, float(value))
                self.peaks[node] = max(self.peaks.get(node, 0.), self._score(heap))

    def _score(self, heap):
        # The single-event channel is never divided by k. Independent repeated
        # evidence can increase the score; zeros cannot decrease it.
        return max(max(heap, default=0.), sum(heap)/math.sqrt(self.k))

    def scores(self):
        return dict(self.peaks)

    def drain_completed(self):
        scores, self.completed = self.completed, []
        return scores

    def window_scores(self):
        return self.completed + [self._score(heap) for heap in self.heaps.values()]


def calibrated_scores(validation, test, false_alert_rate=1e-4, window_seconds=900., chunk=65536):
    if not 0 < false_alert_rate < 1 or len(validation) < 64 or not len(test):
        raise ValueError('Need >=64 benign validation events, nonempty test, and a rate in (0,1)')
    cut = len(validation)//2
    calibrator = RelationCalibrator().fit(validation['relation'][:cut], validation['loss'][:cut])
    results, thresholds = [], []
    for split, rows in enumerate((validation[cut:], test)):
        evidence = NodeEvidence(window_seconds=window_seconds)
        for lo in range(0, len(rows), chunk):
            block = rows[lo:lo+chunk]
            values = calibrator.transform(block['relation'], block['loss'])
            evidence.update(block['src'], block['dst'], block['time'], values)
            # Test needs only persistent peak scores, not all historical windows.
            if split:
                evidence.drain_completed()
        if split == 0:
            thresholds = evidence.window_scores()
        results.append(evidence.scores())
    # A finite-sample conservative order statistic; infinity honestly signals
    # that the requested tail probability lacks sufficient calibration support.
    values = np.sort(np.asarray(thresholds))
    rank = math.ceil((len(values)+1)*(1-false_alert_rate))
    threshold = float(values[rank-1]) if rank <= len(values) else math.inf
    return threshold, results[1], dict(calibration_windows=len(values),
                                     supported=rank <= len(values),
                                     false_alert_rate_per_node_window=false_alert_rate,
                                     window_seconds=window_seconds)
