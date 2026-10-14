
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
# from github_code import current_calc  
from github_code.core_analysis_methods import (
    exclude_peak_idxs_based_on_local_maximum_period,
    quick_get_time_in_samples,
)


def index_peaks_above_threshold(
    detection_coefs: NpArray64,
    data_array: NpArray64,
    thr: NpArray64,
    direction: Literal["positive", "negative"],
    smooth=None,
) -> NDArray[np.integer]:
"""Claude comments:
what this function is for:
Given a noisy detection-score trace, it (1) marks where the score is above threshold,
(2) optionally stitches together nearby blocks so one event isn't double-counted,
(3) finds the boundaries of each above-threshold stretch, and
(4) within each stretch, finds the exact sample index of the true peak in the raw data.
The output, peak_idxs, is a clean list of -here's exactly where each detected event peaks
— ready to be used for measuring amplitude, kinetics, alignment, etc. for each individual event."""
    jj = (detection_coefs > thr).astype(int)

    # smooth any ones by 1 windows width in case of gaps
    if isinstance(smooth, int):
        W = smooth
        box_function = np.concatenate([[0, 0], np.ones(W), [0, 0]])
        jj = np.convolve(jj, box_function)
        jj[jj > 0] = 1
        jj = jj[0 : len(detection_coefs)]

    diffs = np.diff(np.r_[0, jj, 0]).astype(int)
    ups = np.where(diffs == 1)[0]
    downs = np.where(diffs == -1)[0] - 1

    peak_idxs = np.zeros(ups.size, dtype=int)

    for i, (up, down) in enumerate(zip(ups, downs)):

        if direction == "positive":
            peak_idxs[i] = up + np.argmax(data_array[np.arange(up, down + 1)])
        elif direction == "negative":
            peak_idxs[i] = up + np.argmin(data_array[np.arange(up, down + 1)])

    return peak_idxs
