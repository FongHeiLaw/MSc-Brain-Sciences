from __future__ import annotations

from types import SimpleNamespace
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from scipy.signal import find_peaks

import pandas as pd
import os
import sys
import bottleneck as bn
import neo

import github_code as ephys

from github_code import utils

import matplotlib.pyplot as plt

import metadata_lookup

from filter_trace import filter_trace




# --------------------------------------------------------------------------
# Type aliases (point 1 fix) -- same shapes as custom_types.py, minus the
# PySide6 / pyqtgraph ones this file doesn't need.
# --------------------------------------------------------------------------
NpArray64 = NDArray[np.float64]
Direction = Union[Literal[1], Literal[-1]]
Int = Union[int, np.integer]


#

def select_by_peak_distance(peaks, priority, distance):
    """
    Pure-Python replacement for SciPy's internal _select_by_peak_distance.
    """

    peaks = np.asarray(peaks)
    priority = np.asarray(priority)

    if peaks.size == 0:
        return np.zeros(0, dtype=bool)


    # Build a fake signal where each peak index has its priority value
    signal = np.zeros(np.max(peaks) + 1)
    signal[peaks] = priority

    # SciPy keeps the highest-priority peak in each neighborhood
    kept, _ = find_peaks(signal, distance=distance)

    # Return a boolean mask matching the original peaks array
    return np.isin(peaks, kept)


###

def progress_callback():
    print(".", end="", flush=True)





# --------------------------------------------------------------------------------------
# Run pipeline: load file -> detect events -> save CSV
# --------------------------------------------------------------------------------------

def build_run_settings(ts: float, direction: int = 1, threshold_lower: float = -150.0) -> Dict:
    """
    ts: sampling interval in seconds (from RawData.ts)
    direction: 1 for positive-going events, -1 for negative-going events
    """
    return {
        "ts": ts,
        "direction": direction,
        "next_event_baseline_idx" : None,

        # --- template shape ---
        "window_len_s": 20.0 / 1000,     # template window length
        "rise_s": 0.5 / 1000,
        "decay_s": 5.0 / 1000,
        "window_len_samples": int((20.0 / 1000) / ts),

        # --- Amplitude / Decay Settings (from GUI) ---
        "decay_search_period_s": 30.0 / 1000,
        "baseline_search_period_s": 30.0 / 1000,
        "amplitude_threshold": 2.0,  # pA
        "average_baseline_points": {"on": True, "value_s": 1.0 / 1000},
        "min_distance_between_maxima_samples": round((2.5 / 1000) / ts),

        # --- detection method ---
        "detection_threshold_type": "detection_criterion",
        "detection_criterion": 4.0,
        #"detection_threshold_type": "correlation",  # or "detection_criterion" / "deconvolution"

        # --- threshold checks used in check_putative_event / check_peak_against_threshold_lower ---
        "threshold_type": "manual",
        "threshold_lower": [threshold_lower],   # matches amplitude_threshold above
        "rec": 0,
        "threshold_upper_limit_on": False,
        "threshold_upper_limit_value": None,

        "baseline_type": "per_event",
        # ---------------------------------------------------------
        # Baseline for manual/curved/drawn modes
        # ---------------------------------------------------------
        #"baseline": 0.0,

        # --- omit periods (none by default) ---
        "omit_start_stop_times": [],
        "omit_start_stop_mode": "absolute",
        "omit_start_stop_include_exclude": "exclude",

        "corr_cutoff": 0.4,

        "average_peak_points": {
            "on": True,
            "value_s": 0.0005},

        "name" : "event_detection",

        # ---------------------------------------------------------
        # Template + record info
        # ---------------------------------------------------------
        "template_num": 0,
        "records_are_contiguous": True,
        "previous_event_idx": None,

        # ---------------------------------------------------------
        # Smoothing
        # ---------------------------------------------------------
        "smoothing_window_ms": 0.2,

        # ---------------------------------------------------------
        # Decay settings
        # ---------------------------------------------------------
        "decay_period_type": "auto_search_data",
        # NOTE: decay_search_period_s is already set above (30ms) — removed the
        # duplicate "0.010" (10ms) definition that was here and silently
        # overriding it, capping every decay_time_ms measurement at 10ms.
        "decay_or_biexp_fit_method": "monoexp",
        "endpoint_search_method": "entire_search_region",

        "decay_mode": "decay_one_point",
        "decay_times_baseline": "baseline",
        "decay_period_smooth_s": 0.001,
        "decay_amplitude_percent": 37.0,   # FIXED: percent (0-100), not fraction — underlying code divides by 100
        "decay_cutoff_low": 20.0,          # FIXED: percent (0-100), not fraction
        "decay_cutoff_high": 80.0,         # FIXED: percent (0-100), not fraction

        # ---------------------------------------------------------
        # Rise time settings (top-level, used by calculate_event_rise_time)
        # 10-90% rise time is the standard convention; adjust if you want
        # e.g. 20-80%.
        # ---------------------------------------------------------
        "rise_cutoff_low": 10.0,   # FIXED: percent (0-100), not fraction — was causing ~0ms rise times
        "rise_cutoff_high": 90.0,  # FIXED: percent (0-100), not fraction
        
        "interp_200khz": False,


        # ---------------------------------------------------------
        # Backend compatibility flags
        # ---------------------------------------------------------
        "from_fit_not_data": False,
        "edit_kinetics_mode": False,
        "legacy_options": {
            "baseline_method": False,
            "baseline_enhanced_position": False,
            "decay_detection_method": False,
        },

        # ---------------------------------------------------------
        # Monoexp fit options
        # ---------------------------------------------------------
        "monoexp_fit": {
            "maxfev": 10000,
            "ftol": 1e-6,
            "xtol": 1e-6,
            "gtol": 1e-6,

            "adjust_startpoint_r2_on": False,
            "adjust_startpoint_r2_value": 5,
            "adjust_startpoint_bounds_on": False,
            "adjust_startpoint_bounds_value": 5,

            "exclude_from_r2_on": False,
            "exclude_from_r2_value": 0.0,

            "exclude_if_params_not_in_bounds": False,
            "tau_cutoff_min": 0.05,
            "tau_cutoff_max": 50.0,
        },


        # ---------------------------------------------------------
        # Biexp fit options
        # ---------------------------------------------------------
        "biexp_fit": {
            "maxfev": 10000,
            "ftol": 1e-6,
            "xtol": 1e-6,
            "gtol": 1e-6,

            "adjust_startpoint_r2_on": False,
            "adjust_startpoint_r2_value": 5,
            "adjust_startpoint_bounds_on": False,
            "adjust_startpoint_bounds_value": 5,

            "exclude_from_r2_on": False,
            "exclude_from_r2_value": 0.0,

            "exclude_if_params_not_in_bounds": False,
            "rise_cutoff_low": 0.05,
            "rise_cutoff_high": 50.0,
            "decay_cutoff_min": 0.05,
            "decay_cutoff_max": 50.0,
        },

        # ---------------------------------------------------------
        # Canonical biexp defaults
        # ---------------------------------------------------------
        "cannonical_initial_biexp_coefficients": {
            "rise": 0.5,
            "decay": 5.0,
        },



        # ---------------------------------------------------------
        # Max slope settings
        # ---------------------------------------------------------
        "max_slope": {
            "on": False,
            "rise_num_samples": 5,
            "decay_num_samples": 5,
            "smooth": {
                "on": False,
                "window": 1},
            "use_baseline_crossing_endpoint": False,
        },

        # ---------------------------------------------------------
        # Area under curve (AUC) settings (REQUIRED)
        # ---------------------------------------------------------
        "area_under_curve": {
            "on": False,             # AUC thresholding disabled by default
            "value_pa_ms": 0.0,      # threshold value if enabled
            },





    }

def is_valid_abf(filepath: str, ext: str) -> bool:
    """
    Check whether ABF has the required number of sweeps and duration.
    """

    try:
        importer = ephys.importdata.ImportData(filepath, ext)
        data = importer.load_data()

        if data is None:
            print(f"[SKIP] Could not load {filepath}")
            return False

        # Check sweep duration
        sweep_duration = data.num_samples * data.ts


        print(
            f"[OK] {os.path.basename(filepath)}: "
            f"{data.num_recs} sweeps, {sweep_duration:.3f}s each"
        )

        return True

    except Exception as e:
        print(f"[SKIP] Error reading {filepath}: {e}")
        return False


def run_pipeline(
    filepath: str,
    ext: str,
    output_csv: str,
    direction: direction,
    threshold_lower: float = -150.0,
    filtfs: Optional[List[float]] = None,
) -> None:
    """
    Load `filepath`, run template-matching event detection, save events to `output_csv`.

    filtfs : list of band-stop centre frequencies in Hz (e.g. [50, 150] for
        mains + 3rd harmonic), applied via filter_trace before detection.
        Pass None (default) to skip filtering entirely.
    """
    #print(f"[DEBUG] Loading file: {filepath}")
    importer = ephys.importdata.ImportData(filepath, ext)
    data = importer.load_data()

    if data is None:
        print("[DEBUG] Failed to load data — check file path / extension.")
        return

    #print(f"[DEBUG] Loaded. num_recs={data.num_recs}, num_samples={data.num_samples}, ts={data.ts}, fs={data.fs}")

    run_settings = build_run_settings(ts=float(data.ts), direction=direction, threshold_lower=threshold_lower)
    #print(f"[DEBUG] run_settings built: {run_settings}")

    primary_data = data.get_primary_data()
    time_array = data.time_array
    time_array = time_array - 0.028

    summary_rows = []

    for rec in range(data.num_recs):
        #print(f"[DEBUG] Processing record {rec+1}/{data.num_recs}")
        run_settings["rec"] = rec
        run_settings["previous_event_idx"] = None   # <-- reset at start of each sweep

        data_array = primary_data[rec, :]
        time_ = time_array[rec, :]

        if filtfs:
            # sampling rate in Hz
            fs = 1.0 / float(data.ts)

            # don't filter the stimulation period
            # example: leave 9-15 seconds untouched
            excluded_index_ranges = [(int(4 * fs), int(10 * fs))]
            # dt (data.ts) is in seconds; filter_trace expects milliseconds,
            # matching the MATLAB filterMinis.m convention (Fs = 1000/dt).
            data_array = filter_trace(
                data_array,
                dt_ms=float(data.ts) * 1000.0,
                filtfs=filtfs,
                n_sweeps=1,  # each record is filtered individually here
                excluded_index_ranges=excluded_index_ranges,
            )

        betas, detection_coefs = ephys.event_analysis_master.fit_sliding_window(data_array, run_settings, progress_callback)
        #print(f"\n[DEBUG] Record {rec}: detection_coefs shape={detection_coefs.shape}")

        event_info = ephys.event_analysis_master.calculate_event_peaks(detection_coefs, betas, time_, data_array, run_settings)
        #print(f"[DEBUG] Record {rec}: event_info keys={list(event_info.keys()) if isinstance(event_info, dict) else 'N/A'}")

        # -----------------------------
        # FULL KINETICS (per record)
        # -----------------------------
        peak_idxs = [info["peak"]["idx"] for info in event_info.values()]
        time = time_
        current = data_array



        '''for idx in peak_idxs:

            peak_time = time[idx]
            peak_im = current[idx]

            run_settings["_doublet_correction_fired"] = False   # <-- reset before each event

            kinetics = ephys.event_analysis_master.calculate_event_kinetics(
                time,
                current,
                idx,
                peak_time,
                peak_im,
                template_num=0,
                records_are_contiguous=True,
                run_settings=run_settings,
                info=None
            )

            if kinetics is False:
                continue

            was_doublet_corrected = run_settings.get("_doublet_correction_fired", False)  # <-- read it

            baseline_im = kinetics["baseline"]["im"] if "baseline" in kinetics else kinetics["peak"]["im"]
            relative_im = kinetics["amplitude"]["im"] if "amplitude" in kinetics else 0.0

            summary_rows.append({
                "Event Number": len(summary_rows)+1,
                "Record": rec,
                "Event Time (s)": kinetics["peak"]["time"],
                "Baseline (pA)": kinetics["baseline"]["im"],
                "Peak (pA)": kinetics["peak"]["im"],
                "Amplitude (pA)": kinetics["amplitude"]["im"],
                "Rise Time (ms)": kinetics["rise"]["rise_time_ms"],
                "Half-Width (ms)": kinetics["half_width"]["fwhm_ms"],
                "Decay Time (ms)": kinetics["decay_one_point"]["decay_time_ms"],
                "Decay Fit b0": kinetics["monoexp_fit"]["b0"] if "monoexp_fit" in kinetics else None,
                "Decay Fit b1": kinetics["monoexp_fit"]["b1"] if "monoexp_fit" in kinetics else None,
                "Decay Fit Tau (ms)": kinetics["monoexp_fit"]["tau_ms"] if "monoexp_fit" in kinetics else None,
                "AUC (pA ms)": kinetics["area_under_curve"]["im"],
                "rise_max_time": kinetics["rise"]["max_time"],
                "baseline_time": kinetics["baseline"]["time"],
                "event_index": int(idx),
                "doublet_baseline_correction": was_doublet_corrected,   # <-- new column
                })'''

        for idx in peak_idxs:

            peak_time = time[idx]
            peak_im = current[idx]

            kinetics = ephys.event_analysis_master.calculate_event_kinetics(
                time,
                current,
                idx,
                peak_time,
                peak_im,
                template_num=0,                 # correct for single-template workflows
                records_are_contiguous=True,
                run_settings=run_settings,
                info=None
            )

            if kinetics is False:
                continue

            run_settings["previous_event_idx"] = idx   # <-- add this line



            # Fallback baseline if kinetics did not compute one
            baseline_im = kinetics["baseline"]["im"] if "baseline" in kinetics else kinetics["peak"]["im"]

            # Fallback relative amplitude if missing
            relative_im = kinetics["amplitude"]["im"] if "amplitude" in kinetics else 0.0

            summary_rows.append({
                "Event Number": len(summary_rows)+1,
                "Record": rec,
                "Event Time (s)": kinetics["peak"]["time"],
                "Baseline (pA)": kinetics["baseline"]["im"],
                "Peak (pA)": kinetics["peak"]["im"],
                "Amplitude (pA)": kinetics["amplitude"]["im"],
                "Rise Time (ms)": kinetics["rise"]["rise_time_ms"],
                "Half-Width (ms)": kinetics["half_width"]["fwhm_ms"],
                "Decay Time (ms)": kinetics["decay_one_point"]["decay_time_ms"],
                "Decay Fit b0": kinetics["monoexp_fit"]["b0"] if "monoexp_fit" in kinetics else None,
                "Decay Fit b1": kinetics["monoexp_fit"]["b1"] if "monoexp_fit" in kinetics else None,
                "Decay Fit Tau (ms)": kinetics["monoexp_fit"]["tau_ms"] if "monoexp_fit" in kinetics else None,
                "AUC (pA ms)": kinetics["area_under_curve"]["im"],
                "rise_max_time": kinetics["rise"]["max_time"],
                "baseline_time": kinetics["baseline"]["time"],
                "event_index": int(idx),
                })


    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_csv, index=False)
    print(f"[DEBUG] Saved {len(summary_df)} events to {output_csv}")


if __name__ == "__main__":

    # ----------------------------------------------------------
    # Process ALL ABF files in a folder
    # ----------------------------------------------------------

    INPUT_FOLDER = r"C:/Users/jane/Downloads/sort_stimulator_files/mech_stimulator_15s_stimAt5"
    OUTPUT_FOLDER = r"C:/Users/jane/Downloads/sort_stimulator_files/stimulator_analyzed"
    METADATA_CSV = r"C:/Users/jane/Desktop/Data_analysis/metadata_with_base_name_23_to_26.csv"

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    EXT = ".ABF"

    # Band-stop centre frequencies (Hz) to remove mains noise + harmonics,
    # matching the `filtfs = [50 150]` example from filterMinis.m. Set to
    # None to disable filtering.
    FILTFS = [50, 150]

    metadata_df = metadata_lookup.load_metadata(METADATA_CSV)

    # find all ABF files
    abf_files = [
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(".abf")
    ]

    #print(f"Found {len(abf_files)} ABF files")

    skipped_rows = []

    for i, filename in enumerate(abf_files):

        filepath = os.path.join(INPUT_FOLDER, filename)

        # ---------------------------------------
        # Only process ABFs with:
        # 4 sweeps × 25 seconds each
        # ---------------------------------------
        #if not is_valid_abf(filepath, EXT):
        #    continue

        # ---------------------------------------
        # Look up threshold_lower / direction from metadata (by mV).
        # Files with no metadata row, NA mV, or an mV not in
        # metadata_lookup.MV_SETTINGS_TABLE are skipped here and logged,
        # for separate handling.
        # ---------------------------------------
        settings = metadata_lookup.get_settings_for_file(filename, metadata_df)
        if settings is None:
            skipped_rows.append({"filename": filename})
            continue

        threshold_lower, direction, mv = settings

        # remove .abf extension
        file_prefix = os.path.splitext(filename)[0]

        # output csv name uses ABF file name
        output_csv = os.path.join(
            OUTPUT_FOLDER,
            f"{file_prefix}_unfiltered.csv"
        )

        #print("\n" + "="*60)
        print(f"Processing {i+1}/{len(abf_files)}")
        print(f"ABF : {filepath}")
        #print(f"CSV : {output_csv}")
        print(f"mV={mv}  ->  threshold_lower={threshold_lower}, direction={direction}")
        #print("="*60)

        try:
            run_pipeline(
                filepath,
                EXT,
                output_csv,
                direction=direction,
                threshold_lower=threshold_lower,
                filtfs=FILTFS,
            )

        except Exception as e:
            print(f"[ERROR] Failed on {filename}")
            print(e)

    if skipped_rows:
        skipped_path = os.path.join(OUTPUT_FOLDER, "skipped_files_metadata.csv")
        pd.DataFrame(skipped_rows).to_csv(skipped_path, index=False)
        print(f"\n{len(skipped_rows)} file(s) skipped (NA / unmatched mV) - see {skipped_path}")

    print("\nFinished processing all ABF files.")
