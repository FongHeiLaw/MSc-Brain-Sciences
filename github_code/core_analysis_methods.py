
from __future__ import annotations

import copy
from github_code import utils

import numpy as np
import scipy
from github_code import voltage_calc
from scipy import signal
from scipy.signal import find_peaks

# --------------------------------------------------------------------------------------
# Curve Fitting
# --------------------------------------------------------------------------------------

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



def monoexp_function(x: NpArray64, coefs: Tuple) -> NpArray64:

    b0, b1, tau = coefs

    return b0 + b1 * np.exp(-x / tau)


def biexp_decay_function(x: NpArray64, coefs: Tuple) -> NpArray64:

    b0, b1, tau1, b2, tau2 = coefs

    return b0 + b1 * np.exp(-x / tau1) + b2 * np.exp(-x / tau2)


def biexp_event_function(x: NpArray64, coefs: Tuple) -> NpArray64:

    b0, b1, rise, decay = coefs

    return b0 + b1 * (1 - np.exp(-x / rise)) * np.exp(-x / decay)

def triexp_decay_function(x: NpArray64, coefs: Tuple) -> NpArray64:

    b0, b1, tau1, b2, tau2, b3, tau3 = coefs

    return b0 + b1 * np.exp(-x / tau1) + b2 * np.exp(-x / tau2) + b3 * np.exp(-x / tau3)


def gaussian_function(x: NpArray64, coefs: Tuple) -> NpArray64:
    a, mu, sigma = coefs
    gauss = a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return gauss


# Least Squares Cost Functions
# --------------------------------------------------------------------------------------
# Note that scipy.optimize.least_squares expects the cost-function to be residuals
# not the sum of squared error (it handles this under the hood)
# --------------------------------------------------------------------------------------


def monoexp_least_squares_cost_function(coefs: Tuple, x: NpArray64, y: NpArray64) -> NpArray64:
    yhat = monoexp_function(x, coefs)
    return yhat - y


def biexp_event_least_squares_cost_function(coefs: Tuple, x: NpArray64, y: NpArray64) -> NpArray64:
    yhat = biexp_event_function(x, coefs)
    return yhat - y


def biexp_decay_least_squares_cost_function(coefs: Tuple, x: NpArray64, y: NpArray64) -> NpArray64:
    yhat = biexp_decay_function(x, coefs)
    return yhat - y


def triexp_decay_least_squares_cost_function(coefs: Tuple, x: NpArray64, y: NpArray64) -> NpArray64:
    yhat = triexp_decay_function(x, coefs)
    return yhat - y


def gaussian_least_squares_cost_function(coefs: Tuple, x: NpArray64, y: NpArray64) -> NpArray64:
    y_hat = gaussian_function(x, coefs)
    return y_hat - y


# Fit Curve ----------------------------------------------------------------------------


def fit_curve(
    analysis_name: str,
    x_to_fit: NpArray64,
    y_to_fit: NpArray64,
    direction: Direction,
    initial_est: Optional[Tuple] = None,
    bounds: Optional[Tuple] = None,
    normalise_time: Bool = True,
) -> Tuple[Union[Literal[False], Tuple], FalseOrFloat, FalseOrFloat]:

    if normalise_time:
        x_to_fit = x_to_fit - x_to_fit[0]  # x_to_fit is read only (no -=)

    if initial_est is None:
        initial_est = get_initial_est(analysis_name, direction, x_to_fit, y_to_fit)

    if bounds is None:
        bounds = get_curve_fit_bounds(analysis_name)

    try:
        coefs = scipy.optimize.least_squares(
            get_least_squares_fit_function(analysis_name),
            x0=initial_est,
            args=(x_to_fit, y_to_fit),
            bounds=bounds,
        )
        coefs = coefs.x
    except:
        return False, False, False

    func = get_fit_functions(analysis_name)
    fit = func(x_to_fit, coefs)
    r2 = calc_r2(y_to_fit, fit)

    return coefs, fit, r2


def calc_r2(y: NpArray64, y_hat: NpArray64) -> NpArray64:
    """
    Calculate the r-squared for a fit to data
    """
    y_bar = np.mean(y)
    SST = np.sum((y - y_bar) ** 2)
    RSS = np.sum((y - y_hat) ** 2)
    r2 = 1 - RSS / SST
    return utils.fix_numpy_typing(r2)



# Bounds and Initial Estimates
# --------------------------------------------------------------------------------------


def get_curve_fit_bounds(analysis_name: str) -> Tuple:
    """
    Set default bounds on curve fitting. For any Tau coefficients,
    set the start boundary to zero (avoid strange occurrences of negative
    tau when fit is poor).
    """
    bounds = {
        "monoexp": ((-np.inf, -np.inf, 0), (np.inf, np.inf, np.inf)),
        "biexp_decay": (
            (-np.inf, -np.inf, 0, -np.inf, 0),
            (np.inf, np.inf, np.inf, np.inf, np.inf),
        ),
        "biexp_event": ((-np.inf, -np.inf, 0, 0), (np.inf, np.inf, np.inf, np.inf)),
        "triexp": (
            (-np.inf, -np.inf, 0, -np.inf, 0, -np.inf, 0),
            (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
        ),
        "gaussian": ((-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf)),
    }

    return bounds[analysis_name]


def get_initial_est(analysis_name: str, direction: Direction, x_to_fit: NpArray64, y_to_fit: NpArray64) -> Tuple:

    if analysis_name == "monoexp":
        est_offset, est_slope, est_tau = get_exponential_function_initial_est(x_to_fit, y_to_fit)
        initial_est = (est_offset, est_slope, est_tau)

    elif analysis_name == "biexp_decay":
        est_offset, est_slope, est_tau = get_exponential_function_initial_est(x_to_fit, y_to_fit)
        initial_est = (est_offset, est_slope, est_tau * 0.1, est_slope, est_tau * 0.9)

    elif analysis_name == "biexp_event":
        initial_est = get_biexp_event_initial_est(x_to_fit, y_to_fit, direction)

    elif analysis_name == "triexp":
        est_offset, est_slope, est_tau = get_exponential_function_initial_est(x_to_fit, y_to_fit)
        initial_est = (
            est_offset,
            est_slope,
            est_tau / 3,
            est_slope,
            est_tau / 3,
            est_slope,
            est_tau / 3,
        )

    elif analysis_name == "gaussian":
        initial_est = get_gaussian_initial_est(x_to_fit, y_to_fit)

    return initial_est


def get_exponential_function_initial_est(
    x_to_fit: NpArray64, y_to_fit: NpArray64
) -> Tuple[NpArray64, NpArray64, NpArray64]:

    start = y_to_fit[0]
    end = y_to_fit[-1]

    est_offset = np.mean(y_to_fit)
    est_slope = start - end

    half_amp = start + ((end - start) / 2)
    rough_midpoint = np.argmin(np.abs(half_amp - y_to_fit))
    est_tau = x_to_fit[rough_midpoint] / np.log(2)

    return (
        utils.fix_numpy_typing(est_offset),
        utils.fix_numpy_typing(est_slope),
        utils.fix_numpy_typing(est_tau),
    )


def get_biexp_event_initial_est(x_to_fit: NpArray64, y_to_fit: NpArray64, direction: Direction) -> Tuple:

    default_rise = default_decay = None

    est_offset, est_slope, __ = get_exponential_function_initial_est(x_to_fit, y_to_fit)

    if direction == -1:
        est_slope = np.min(y_to_fit) - y_to_fit[0]
    elif direction == 1:
        est_slope = np.max(y_to_fit) - y_to_fit[0]
    initial_est = (est_offset, est_slope, default_rise, default_decay)

    return initial_est


def get_gaussian_initial_est(x: NpArray64, y: NpArray64) -> Tuple[NpArray64, NpArray64, NpArray64]:

    a = np.max(y)
    mu = np.sum(x * y) / np.sum(y)
    sigma = np.sqrt(np.sum(y * (x - mu) ** 2) / np.sum(y))
    return (
        utils.fix_numpy_typing(a),
        utils.fix_numpy_typing(mu),
        utils.fix_numpy_typing(sigma),
    )


# Convenience Functions
# --------------------------------------------------------------------------------------


def get_least_squares_fit_function(analysis_name: str) -> Callable:

    least_squares_fit_functions = {
        "monoexp": monoexp_least_squares_cost_function,
        "biexp_decay": biexp_decay_least_squares_cost_function,
        "biexp_event": biexp_event_least_squares_cost_function,
        "triexp": triexp_decay_least_squares_cost_function,
        "gaussian": gaussian_least_squares_cost_function,
    }
    return least_squares_fit_functions[analysis_name]


def get_fit_functions(analysis_name: str) -> Callable:
    """
    Convenient access for fitting functions.
    """
    fit_functions = {
        "monoexp": monoexp_function,
        "biexp_decay": biexp_decay_function,
        "biexp_event": biexp_event_function,
        "triexp": triexp_decay_function,
        "gaussian": gaussian_function,
    }

    return fit_functions[analysis_name]


# --------------------------------------------------------------------------------------
# Spike Detection - local maximum period thresholding
# --------------------------------------------------------------------------------------

def exclude_peak_idxs_based_on_local_maximum_period(
    peak_idxs: NDArray[np.integer], peaks_im: NpArray64, min_distance: Int
) -> NDArray[np.integer]:

    
    # keep = _peak_finding_utils._select_by_peak_distance(peak_idxs.astype(np.intp), peaks_im, float(min_distance))

    # Use  my new Python version of select_by_peak_distance()
    keep = select_by_peak_distance(
        peak_idxs.astype(int),
        peaks_im,
        float(min_distance)
    )


    return peak_idxs[keep]  # type: ignore


def first_derivative_threshold(cfgs: ConfigsClass, first_deriv: NpArray64) -> Int:
    if cfgs.skinetics["first_deriv_max_or_cutoff"] == "max":
        idx = np.nanargmax(first_deriv)
    else:
        cutoff = cfgs.skinetics["first_deriv_cutoff"]
        idx = (first_deriv > cutoff).argmax()
    return idx


def third_derviative_threshold(cfgs: ConfigsClass, third_deriv: NpArray64) -> Int:
    if cfgs.skinetics["third_deriv_max_or_cutoff"] == "max":
        idx = np.nanargmax(third_deriv)
    else:
        cutoff = cfgs.skinetics["third_deriv_cutoff"]
        idx = (third_deriv > cutoff).argmax()
    return idx


def method_i_threshold(cfgs: ConfigsClass, first_deriv: NpArray64, second_deriv: NpArray64) -> Int:
    g = second_deriv / first_deriv
    lower_bound = cfgs.skinetics["method_I_lower_bound"]
    g[first_deriv <= lower_bound] = np.nan

    if np.all(np.isnan(g)):
        return False

    idx = np.nanargmax(g)
    return idx


def method_ii_threshold(
    cfgs: ConfigsClass,
    first_deriv: NpArray64,
    second_deriv: NpArray64,
    third_deriv: NpArray64,
) -> Int:

    num = third_deriv * first_deriv - second_deriv**2
    denom = first_deriv**3

    h = np.divide(num, denom, out=np.zeros_like(num, dtype=np.float64), where=denom != 0)

    lower_bound = cfgs.skinetics["method_II_lower_bound"]

    h[first_deriv <= lower_bound] = 0

    if not np.any(h):
        return False

    idx = np.argmax(h)

    return idx


def leading_inflection_threshold(first_deriv: NpArray64) -> Int:
    idx = np.nanargmin(first_deriv)
    return idx


def max_curvature_threshold(first_deriv: NpArray64, second_deriv: NpArray64) -> Int:
    k = second_deriv * (1 + first_deriv**2) ** (-3 / 2)
    idx = np.nanargmax(k)
    return idx

def calculate_fwhm(
    start_to_peak_time: NpArray64,
    start_to_peak_data: NpArray64,
    peak_to_end_time: NpArray64,
    peak_to_end_data: NpArray64,
    half_amp: NpArray64,
    interp: bool,
) -> Tuple[NpArray64, NpArray64, NpArray64, NpArray64, NpArray64, np.integer, np.integer]:

    # save the un-interpolated index to calculate average event from
    orig_rise_mid_idx = np.abs(half_amp - start_to_peak_data).argmin()
    orig_decay_mid_idx = np.abs(half_amp - peak_to_end_data).argmin()

    if interp:
        start_to_peak_data, start_to_peak_time = twohundred_kHz_interpolate(start_to_peak_data, start_to_peak_time)
        peak_to_end_data, peak_to_end_time = twohundred_kHz_interpolate(peak_to_end_data, peak_to_end_time)

    rise_mid_idx = np.abs(half_amp - start_to_peak_data).argmin()
    decay_mid_idx = np.abs(half_amp - peak_to_end_data).argmin()

    rise_midpoint = start_to_peak_data[rise_mid_idx]
    rise_midtime = start_to_peak_time[rise_mid_idx]
    fall_midpoint = peak_to_end_data[decay_mid_idx]
    fall_midtime = peak_to_end_time[decay_mid_idx]
    fwhm = fall_midtime - rise_midtime

    return (
        rise_midtime,
        rise_midpoint,
        fall_midtime,
        fall_midpoint,
        fwhm,
        orig_rise_mid_idx,
        orig_decay_mid_idx,
    )

def calculate_max_slope_rise_or_decay(
    time_: NpArray64,
    data: NpArray64,
    start_idx: Int,
    stop_idx: Int,
    window_samples: Int,
    ts: NpArray64,
    smooth_settings: Dict,
    argmax_func: Callable,
) -> Tuple[NpArray64, ...]:

    stop_idx -= window_samples
    n_samples = stop_idx - start_idx + 1

    if n_samples < 1:
        return np.array([np.nan]), np.array([np.nan]), np.array([np.nan])

    if smooth_settings["on"]:
        data = voltage_calc.quick_moving_average(data, smooth_settings["num_samples"])
    if window_samples == 2:
        idx = np.arange(start_idx, stop_idx + 1)
        time_step_ms = ts * 1000

        windowed_deriv = (data[idx + 1] - data[idx]) / time_step_ms
        max_slope_idx = argmax_func(windowed_deriv)
        max_slope_ms = windowed_deriv[max_slope_idx]
        max_slope_idx += start_idx

        fit_time = np.array([time_[max_slope_idx], time_[max_slope_idx + 1]])
        fit_data = np.array([data[max_slope_idx], data[max_slope_idx + 1]])
    else:
        idx = (
            np.tile(np.arange(start_idx, start_idx + window_samples), (n_samples, 1))
            + np.atleast_2d(np.arange(n_samples)).T
        )
        all_x = time_[idx]
        all_y = data[idx]

        params = utils.np_empty_nan((n_samples, 2))
        for i in range(n_samples):
            x = all_x[i, :]
            y = all_y[i, :]
            params[i, 0], params[i, 1], __, __, __ = scipy.stats.linregress(x, y)

        max_slope_idx = argmax_func(params[:, 0])
        slope, intercept = params[max_slope_idx, :]

        fit_time = all_x[max_slope_idx, :]
        fit_data = fit_time * slope + intercept
        max_slope_ms = slope / 1000

    return max_slope_ms, fit_time, fit_data


def twohundred_kHz_interpolate(
    vm: NpArray64,
    time_: NpArray64,
    khz_interpolate: float = 200000.0,
    interp_method: str = "linear",
) -> Tuple[NpArray64, NpArray64]:
    """
    Interpolate to 200 kHz with linear interpolation.
    """
    fs, _ = calc_fs_and_ts(time_)
    interp_factor = khz_interpolate / fs
    vm = interpolate_data(vm, time_, interp_method, interp_factor, 0)
    time_array = interpolate_data(time_, time_, "linear", interp_factor, 0)

    return vm, time_array


def area_under_curve_ms(y: NpArray64, ts: NpArray64) -> Tuple[NpArray64, NpArray64]:

    ts_ms = ts * 1000

    area_under_curve_time_ms = ts_ms * (y.size - 1)

    area_under_curve = np.trapz(y, dx=ts_ms)  # type: ignore

    return area_under_curve, area_under_curve_time_ms



def filter_data(
    data: NpArray64,
    fs: Union[float, NpArray64],
    filter_: Literal["bessel", "butter"],
    order: Int,
    cutoff_hz: Union[float, NpArray64, Tuple],
    btype: str,
    axis: Int,
) -> NpArray64:

    if filter_ == "bessel":
        sos = get_bessel(fs, order, cutoff_hz, btype)
    elif filter_ == "butter":
        sos = get_butterworth(fs, order, cutoff_hz, btype)

    filtered_data = scipy.signal.sosfiltfilt(sos, data, axis=axis)

    return filtered_data


def get_bessel(
    fs: Union[float, NpArray64],
    order: Int,
    cutoff_hz: Union[float, NpArray64, Tuple],
    btype: str,
) -> NpArray64:
    sos = scipy.signal.bessel(N=order, Wn=cutoff_hz, btype=btype, output="sos", fs=fs)
    return sos


def get_butterworth(
    fs: Union[float, NpArray64],
    order: Int,
    cutoff_hz: Union[float, NpArray64, Tuple],
    btype: str,
) -> NpArray64:
    sos = scipy.signal.butter(N=order, Wn=cutoff_hz, btype=btype, output="sos", fs=fs)
    return sos

def interpolate_data(
    data_to_interp: NpArray64,
    time_array: Union[NDArray[np.integer], NpArray64],
    interp_method: str,
    interp_factor: Union[Int, NpArray64],
    axis: Int,
) -> NpArray64:


    interp_f = scipy.interpolate.interp1d(time_array, data_to_interp, axis=axis, kind=interp_method)

    n = len(time_array)
    new_sample_num = interp_factor * (n - 1) + 1

    time_interp = np.linspace(time_array[0], time_array[-1], int(new_sample_num))

    interp_data = interp_f(time_interp)

    return interp_data


def fit_polynomial(x: NDArray, y: NpArray64, poly_order: Int, progress_bar_callback=None) -> NpArray64:

    if y.ndim == 1:
        coefs = np.polyfit(x, y, deg=poly_order)
        fit = np.polyval(coefs, x)
        fit = np.reshape(fit, (1, len(fit)))
    else:
        coefs = np.polyfit(x, y.T, deg=poly_order).T

        fit = np.zeros(y.shape, dtype=np.float64)
        for irow in range(y.shape[0]):
            fit[irow, :] = np.polyval(coefs[irow, :], x)

            if progress_bar_callback is not None:
                progress_bar_callback()

    return fit


def cut_trace_and_normalise_time(
    time_array: NpArray64,
    new_start_sample: Union[NDArray[np.integer], Int],
    new_stop_sample: Union[NDArray[np.integer], Int],
) -> NpArray64:

    norm_time_array = copy_and_normalise_time_array(time_array)
    norm_time_array = norm_time_array[:, new_start_sample:new_stop_sample]
    norm_time_array = norm_time_array - norm_time_array[0][0]

    return norm_time_array


def copy_and_normalise_time_array(time_array: NpArray64) -> NpArray64:

    time_array = copy.deepcopy(time_array)
    time_array[1:None, :] = time_array[0, :]
    return time_array


# Reshape Data
# --------------------------------------------------------------------------------------


def reshape(data: NpArray64, num_samples: Int, num_records: Int) -> NpArray64:
    reshaped_data = np.reshape(data, [num_records, int(num_samples / num_records)])

    return reshaped_data



def get_bins_and_limits_for_custom_binsize(data, settings, parameter):
    """
    Custom binsize returns the full array of bins, giving more control
    that passing the number of bins to the underlying numpy functions.
    It also exposes the ability to adjust the starting bin value.

    see calc_cumulative_probability_or_histogram() for inputs

    limits: tuple
        (min, max) value of the bins. Only used for custom_binsize.
    """
    max_ = np.max(data)

    if settings["fix_start_bin"][parameter]["on"]:
        min_ = settings["fix_start_bin"][parameter]["value"]
        if min_ >= max_:
            min_ = np.min(data)
        limits = (min_, max_)
    else:
        limits = (np.min(data), max_)

    if settings["custom_binsize"][parameter] == 0:
        bins_or_num_bins = int(get_auto_num_bins(data))
    else:
        bins_or_num_bins = np.arange(
            limits[0], limits[1] + settings["custom_binsize"][parameter], settings["custom_binsize"][parameter]
        )

    return limits, bins_or_num_bins


def get_num_bins_from_settings(data: NpArray64, settings: Dict) -> Int:

    binning_method = settings["binning_method"]
    n_samples = len(data)

    if binning_method == "auto":
        num_bins = int(get_auto_num_bins(data))

    elif binning_method == "custom_binnum":
        num_bins = int(settings["custom_binnum"]) if settings["custom_binnum"] <= n_samples else n_samples

    elif binning_method == "num_events_divided_by":
        num_bins = int(np.round(len(data) / settings["divide_by_number"]))

    if num_bins > len(data) or num_bins == 0:
        num_bins = len(data)
    elif num_bins < 2:
        num_bins = 2

    return num_bins


def get_auto_num_bins(data: NpArray64) -> NpArray64:
    """
    Get bin numbers for cumulative frequency / histogram
    plots automatically.
    """
    try:
        num_bins = len(np.histogram_bin_edges(data, bins="auto"))
    except MemoryError:
        # can occur with this function when bin sizes are very small.
        num_bins = len(np.histogram_bin_edges(data, bins="sqrt"))
    return num_bins


def calc_histogram(data: NpArray64, bins_or_num_bins: Int, limits: Tuple) -> Tuple[NpArray64, NpArray64, NpArray64]:
    """ """
    if isinstance(bins_or_num_bins, int):
        info = np.histogram(data, bins=bins_or_num_bins, range=limits)
    else:
        info = np.histogram(data, bins=bins_or_num_bins)

    y_values = info[0]
    bin_edges = info[1]
    binsize = bin_edges[1] - bin_edges[0]

    return y_values, bin_edges, binsize


def calc_cumulative_probability(
    data: NpArray64, bins_or_num_bins: Int | NpArray64, limits: Tuple
) -> Tuple[NpArray64, NpArray64, NpArray64]:

    counts, bin_edges, binsize = calc_histogram(data, bins_or_num_bins, limits)

    pdf = counts / np.sum(counts)
    cdf = np.cumsum(pdf)

    return cdf, bin_edges, binsize


def format_bin_edges(bin_edges: NpArray64, x_axis_display: str) -> NpArray64:

    if x_axis_display == "bin_centre":
        x_values = get_bin_centers(bin_edges)
    elif x_axis_display == "left_edge":
        x_values = bin_edges[0:-1]
    elif x_axis_display == "right_edge":
        x_values = bin_edges[1:]

    return x_values


def get_bin_centers(bin_edges: NpArray64) -> NpArray64:
    return (bin_edges[:-1] + bin_edges[1:]) / 2



def handle_ks_test_results_tuple(results_tuple: Tuple, n: Int, format_p_for_gui: bool) -> Dict:
    results = {
        "statistic": results_tuple[0],
        "pvalue": results_tuple[1],
        "n": n,
    }

    if format_p_for_gui:
        results["pvalue"] = format_p_value(results["pvalue"])

    return results


def format_p_value(p: NpArray64) -> str:

    if p >= 1e-08:
        format_p = "{0:.8f}".format(p).rstrip("0")
    elif p < 1e-08:
        format_p = "<1e-08"
    return format_p

def remove_stimulus_artefact_from_record(
    data: NpArray64,
    ts: NpArray64,
    peak_detection_settings: Dict,
    forward_search_samples: Int,
    back_search_samples: Int,
    back_search_threshold: NpArray64,
    end_of_stimulus_perc_median: NpArray64,
    window_pad: Int = 1,
) -> NpArray64:
    """
    See get_stimulus_artefact_removed_data() for arguments.
    """
    main_peaks = calculate_stimulus_artefact_peaks(data, peak_detection_settings, ts)

    # For each peak, determine the local mean and median, and
    # forward and backward ranges to fill in.
    all_idx = []
    for peak_idx in main_peaks:
        left_edge, right_edge = get_artefact_peak_window_edges(data, peak_idx, forward_search_samples, window_pad)

        local_dat = data[np.arange(left_edge, right_edge)]

        diff_median_, median_ = calculate_local_statistics(local_dat)

        # Backward range
        back_search_start_idx = peak_idx - back_search_samples
        if back_search_start_idx < 0:
            back_search_start_idx = 0

        back_search_range = np.arange(back_search_start_idx, peak_idx + 1)
        back_search_range_diff_ms = np.abs(np.diff(data[back_search_range])) / (ts * 1000)

        over_thr_idx = back_search_start_idx + np.where(back_search_range_diff_ms > back_search_threshold)[0]
        if np.any(over_thr_idx):
            bad_back_idx = np.arange(np.min(over_thr_idx), peak_idx + 1)
        else:
            bad_back_idx = np.array([])

        # Forward Range
        perc_median = np.abs(diff_median_ * (end_of_stimulus_perc_median / 100))

        forward_search_range = data[peak_idx : peak_idx + forward_search_samples + 1]
        below_threshold = np.where(np.abs(forward_search_range - median_) < perc_median)[0]

        if np.any(below_threshold):
            forward_end_idx = peak_idx + np.min(np.where(np.abs(forward_search_range - median_) < perc_median)[0])
        else:
            forward_end_idx = data.size - 1

        bad_forward_idx = np.arange(peak_idx, forward_end_idx + 1)
        all_idx.append([bad_back_idx, bad_forward_idx, median_, diff_median_])

    # For all sections of data, fill in with Gaussian
    # random samples
    for idx_pair in all_idx:
        bad_back_idx, bad_forward_idx, median_, diff_median_ = idx_pair

        if np.any(bad_forward_idx):
            data[bad_forward_idx] = np.random.normal(median_, diff_median_, bad_forward_idx.size)

        if np.any(bad_back_idx):
            data[bad_back_idx] = np.random.normal(median_, diff_median_, bad_back_idx.size)

    return data


def calculate_stimulus_artefact_peaks(
    data: NpArray64, peak_detection_settings: Dict, ts: NpArray64
) -> NDArray[np.integer]:

    if peak_detection_settings["method"] == "derivative":
        forward_search_range_diff_ms = np.abs(np.diff(data)) / (ts * 1000)

        # shift diff
        main_peaks = np.where(forward_search_range_diff_ms > peak_detection_settings["derivative_threshold"])[0] + 1

    elif peak_detection_settings["method"] == "abs_threshold":
        if peak_detection_settings["abs_threshold_direction"] == "positive":
            main_peaks = np.where(data > peak_detection_settings["abs_threshold"])[0]

        elif peak_detection_settings["abs_threshold_direction"] == "negative":
            main_peaks = np.where(data < peak_detection_settings["abs_threshold"])[0]

    if len(main_peaks) > 1:
        main_peaks = strip_repeats(data, main_peaks)

    return main_peaks


def calculate_local_statistics(local_dat: NpArray64) -> Tuple[NpArray64, NpArray64]:

    diffs = np.abs(np.diff(local_dat))
    diff_median_ = np.median(diffs)
    median_ = np.median(local_dat)
    return utils.fix_numpy_typing(diff_median_), utils.fix_numpy_typing(median_)


def strip_repeats(data: NpArray64, main_peaks: NDArray[np.integer]) -> NDArray[np.integer]:

    bools = np.diff(np.r_[np.inf, main_peaks]) == 1
    tags = np.diff(bools.astype("int"))
    ups = np.where(tags == 1)[0]
    downs = np.where(tags == -1)[0]

    if not any(ups):
        return main_peaks

    if ups.size > downs.size:
        downs = np.r_[downs, tags.size]

    idx = [np.arange(up, down + 1) for up, down in zip(ups, downs)]

    main_peaks_repeated = np.array([idxs[np.argmax(data[main_peaks[idxs]])] for idxs in idx]).ravel()

    single_idx = np.ones(main_peaks.size, dtype=bool)
    single_idx[np.hstack(idx)] = False

    main_peaks = np.r_[main_peaks[main_peaks_repeated], main_peaks[single_idx]]

    return main_peaks


def get_artefact_peak_window_edges(
    data: NpArray64, peak_idx: Int, forward_search_samples: Int, pad: Int
) -> Tuple[Int, Int]:

    left_edge = peak_idx - pad
    right_edge = peak_idx + forward_search_samples + pad

    if left_edge < 0:
        left_edge = 0
        right_edge = pad * 2

    if right_edge > data.size:
        right_edge = data.size
        left_edge = peak_idx - (pad * 2)

    if left_edge < 0:
        left_edge = 0

    return left_edge, right_edge


def check_if_record_time_is_contiguous(min_max_time: NpArray64, num_recs: int, ts: NpArray64) -> bool:

    if num_recs == 1:
        return True

    if np.unique(min_max_time[:, 0]).size == 1:
        # a 'normalised' time recording
        return False

    diffs = min_max_time[1:, 0] - min_max_time[:-1, 1]

    is_contiguous = np.allclose(diffs, ts, rtol=0, atol=1e-10)

    return is_contiguous


# --------------------------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------------------------


def calc_fs_and_ts(time_: NpArray64) -> Tuple[NpArray64, ...]:
    ts = time_[1] - time_[0]
    fs = 1 / ts
    return fs, ts


def nearest_point_euclidean_distance(
    x1: NpArray64, timepoints: NpArray64, y1: NpArray64, datapoints: NpArray64
) -> NpArray64:

    time_std = np.std(timepoints)
    data_std = np.std(datapoints)
    distance = np.sqrt(((y1 - datapoints) / data_std) ** 2 + ((x1 - timepoints) / time_std) ** 2)

    return distance


def quick_get_time_in_samples(ts: NpArray64, timepoint: Union[float, NpArray64]) -> Int:
    return int(np.round(timepoint / ts))

def generate_time_array(
    start: float,
    stop: float,
    num_samples: Int,
    known_ts: NpArray64,
    start_stop_time_in_ms: bool = False,
) -> NpArray64:

    if start_stop_time_in_ms:
        start /= 1000
        stop /= 1000

    time_array = np.linspace(start, (stop - known_ts), num_samples)

    if start_stop_time_in_ms:
        time_array *= 1000

    return time_array


def get_conversion_to_pa_table() -> Dict:
    """
    Reference table for converting units to pA on file load (see importdata.py)
    """
    conversion_to_pa_table = {
        "fA": 1e-3,
        "nA": 1e3,
        "uA": 1e6,
        "mA": 1e9,
        "A": 1e12,
    }
    return conversion_to_pa_table


def get_conversion_to_mv_table() -> Dict:
    """
    Reference table for converting units to mV on file load (see importdata.py)
    """
    conversion_to_mv_table = {
        "pV": 1e-9,
        "nV": 1e-6,
        "uV": 1e-3,
        "V": 1e3,
    }
    return conversion_to_mv_table


def total_num_events(event_info: Union[List[Dict], Info], return_per_rec: bool = False) -> NDArray[np.integer]:
    """
    Calculate the total number of events in the event info
    (rec of dicts) or spkcnt / skinetics Info type
    either summed or per-rec
    """
    num_recs = len(event_info)
    per_rec = np.zeros(num_recs, dtype=int)
    for rec in range(num_recs):
        rec_info = event_info[rec]
        if rec_info != 0 and any(rec_info):
            per_rec[rec] = len(rec_info)

    events_to_return = np.sum(per_rec).astype(int) if not return_per_rec else per_rec

    return events_to_return

