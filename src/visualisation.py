import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import napari
import os
import imageio
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib import cm
from typing import Union, Optional, Sequence, Tuple, Callable, Iterable, Any, List, Dict
import matplotlib
import pandas as pd

from response_tabel_creator import stim_avg_flexible_1d

SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def save_stim_to_csv(
    dict_avg: dict,
    dirtosave: Path,
    mouse: str,
    protocol: str,
    exp: str,
    state: str,
    stim_type: str,
    signal: str,
    rate: int = 20,
):
    """
    Save stimulus-aligned traces to CSV with metadata.
    """
    df = pd.DataFrame(dict_avg).T.reset_index()
    time = [f"{t:.2f}" for t in np.arange(0, len(dict_avg['st1']) / rate, 1 / rate)]
    df.columns = ['stim#'] + time

    df.insert(0, "mouse", mouse)
    df.insert(0, "protocol", protocol)
    df.insert(0, "exp", exp)
    df.insert(0, "state", state)
    df.insert(0, "hemisphere", 'L' if stim_type.startswith('R') else 'R')
    df.insert(0, "stim_type", stim_type[1:])
    df.insert(0, "signal", signal)

    fname = dirtosave / f'{signal}_stim_info.csv'
    if not fname.exists():
        df.to_csv(fname, index=False)
    else:
        existing = pd.read_csv(fname)
        pd.concat([existing, df], axis=0).to_csv(fname, index=False)

def process_signal(
    stim_signal: np.ndarray,
    signal_label: str,
    dir_save: Union[str, Path],
    mouse: str,
    protocol: str,
    exp: str,
    state: str,
    stim_type: str,
    stim_cfg: dict,
    save_csv: bool = True
) -> None:
    """
    Processes a 1D stimulation signal and optionally saves the averaged response to CSV.

    This function extracts stimulus-triggered windows from a 1D signal (e.g., global Ca or HBT trace),
    computes averaged responses across repetitions, and saves the results to disk.

    Args:
        stim_signal (np.ndarray): The 1D signal array (shape: [T]) to be processed.
        signal_label (str): A label for the signal (e.g., 'Ca', 'HBT'), used in the filename.
        dir_save (Union[str, Path]): Directory where CSV output should be saved.
        mouse (str): Mouse ID.
        protocol (str): Full protocol name.
        exp (str): Experiment ID.
        state (str): Experiment state (e.g., 'awake', 'anesthetized').
        stim_type (str): Stimulation type string used for region identification.
        stim_cfg (dict): Configuration for stimulation averaging (e.g., frame timings, baseline subtraction).
        save_csv (bool, optional): Whether to save the output to CSV. Defaults to True.

    Returns:
        None
    """
    stim_dict, _ = stim_avg_flexible_1d(
        dataset=stim_signal,
        stim_start_frame=stim_cfg['stim_start_frame'],
        stim_count=stim_cfg['stim_count'],
        stim_step=stim_cfg['stim_step'],
        stim_window=tuple(stim_cfg['stim_window']),
        subtr_baseline=stim_cfg['subtr_baseline'],
        subtr_baseline_interval=stim_cfg['subtr_baseline_interval']
    )
    os.makedirs(dir_save, exist_ok=True)
    
    if save_csv:
        save_stim_to_csv(
            stim_dict,
            dir_save,
            mouse, protocol, exp, state,
            stim_type,
            signal = signal_label
        )

def summarize_stim_dict_by_type(
    stim_dict: dict,
    signal_type: str = 'Ca',
    continuous: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Summarizes or averages stimulus windows from a dictionary of responses.

    This function extracts and sums (or averages) temporal windows from 
    a stimulus dictionary for calcium or hemoglobin signals.

    Args:
        stim_dict (dict): Dictionary where each value is a 3D array [T, Y, X] 
            representing one stimulus repetition.
        signal_type (str): Type of signal to process — either 'Ca' for calcium
            or 'Hb', 'HBO', 'HBR', or 'HBT' for hemoglobin signals.
        continuous (bool): For calcium signals, whether to use a continuous window 
            (0–40 frames) or three short repeated windows [(20–24), (40–44), (60–64)].

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - sum_img: Total summed image across all stimuli [Y, X].
            - mean_img: Averaged image across stimuli [Y, X].

    Warns:
        Prints a warning if a selected window exceeds the stimulus array length.
    """
    sum_img = None
    count = 0

    for stim_clip in stim_dict.values():  # [T, Y, X]
        if signal_type.lower().startswith('ca'):
            if continuous:
                windows = [(0, 40)]
            else:
                windows = [(20, 24), (40, 44), (60, 64)]
        else:
            windows = [(0, 100)]

        stim_sum = np.zeros_like(stim_clip[0])  # [Y, X]
        for start, end in windows:
            if end <= stim_clip.shape[0]:
                stim_sum += np.sum(stim_clip[start:end], axis=0)
            else:
                print(f"Warning: window ({start}:{end}) out of bounds for stim of shape {stim_clip.shape}")

        if sum_img is None:
            sum_img = np.zeros_like(stim_sum)
        sum_img += stim_sum
        count += 1

    mean_img = sum_img / count if count > 0 else None
    return sum_img, mean_img

def average_stimulus_from_dict(stim_dict: dict) -> np.ndarray:
    """
    Averages stimulus responses over repetitions from a dictionary.

    Takes a dictionary of stimulus clips (each a 3D array [T, Y, X])
    and returns the average response across all repetitions.

    Args:
        stim_dict (dict): Dictionary with keys as stimulus labels and 
            values as 3D numpy arrays of shape [T, Y, X].

    Returns:
        np.ndarray: Averaged stimulus of shape [T, Y, X].

    Raises:
        ValueError: If the dictionary is empty or if stimuli have mismatched time dimensions.
    """
    if not stim_dict:
        raise ValueError("stim_dict is empty")
    stim_list = list(stim_dict.values())
    lengths = [stim.shape[0] for stim in stim_list]
    if len(set(lengths)) > 1:
        raise ValueError("All stimuli must have the same number of frames (T)")
    stacked = np.stack(stim_list, axis=0)  # [N, T, Y, X]
    return np.nanmean(stacked, axis=0)     # [T, Y, X]


def save_single_roi_as_bmp(mask_array: np.ndarray, save_path: str) -> None:
    """
    Saves a single ROI mask as a BMP image.

    The input mask is binarized and saved as a grayscale BMP file, where
    the region of interest is white (255) and the background is black (0).

    Args:
        mask_array (np.ndarray): A 2D array representing the ROI mask.
            Non-zero values are considered part of the ROI.
        save_path (str): Path where the BMP image will be saved.

    Returns:
        None
    """
    # Binarize the mask → Convert non-zero values to 255
    mask_binary = (mask_array > 0).astype(np.uint8) * 255
    Image.fromarray(mask_binary).save(save_path)
    print(f"ROI mask saved as BMP: {save_path}")


def apply_mask_and_average(signal_3d: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
    """
    Applies a 2D mask to a 3D signal and computes the spatial average at each time point.

    The mask is applied across all time frames, selecting only the regions where the mask is True.
    The function then returns the mean signal over those masked spatial locations for each time frame.

    Args:
        signal_3d (np.ndarray): 3D array with shape [T, H, W], representing time-series image data.
        mask_2d (np.ndarray): 2D boolean array with shape [H, W], where True indicates regions to include.

    Returns:
        np.ndarray: 1D array of length T, representing the average signal over masked area at each time frame.
    """
    masked = np.where(mask_2d, signal_3d, np.nan)
    return np.nanmean(masked, axis=(1, 2))

def average_stim_repeats(
    array: np.ndarray,
    fps: int,
    window_sec: float,
    start_sec: float = 0.0,
    pre_stim_sec: float = 0.0,
    return_all: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Segments a 1D signal into repeated stimulation trials and averages them over a defined time window.

    This function is commonly used to analyze repeated stimulus-evoked signals by segmenting the time-series
    into equal-length windows, aligned with stimulus onset times, and computing the average response.

    Args:
        array (np.ndarray): 1D array of shape (T,) representing the time-series signal.
        fps (int): Frames per second (sampling rate).
        window_sec (float): Duration of each trial window in seconds.
        start_sec (float, optional): Time (in seconds) to the start of the first stimulation. Default is 0.0.
        pre_stim_sec (float, optional): Time before stimulation to include (e.g., for baseline). Default is 0.0.
        return_all (bool, optional): If True, returns both the averaged response and all individual trials.

    Returns:
        np.ndarray or tuple of (np.ndarray, np.ndarray):
            - If return_all is False: 1D array of shape (window_length,) with the averaged signal.
            - If return_all is True: A tuple (average, all_trials) where:
                - average: 1D array of shape (window_length,)
                - all_trials: 2D array of shape (num_trials, window_length)

    Raises:
        ValueError: If input array is not 1D or if pre_stim_sec causes indexing before start of array.
    """
    if array.ndim != 1:
        raise ValueError("Only 1D arrays (shape: [T,]) are supported.")

    frames_per_window = int(fps * window_sec)
    pre_stim_frames = int(fps * pre_stim_sec)
    start_frame = int(fps * start_sec) - pre_stim_frames

    if start_frame < 0:
        raise ValueError("pre_stim_sec is too large — results in negative start index.")

    total_frames = array.shape[0]
    available_frames = total_frames - start_frame
    num_repeats = available_frames // frames_per_window
    end_frame = start_frame + num_repeats * frames_per_window

    trimmed = array[start_frame:end_frame]
    reshaped = trimmed.reshape((num_repeats, frames_per_window))

    avg = reshaped.mean(axis=0)
    return (avg, reshaped) if return_all else avg

def adjust_yaxis(ax: matplotlib.axes.Axes, ydiff: float, v: float) -> None:
    """
    Adjusts the Y-axis of a Matplotlib Axes object to better align vertically offset plots.

    This is typically used when plotting multiple signals with vertical shifts to ensure they are
    scaled properly relative to each other and don't compress/stretch disproportionately.

    Args:
        ax (matplotlib.axes.Axes): The axes object whose Y-axis needs to be adjusted.
        ydiff (float): Vertical spacing to account for (in display coordinates).
        v (float): Vertical offset applied to the current plot.

    Returns:
        None
    """
    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, ydiff))
    miny, maxy = ax.get_ylim()
    miny -= v
    maxy -= v

    if -miny > maxy or (-miny == maxy and dy > 0):
        nminy = miny
        nmaxy = miny * (maxy + dy) / (miny + dy)
    else:
        nmaxy = maxy
        nminy = maxy * (miny + dy) / (maxy + dy)

    ax.set_ylim(nminy + v, nmaxy + v)

def align_yaxis(ax1: matplotlib.axes.Axes, v1: float,
                ax2: matplotlib.axes.Axes, v2: float) -> None:
    """
    Aligns two y-values (v1, v2) from two different axes objects so they appear
    at the same vertical position on the figure.

    This is useful when plotting related data on two axes and ensuring that
    reference points (e.g., baselines) align visually.

    Args:
        ax1 (matplotlib.axes.Axes): The first axes object.
        v1 (float): The y-value in ax1 to align.
        ax2 (matplotlib.axes.Axes): The second axes object.
        v2 (float): The y-value in ax2 to align.

    Returns:
        None
    """
    fig = ax1.get_figure()
    fig.canvas.draw()  # Ensure layout is updated before computing transforms

    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))

    adjust_yaxis(ax2, (y1 - y2) / 2, v2)
    adjust_yaxis(ax1, (y2 - y1) / 2, v1)

def get_avg_and_error(
    signal: np.ndarray,
    fps: int,
    pattern_length_sec: float,
    pattern_start_sec: float,
    pre_stim_sec: float = 0.0,
    normalize_to_baseline: bool = True,
    error_type: str = 'sem'
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    Computes the average response and error (standard deviation or SEM) across repeated stimulus patterns.

    Args:
        signal (np.ndarray): 1D array representing the signal over time [T,].
        fps (int): Frames per second of the signal.
        pattern_length_sec (float): Duration of each stimulus pattern in seconds.
        pattern_start_sec (float): Time (in seconds) when the first stimulus starts.
        pre_stim_sec (float, optional): Duration before stimulus to use as baseline. Defaults to 0.0.
        normalize_to_baseline (bool, optional): Whether to subtract the baseline from each repeat. Defaults to True.
        error_type (str, optional): Type of error to compute; 'sd' for standard deviation, 'sem' for standard error 
            of the mean, or 'none' to skip error calculation. Defaults to 'sem'.

    Returns:
        tuple[np.ndarray, np.ndarray or None]: A tuple containing:
            - avg (np.ndarray): Averaged signal across stimulus repeats [window_frames,].
            - err (np.ndarray or None): Error across repeats [window_frames,] or None if error_type is invalid.
    """
    avg, all_reps = average_stim_repeats(
        signal, fps,
        window_sec=pattern_length_sec,
        start_sec=pattern_start_sec,
        pre_stim_sec=pre_stim_sec,
        return_all=True
    )

    if normalize_to_baseline:
        baseline_frames = int(pre_stim_sec * fps)
        baseline_vals = all_reps[:, :baseline_frames].mean(axis=1, keepdims=True)
        all_reps = all_reps - baseline_vals
        avg = all_reps.mean(axis=0)

    if error_type == 'sd':
        err = all_reps.std(axis=0)
    elif error_type == 'sem':
        err = all_reps.std(axis=0) / np.sqrt(all_reps.shape[0])
    else:
        err = None

    return avg, err

def plot_single_panel_signals(
    signal_ca_corr: np.ndarray,
    signal_ca_wocorr: np.ndarray,
    signal_dhbo: np.ndarray,
    signal_dhbr: np.ndarray,
    signal_dhbt: np.ndarray,
    fps: int,
    pattern_start_sec: float,
    pattern_length_sec: float,
    pre_stim_sec: float = 0.0,
    title: str = '',
    stim_label: str = 'Stimulus',
    x_label: str = 'Time (s)',
    y_ca: str = 'ΔF/F₀ Ca²⁺ (%)',
    y_hb: str = 'Δ[Hb] (μM)',
    normalize_to_baseline: bool = True,
    error_type: str = 'sem',
    label_position: str = 'center',
    save_path: Path = Path("my_signals_plot.png")
) -> None:
    """
    Plots corrected and uncorrected calcium signals along with hemodynamic signals 
    (HbO, HbR, HbT) on a shared time axis in a dual y-axis panel.

    Args:
        signal_ca_corr (np.ndarray): Corrected calcium signal (1D array).
        signal_ca_wocorr (np.ndarray): Uncorrected calcium signal (1D array).
        signal_dhbo (np.ndarray): HbO signal (1D array).
        signal_dhbr (np.ndarray): HbR signal (1D array).
        signal_dhbt (np.ndarray): HbT signal (1D array).
        fps (int): Frames per second of recording.
        pattern_start_sec (float): Time (s) of the first stimulus onset.
        pattern_length_sec (float): Duration (s) of each stimulation window.
        pre_stim_sec (float, optional): Time before stimulus used for baseline. Defaults to 0.0.
        title (str, optional): Plot title. Defaults to ''.
        stim_label (str, optional): Label for stimulus indicator. Defaults to 'Stimulus'.
        x_label (str, optional): X-axis label. Defaults to 'Time (s)'.
        y_ca (str, optional): Y-axis label for calcium signals. Defaults to 'ΔF/F₀ Ca²⁺ (%)'.
        y_hb (str, optional): Y-axis label for hemodynamic signals. Defaults to 'Δ[Hb] (μM)'.
        normalize_to_baseline (bool, optional): Whether to baseline normalize each signal. Defaults to True.
        error_type (str, optional): Type of error to show: 'sem', 'sd', or 'none'. Defaults to 'sem'.
        label_position (str, optional): Position of signal labels: 'left', 'center', or 'right'. Defaults to 'center'.
        save_path (Path, optional): Path to save the figure. If None, saving is skipped.

    Returns:
        None
    """
    # Averaging signals across stimulation repeats
    ca_corr_avg, ca_corr_err = get_avg_and_error(signal_ca_corr, fps, pattern_length_sec, pattern_start_sec, pre_stim_sec, normalize_to_baseline, error_type)
    ca_wocorr_avg, ca_wocorr_err = get_avg_and_error(signal_ca_wocorr, fps, pattern_length_sec, pattern_start_sec, pre_stim_sec, normalize_to_baseline, error_type)
    dhbo_avg, dhbo_err = get_avg_and_error(signal_dhbo, fps, pattern_length_sec, pattern_start_sec, pre_stim_sec, normalize_to_baseline, error_type)
    dhbr_avg, dhbr_err = get_avg_and_error(signal_dhbr, fps, pattern_length_sec, pattern_start_sec, pre_stim_sec, normalize_to_baseline, error_type)
    dhbt_avg, dhbt_err = get_avg_and_error(signal_dhbt, fps, pattern_length_sec, pattern_start_sec, pre_stim_sec, normalize_to_baseline, error_type)

    time = np.arange(len(ca_corr_avg)) / fps
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()

    def plot_with_error(ax, x, y, err, color):
        ax.plot(x, y, color=color)
        if err is not None:
            ax.fill_between(x, y - err, y + err, color=color, alpha=0.1)

    # Define curve parameters: axes, time, signal, error, color, and label
    curves = [
        (ax, time, ca_wocorr_avg, ca_wocorr_err, 'black', 'Ca'),
        (ax, time, ca_corr_avg, ca_corr_err, 'green', 'Ca (corrected)'),
        (ax2, time, dhbt_avg, dhbt_err, 'magenta', 'HbT'),
        (ax2, time, dhbo_avg, dhbo_err, 'red', 'HbO'),
        (ax2, time, dhbr_avg, dhbr_err, 'blue', 'HbR'),
    ]

    for curve_ax, x, y, err, color, _ in curves:
        plot_with_error(curve_ax, x, y, err, color)

    # Individual X-axis offsets for annotation labels (in seconds)
    x_offsets_sec = {
        'Ca': -2,
        'Ca (corrected)': 0.0,
        'HbT': 2,
        'HbO': -4,
        'HbR': 5
    }

    # Annotate signal labels on the plot
    annotation_positions = [
        (ax, time, ca_wocorr_avg, 'Ca', 'black', 0.7),
        (ax, time, ca_corr_avg, 'Ca (corrected)', 'green', 0.8),
        (ax2, time, dhbt_avg, 'HbT', 'magenta', 0.65),
        (ax2, time, dhbo_avg, 'HbO', 'red', 0.45),
        (ax2, time, dhbr_avg, 'HbR', 'blue', 0.25)
    ]

    for ann_ax, x, y, label, color, y_fraction in annotation_positions:
        if label_position == 'left':
            idx = int(len(x) * 0.2)
        elif label_position == 'right':
            idx = int(len(x) * 0.8)
        else:
            idx = int(len(x) * 0.5)

        x_ann = x[idx]
        y_ann = y[idx]
        x_offset = x_offsets_sec.get(label, 0.0)

        ann_ax.annotate(
            label,
            xy=(x_ann, y_ann),
            xytext=(x_ann + x_offset, ann_ax.get_ylim()[0] + y_fraction * (ann_ax.get_ylim()[1] - ann_ax.get_ylim()[0])),
            color=color,
            fontsize=13,
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.3", fc='white', ec=color, lw=1),
            ha='left', va='center'
        )

    for stim_time in [1, 2, 3, 4]:
        ax.axvspan(stim_time, stim_time + 0.2, facecolor='gray', alpha=0.2)

    ax.axhline(y=0.0, color='black', linestyle='--')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_ca, color='green')
    ax2.set_ylabel(y_hb, color='red')
    ax.tick_params(axis='y', colors='green')
    ax2.tick_params(axis='y', colors='red')
    ax2.spines['right'].set_color('red')
    ax2.yaxis.label.set_color('red')
    ax.set_title(title)

    align_yaxis(ax, 0, ax2, 0)
    plt.tight_layout()
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()