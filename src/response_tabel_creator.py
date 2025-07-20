#Import
import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from tqdm import tqdm

from common import load_conv_results
from signal_converter import load_mask
from PIL import Image

def stim_avg_flexible_1d(
    dataset: np.ndarray,
    stim_start_frame: int = 300,
    stim_count: int = 21,
    stim_step: int = 400,
    stim_window: tuple[int, int] = (0, 400),
    subtr_baseline: str = "mean",
    subtr_baseline_interval: int = 20,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Computes averaged stimulus responses from a 1D signal with flexible windowing and baseline subtraction.

    Args:
        dataset (np.ndarray): Input 1D signal array of shape (T,).
        stim_start_frame (int): Index of the first stimulus onset (in frames).
        stim_count (int): Number of stimulus repetitions to process.
        stim_step (int): Number of frames between stimulus onsets.
        stim_window (tuple[int, int]): Relative time window (in frames) to extract per stimulus.
        subtr_baseline (str): Baseline subtraction method. One of:
            - 'mean': subtract global mean of the dataset.
            - 'prestart_mean': subtract mean before each stimulus window.
            - 'no': no baseline subtraction.
        subtr_baseline_interval (int): Number of frames before each stimulus window to use for prestart mean (only used if `subtr_baseline='prestart_mean'`).

    Returns:
        tuple:
            stim_dict (dict[str, np.ndarray]): Dictionary mapping stimulus IDs (e.g. 'st1') to extracted traces.
            avg (np.ndarray): Average trace computed over all stimulus repetitions.
    """
    stim_dict = {}

    for i in range(stim_count):
        stim_start = stim_start_frame + i * stim_step + stim_window[0]
        stim_end = stim_start_frame + i * stim_step + stim_window[1]

        stim = dataset[stim_start:stim_end].copy()

        if subtr_baseline == 'mean':
            stim -= np.mean(dataset)
        elif subtr_baseline == 'prestart_mean':
            baseline_start = stim_start_frame + i * stim_step + stim_window[0] - subtr_baseline_interval
            baseline_end = stim_start_frame + i * stim_step + stim_window[0]
            stim -= np.mean(dataset[baseline_start:baseline_end])

        stim_dict[f'st{i + 1}'] = stim

    avg = sum(stim_dict[f'st{i + 1}'] for i in range(stim_count)) / stim_count
    return stim_dict, avg

def masked_mean(dataframe: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Computes mean signal across masked area in a 3D dataset.

    Parameters
    ----------
    dataframe : np.ndarray
        3D array of shape (frames, height, width).
    mask : np.ndarray
        2D binary mask (height, width).

    Returns
    -------
    np.ndarray
        1D array of masked means per frame.
    """
    mask_3d = np.repeat(mask[:, :, np.newaxis], dataframe.shape[0], axis=2)
    mask_3d = np.moveaxis(mask_3d, -1, 0).astype(bool)

    masked_data = np.ma.masked_array(dataframe, mask=~mask_3d).filled(np.nan)
    return np.nanmean(masked_data, axis=(1, 2))

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

def load_masks_by_format(mask_dir: Path, base_name: str, load_format: str = 'hdf5') -> dict:
    """
    Load masks in the selected format: 'hdf5', 'png', 'tiff', or 'bmp'.

    Args:
        mask_dir (Path): Directory containing the mask files
        base_name (str): Base name of the experiment (e.g., 'mouse1_exp1_protocol1_state1')
        load_format (str): Format to load masks from ('hdf5', 'png', 'tiff', 'bmp')

    Returns:
        dict: Dictionary of masks with keys: 'Ca_L', 'Ca_R', 'HBT_L', 'HBT_R'
    """

    masks = {}

    if load_format == 'hdf5':
        mask_path = mask_dir / f"{base_name}_mask_stim.hdf5"
        if not mask_path.exists():
            raise FileNotFoundError(f"HDF5 mask file not found: {mask_path}")

        with h5py.File(mask_path, "r") as f:
            for name in ['Ca_L', 'Ca_R', 'HBT_L', 'HBT_R']:
                dataset_name = f"Response_mask_{name}"
                if dataset_name not in f:
                    raise KeyError(f"Dataset {dataset_name} not found in {mask_path}")
                masks[name] = f[dataset_name][...]

    elif load_format in {'png', 'tiff', 'bmp'}:
        for name in ['Ca_L', 'Ca_R', 'HBT_L', 'HBT_R']:
            mask_path = mask_dir / f"{base_name}_{name}_mask.{load_format}"
            if not mask_path.exists():
                raise FileNotFoundError(f"Image mask file not found: {mask_path}")
            masks[name] = np.array(Image.open(mask_path)) > 127  # convert to boolean mask

    else:
        raise ValueError(f"Unsupported load_format: {load_format}")

    return masks

def process_experiment(
    img_dir: Path,
    exp: str,
    mouse: str,
    state: str,
    protocol: str,
    res_dir: Path,
    result_directory_stim: Path,
    config: dict,
    stim_config: dict
) -> None:
    """
    Processes a single experiment by loading data, extracting masked signals,
    computing stimulus-locked averages, and saving results to CSV.

    Args:
        img_dir (Path): Path to the experiment folder (unused in this function).
        exp (str): Experiment ID.
        mouse (str): Mouse identifier.
        state (str): Experimental state (e.g., "Awake", "Iso").
        protocol (str): Stimulation protocol string (e.g., "RHL-5Hz").
        res_dir (Path): Directory containing preprocessed results and metadata.
        result_directory_stim (Path): Directory where stimulus results will be saved.
        config (dict): Main configuration dictionary.
        stim_config (dict): Stimulation configuration dictionary.
    """
    base_name = f'{mouse}_{exp}_{protocol}_{state}'
    stim_type = protocol.split('-')[0]

    # Load ROI metadata and filter by Q == 1 (manually approved)
    metadata_file = res_dir / 'Info' / 'info_roi_coord.csv'
    if metadata_file.exists():
        metadata_df = pd.read_csv(metadata_file)
        matching_rows = metadata_df[
            (metadata_df['mouse#'] == mouse) &
            (metadata_df['exp'] == exp) &
            (metadata_df['protocol'] == protocol) &
            (metadata_df['state'] == state)
        ]
        if matching_rows.empty:
            print(f"[SKIP] No matching metadata for: {base_name}")
            return
        if int(matching_rows.iloc[0]['Q']) != 1:
            print(f"[SKIP] Q != 1 for: {base_name}")
            return
    else:
        print(f"[WARNING] Metadata file not found: {metadata_file}")
        return

    # Check if the experiment has already been processed
    result_file = result_directory_stim / 'Ca_stim_info.csv'
    if result_file.exists():
        df_existing = pd.read_csv(result_file)
        is_processed = (
            (df_existing['mouse'] == mouse) &
            (df_existing['exp'] == exp) &
            (df_existing['protocol'] == protocol)
        ).any()
        if is_processed:
            print(f"[SKIP] Already processed: {base_name}")
            return

    print(f"[INFO] Processing: {base_name}")

    # Load converted signal data
    filename = f'{mouse}_{exp}_{protocol}_{state}_conv'
    try:
        ca_corr, dchbo, dchbr, dchbt, ca_wocorr = load_conv_results(
            save_path=res_dir / "Conv_signal",
            filename=filename,
            save_format=config['save_format'],
            return_wocorr=True
        )
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        return
    except Exception as e:
        print(f"[ERROR] Unexpected error while loading data: {e}")
        return

    # Load binary brain mask
    try:
        mask = load_mask(res_dir, mouse, exp, protocol, state, config)
    except Exception as e:
        print(f"[ERROR] Failed to load mask: {e}")
        return

    # Select ROI mask depending on stimulation side
    mask_path = res_dir / f'Mask_stim_{stim_type}'
    masks = load_masks_by_format(mask_path, base_name, load_format=stim_config["load_format"])
    if stim_type.startswith('R'):
        response_mask_chsph = masks.get("Ca_L")
    elif stim_type.startswith('L'):
        response_mask_chsph = masks.get("Ca_R")
    else:
        print(f"[WARNING] Unknown stim type: {stim_type}")
        return

    if response_mask_chsph is None:
        print(f"[ERROR] Mask not found for stim type: {stim_type}")
        return

    stim_cfg = stim_config['stim_1d']

    def process_signal(signal_data: np.ndarray, signal_label: str):
        stim_signal = masked_mean(signal_data, response_mask_chsph)
        stim_dict, _ = stim_avg_flexible_1d(
            dataset=stim_signal,
            stim_start_frame=stim_cfg['stim_start_frame'],
            stim_count=stim_cfg['stim_count'],
            stim_step=stim_cfg['stim_step'],
            stim_window=tuple(stim_cfg['stim_window']),
            subtr_baseline=stim_cfg['subtr_baseline'],
            subtr_baseline_interval=stim_cfg['subtr_baseline_interval']
        )
        save_stim_to_csv(
            stim_dict,
            result_directory_stim,
            mouse, protocol, exp, state,
            stim_type,
            signal=signal_label
        )

    process_signal(ca_corr, "Ca")
    process_signal(ca_wocorr, "Cawocorr")
    process_signal(dchbt, "dcHbT")
    process_signal(dchbo, "dcHbO")
    process_signal(dchbr, "dcHbR")

def log_stim_quality(
    mouse: str,
    exp: str,
    protocol: str,
    state: str,
    field: str,
    prompt: str,
    csv_dir: Path = Path("quality_scores"),
    csv_name: str = "stim_quality_scores.csv"
):
    """
    Logs or updates a stimulus quality score in a CSV file.

    Parameters
    ----------
    mouse : str
        Mouse identifier.
    exp : str
        Experiment name or ID.
    protocol : str
        Protocol name.
    state : str
        Experimental state (e.g., 'Awake').
    field : str
        Column name to update (e.g., 'avg_3d_q').
    prompt : str
        Message shown to the user for input.
    csv_dir : Path, optional
        Directory to store the CSV file. Defaults to "quality_scores".
    csv_name : str, optional
        Name of the CSV file. Defaults to "stim_quality_scores.csv".
    """
    csv_dir.mkdir(exist_ok=True)
    csv_path = csv_dir / csv_name

    fields = [
        "mouse", "exp", "protocol", "state",
        "stim_stack_q_Ca", "stim_stack_q_HbT",
        "avg_3d_q_Ca", "avg_3d_q_HbT",
        "avg_2d_q_Ca", "avg_2d_q_HbT",
        "response_plot_q", "hemo_corr_q"
    ]

    score = input(f"{prompt}").strip()
    if score and score not in {'0', '1', '2'}:
        print("Please enter a number between 0 and 2, or leave empty.")
        return

    # Read current contents of the CSV
    rows = []
    found = False
    if csv_path.exists():
        with open(csv_path, mode='r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (
                    row["mouse"] == mouse and
                    row["exp"] == exp and
                    row["protocol"] == protocol and
                    row["state"] == state
                ):
                    row[field] = score
                    found = True
                rows.append(row)

    # Add a new row if this experiment is not yet present
    if not found:
        new_row = {f: "" for f in fields}
        new_row.update({
            "mouse": mouse,
            "exp": exp,
            "protocol": protocol,
            "state": state,
            field: score
        })
        rows.append(new_row)

    # Write updated data back to the CSV
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Field '{field}' saved with value: {score or '—'}")