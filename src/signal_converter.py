# Import
import os
import shutil
from pathlib import Path
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import pandas as pd
import imutils
import h5py
import imageio
from PIL import Image
import traceback
import joblib
from joblib import Parallel, delayed
import gc
import zarr
from scipy.ndimage import uniform_filter1d
from scipy import ndimage
from tqdm import tqdm
from typing import Union, Optional, Sequence, Tuple, Callable, Iterable, Any, List, Dict

from common import sortByNumberofFrame
from transformation import (
    move_vertical,
    move_horizontal,
    resize
)

def load_and_process_image(
    path: Union[str, Path],
    scalingfactor: Union[int, float],
    angle: Union[int, float]
) -> np.ndarray:
    """
    Load an image from disk, resize it by a scaling factor, and rotate it by a given angle.

    Args:
        path (str or Path): Path to the input image file (grayscale expected).
        scalingfactor (int or float): Scaling factor as a percentage of the original size (e.g., 50 = 50%).
        angle (int or float): Angle in degrees to rotate the image. Positive values rotate counterclockwise.

    Returns:
        np.ndarray: Processed grayscale image, resized and rotated.

    Raises:
        FileNotFoundError: If the image path is invalid.
        ValueError: If scale is non-positive or image is not valid.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    if scalingfactor <= 0:
        raise ValueError("Scaling factor must be positive.")

    # Call your custom resize function (should return a valid grayscale image)
    img = resize(path, scalingfactor)
    if img is None:
        raise ValueError(f"Failed to load or resize image from: {path}")

    return imutils.rotate(img, angle)

def sort_image_paths(
    path_list: list[Path],
    ex_ptrn: Optional[Tuple[int, int]] = None,
    r1_ptrn: Optional[Tuple[int, int]] = None,
    r2_ptrn: Optional[Tuple[int, int]] = None
) -> Tuple[Optional[list[Path]], Optional[list[Path]], Optional[list[Path]]]:
    """
    Sort a list of image paths by frame number and optionally extract patterns from it.

    Args:
        path_list (list[Path]): List of image file paths.
        ex_ptrn (tuple[int, int], optional): Pattern (start, step) for extracting excitation images.
        r1_ptrn (tuple[int, int], optional): Pattern (start, step) for extracting reflectance 1 images.
        r2_ptrn (tuple[int, int], optional): Pattern (start, step) for extracting reflectance 2 images.

    Returns:
        tuple: Three lists (or None) of paths:
            - ex_list: Subset of paths matching `ex_ptrn`, or None if not given.
            - r1_list: Subset matching `r1_ptrn`, or None.
            - r2_list: Subset matching `r2_ptrn`, or None.

    Notes:
        - Sorting is done in-place based on `sortByNumberofFrame`, which should extract a numeric frame index from filename.
        - Each pattern is applied using list slicing: [start::step].
    """
    path_list.sort(key=sortByNumberofFrame)
    
    ex_list = path_list[ex_ptrn[0]::ex_ptrn[1]] if ex_ptrn else None
    r1_list = path_list[r1_ptrn[0]::r1_ptrn[1]] if r1_ptrn else None
    r2_list = path_list[r2_ptrn[0]::r2_ptrn[1]] if r2_ptrn else None

    return ex_list, r1_list, r2_list

def get_mouse_info(
    info_moving: pd.DataFrame,
    mouse: str,
    exp: str,
    protocol: str,
    state: str
) -> pd.DataFrame:
    """
    Extract a subset of the info DataFrame corresponding to a specific mouse, experiment, protocol, and state.

    Args:
        info_moving (pd.DataFrame): DataFrame containing metadata about all processed experiments.
        mouse (str): Mouse identifier (column: 'mouse#').
        exp (str): Experiment day identifier (column: 'exp').
        protocol (str): Protocol name (column: 'protocol').
        state (str): Experimental state (column: 'state').

    Returns:
        pd.DataFrame: Filtered DataFrame with rows matching all provided conditions.
            May be empty if no match is found.
    """
    mouse_info = info_moving[
        (info_moving['mouse#'] == mouse) &
        (info_moving['exp'] == exp) &
        (info_moving['protocol'] == protocol) &
        (info_moving['state'] == state)
    ]
    return mouse_info

def apply_shift(images: np.ndarray, move_x: int, move_y: int) -> np.ndarray:
    """
    Apply horizontal and vertical shift to an image or a stack of images.

    Args:
        images (np.ndarray): A 2D (H, W) or 3D (D, H, W) array representing a single image or multiple frames.
        move_x (int): Number of pixels to shift along the X-axis (horizontal).
            - Positive: shift left
            - Negative: shift right
        move_y (int): Number of pixels to shift along the Y-axis (vertical).
            - Positive: shift up
            - Negative: shift down

    Returns:
        np.ndarray: Shifted image or image stack, same shape as input.

    Notes:
        - Areas revealed by the shift are zero-filled.
        - Uses `move_horizontal()` and `move_vertical()` functions in that order.
    """
    return move_vertical(move_horizontal(images, move_x), move_y)

def save_hdf5_data(save_path: Union[str, Path], filename: str, data_dict: dict[str, np.ndarray]) -> None:
    """
    Save multiple datasets into a single HDF5 file.

    Args:
        save_path (str or Path): Directory where the HDF5 file will be saved.
        filename (str): Name of the output HDF5 file (e.g., 'data.hdf5').
        data_dict (dict[str, np.ndarray]): Dictionary of named datasets to store. 
            Keys are dataset names, values are NumPy arrays.

    Returns:
        None

    Raises:
        OSError: If the file cannot be created or written.
        ValueError: If the data_dict contains unsupported types.

    Notes:
        - All datasets are stored using dtype='i8' (64-bit signed integer).
        - The directory is created if it does not exist.
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    file_path = save_path / filename
    with h5py.File(str(file_path), 'w') as f:
        for name, data in data_dict.items():
            f.create_dataset(name, data=data, dtype='i8')

def parallel_with_progress(
    func: Callable[[Any], Any],
    iterable: Iterable[Any],
    n_jobs: int = -1,
    desc: str = ""
) -> list[Any]:
    """
    Run a function in parallel over an iterable with a progress bar.

    Args:
        func (Callable): Function to apply to each item in the iterable.
        iterable (Iterable): A collection of items to process.
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to -1 (all CPUs).
        desc (str, optional): Description to display in the progress bar. Defaults to "".

    Returns:
        list: List of results from applying `func` to each element in `iterable`.

    Notes:
        - Uses `joblib.Parallel` for parallelism and `tqdm` for progress tracking.
        - Ensure that `func` is picklable (can be serialized) if using multiprocessing.
    """
    return Parallel(n_jobs=n_jobs, backend='threading')(
        delayed(func)(item) for item in tqdm(iterable, desc=desc)
    )

def load_processed_log(log_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load the processed log CSV file, ensuring required columns are present.

    Args:
        log_path (str or Path): Path to the CSV file logging processed experiments.

    Returns:
        pd.DataFrame: Log DataFrame with the following columns:
            ['mouse', 'exp', 'protocol', 'state', 'filename', 'move_status', 'conv_status'].
            If the file does not exist or lacks columns, they will be created with empty values.

    Notes:
        - If the file does not exist, an empty DataFrame is returned with required columns.
        - Missing columns in an existing file are added with empty strings.
    """
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        print(f"Log file not found at {log_path}. Creating a new one.")
        df = pd.DataFrame()

    required_cols = ["mouse", "exp", "protocol", "state", "filename", "move_status", "conv_status"]
    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[required_cols]
    return df


def already_processed(
    log_df: pd.DataFrame,
    mouse: str,
    exp: str,
    protocol: str,
    state: str
) -> bool:
    """
    Check if a given experiment has already been logged as processed.

    Args:
        log_df (pd.DataFrame): DataFrame containing processed experiment logs.
        mouse (str): Mouse identifier.
        exp (str): Experiment day identifier.
        protocol (str): Protocol name.
        state (str): Experimental state.

    Returns:
        bool: True if a matching log entry exists, False otherwise.

    Notes:
        - A match is considered valid if all four parameters match a row in the log.
        - This function assumes that the log contains the columns: 'mouse', 'exp', 'protocol', 'state'.
    """
    return not log_df[
        (log_df["mouse"] == mouse) &
        (log_df["exp"] == exp) &
        (log_df["protocol"] == protocol) &
        (log_df["state"] == state)
    ].empty
    
def log_result(
    log_path: Union[str, Path],
    log_df: pd.DataFrame,
    mouse: str,
    exp: str,
    protocol: str,
    state: str,
    filename: str,
    move_status: Optional[str] = None,
    conv_status: Optional[str] = None
) -> pd.DataFrame:
    """
    Update the processing log for a given experiment.

    Adds a new row if the experiment is not yet logged, or updates the `move_status`
    and/or `conv_status` fields if it already exists.

    Args:
        log_path (str or Path): Path to the CSV log file.
        log_df (pd.DataFrame): Current log DataFrame (typically from `load_processed_log`).
        mouse (str): Mouse identifier.
        exp (str): Experiment day identifier.
        protocol (str): Protocol name.
        state (str): Experimental state.
        filename (str): Output file name for the processed result.
        move_status (str, optional): Result or status from the movement correction stage.
        conv_status (str, optional): Result or status from the signal conversion stage.

    Returns:
        pd.DataFrame: Updated log DataFrame with the new or modified entry.

    Notes:
        - The DataFrame is saved back to CSV immediately after the update.
        - If the same mouse/exp/protocol/state combination already exists, only the
          `move_status` and/or `conv_status` fields will be updated (not duplicated).
    """
    condition = (
        (log_df['mouse'] == mouse) &
        (log_df['exp'] == exp) &
        (log_df['protocol'] == protocol) &
        (log_df['state'] == state)
    )

    if not condition.any():
        new_row = {
            'mouse': mouse,
            'exp': exp,
            'protocol': protocol,
            'state': state,
            'filename': filename,
            'move_status': move_status if move_status else '',
            'conv_status': conv_status if conv_status else ''
        }
        log_df = pd.concat([log_df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        if move_status is not None:
            log_df.loc[condition, 'move_status'] = move_status
        if conv_status is not None:
            log_df.loc[condition, 'conv_status'] = conv_status

    log_df.to_csv(log_path, index=False)
    return log_df

def nparrayFromimages_2waves_IOS_GCAMP(
    img_dir: Union[str, Path],
    res_dir: Union[str, Path],
    demonstr_dir: Union[str, Path],
    info_dir: Union[str, Path],
    info_moving_dir: Union[str, Path],
    protocol: str,
    state: str,
    exp: str,
    mouse: str,
    img_number: list[int],
    rotation: float,
    scalingfactor: int = 25,
    br_y: float = 0.6,
    Save_format: str = 'hdf5',
    ex_ptrn: Optional[list[int]] = None,
    r1_ptrn: Optional[list[int]] = None,
    r2_ptrn: Optional[list[int]] = None
) -> tuple:
    """
    Load, align, scale, and save WFOI experimental images for fluorescence and reflectance channels.

    Args:
        img_dir (str or Path): Directory with raw image frames.
        res_dir (str or Path): Output directory for processed data.
        demonstr_dir (str or Path): Directory to save preview images (TIFF).
        info_dir (str or Path): Reserved for future use.
        info_moving_dir (str or Path): CSV file with alignment/movement parameters.
        protocol (str): Protocol name (e.g., stimulus type).
        state (str): Animal state label (e.g., 'Awake').
        exp (str): Experiment ID.
        mouse (str): Mouse identifier.
        img_number (list of int): Expected valid number of image frames.
        rotation (float): Reserved for future rotation handling.
        scalingfactor (int): Resize percentage (default 25).
        br_y (float): Expected vertical bregma position (0–1).
        Save_format (str): Either 'hdf5' or 'zarr'.
        ex_ptrn (list[int] or None): [start, step] for fluorescence frames.
        r1_ptrn (list[int] or None): Reflectance1 frame pattern.
        r2_ptrn (list[int] or None): Reflectance2 frame pattern.

    Returns:
        tuple: (
            moved_fluo_images (np.ndarray or 'NaN'),
            moved_refl1_images (np.ndarray or 'NaN'),
            moved_refl2_images (np.ndarray or 'NaN'),
            bregma (int or 'NaN'),
            img_size (tuple[int, int] or 'NaN'),
            Err (str)
        )

    Notes:
        - If alignment or image count fails, returns 'NaN' and an error reason.
        - Uses external movement info for alignment from `info_moving_dir`.
        - Intermediate and final images are saved to disk.
    """

    Err = 'Ok'
    img_directory = Path(img_dir)
    demonstr_dir = Path(demonstr_dir)
    res_dir = Path(res_dir)

    try:
        # === Load all filenames from image directory === 
        path_for_frames = [f for f in os.listdir(img_directory) if f != 'Thumbs.db']
        if len(path_for_frames) not in img_number:
            return 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'img_number'

        # === Load movement and geometry info === 
        info_moving = pd.read_csv(info_moving_dir)
        info_moving['date'] = info_moving['date'].astype(str)
        mouse_info = get_mouse_info(info_moving, mouse, exp, protocol, state)
        if mouse_info.empty:
            return 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'info_moving'

        # === Check for wave inversion (e.g., imaging error) === 
        invert = mouse_info['invert'].values[0]
        if invert == '1':
            print('Invert waves!!!')
            return 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', 'invert'

        # === Sort image paths for each channel using frame pattern === 
        full_paths = [img_directory / p for p in path_for_frames]
        fluo_paths, refl1_paths, refl2_paths = sort_image_paths(full_paths, ex_ptrn, r1_ptrn, r2_ptrn)

        # === Extract average alignment geometry === 
        angle = mouse_info.angle.mean()
        U_x, U_y, D_x, D_y, angle, Br_x, Br_y = (mouse_info.loc[:, "U_x":"Br_y"].mean().values * scalingfactor / 100)

        moved_data = {}
        img_size = None

        # === Process Fluorescence Images (if available) ===
        if fluo_paths:
            fluo_imgs = np.asarray(
                parallel_with_progress(
                    lambda p: load_and_process_image(p, scalingfactor, angle),
                    fluo_paths, desc=f"fluo images [{mouse}]"
                )
            )
            img_size = fluo_imgs[4].shape
            print(img_size)
            moved_data['moved_fluo_images'] = apply_shift(
                fluo_imgs,
                math.ceil(Br_x) - math.ceil(fluo_imgs.shape[2] / 2),
                math.ceil(Br_y) - math.ceil(br_y * fluo_imgs.shape[1])
            )

        # === Process Reflectance 1 Images ===
        if refl1_paths:
            refl1_imgs = np.asarray(
                parallel_with_progress(
                    lambda p: load_and_process_image(p, scalingfactor, angle),
                    refl1_paths, desc=f"refl1 images [{mouse}]"
                )
            )
            if img_size is None:
                img_size = refl1_imgs[4].shape
            moved_data['moved_refl1_images'] = apply_shift(
                refl1_imgs,
                math.ceil(Br_x) - math.ceil(refl1_imgs.shape[2] / 2),
                math.ceil(Br_y) - math.ceil(br_y * refl1_imgs.shape[1])
            )
            

        # === Process Reflectance 2 Images ===
        if refl2_paths:
            refl2_imgs = np.asarray(
                parallel_with_progress(
                    lambda p: load_and_process_image(p, scalingfactor, angle),
                    refl2_paths, desc=f"refl2 images [{mouse}]"
                )
            )
            if img_size is None:
                img_size = refl2_imgs[4].shape
            moved_data['moved_refl2_images'] = apply_shift(
                refl2_imgs,
                math.ceil(Br_x) - math.ceil(refl2_imgs.shape[2] / 2),
                math.ceil(Br_y) - math.ceil(br_y * refl2_imgs.shape[1])
            )

        # === Final bregma position (middle of image width) === 
        bregma = math.ceil(img_size[1] / 2)

        # === Save to file ===
        save_path = res_dir / 'Moved_data'
        save_path.mkdir(parents=True, exist_ok=True)
        filename = f'{mouse}_{exp}_{protocol}_{state}'

        if Save_format == 'hdf5':
            save_hdf5_data(save_path, f'{filename}.hdf5', moved_data)
        elif Save_format == 'zarr':
            zarr_save_path = save_path / f'{filename}.zarr'
            root = zarr.open(str(zarr_save_path), mode='w')
            for key, arr in moved_data.items():
                root.create_dataset(key, data=arr, shape=arr.shape, dtype='i8')
        else:
            print(f"Unsupported format: {Save_format}")
            Err = 'unsupported_format'

        gc.collect()

        # === Return only available image arrays; fallback is 'NaN' === 
        return (
            moved_data.get('moved_fluo_images', 'NaN'),
            moved_data.get('moved_refl1_images', 'NaN'),
            moved_data.get('moved_refl2_images', 'NaN'),
            bregma, img_size, Err
        )

    except Exception as e:
        print("Unexpected error occurred:")
        traceback.print_exc()
        Err = f'Exception: {str(e)}'
        return 'NaN', 'NaN', 'NaN', 'NaN', 'NaN', Err

# Сomputation F470(t)/F470(t0)
def compute_fluo_f_over_f0(fluo_images: np.ndarray) -> np.ndarray:
    """
    Compute ΔF/F₀ from a fluorescence image sequence.

    Args:
        fluo_images (np.ndarray): 3D array of fluorescence images with shape (T, H, W),
            where T is the number of timepoints.

    Returns:
        np.ndarray: 3D array of ΔF/F₀ values (same shape as input), dtype float32.

    Notes:
        - F₀ is computed as the mean image across time.
        - To avoid division by zero, an epsilon is added to F₀.
        - All-zero values after normalization are replaced with 1.0.
    """
    F0 = np.mean(fluo_images, axis=0)
    F0_safe = F0 + np.finfo(np.float32).eps  # Avoid division by zero
    f_f0 = fluo_images / F0_safe
    f_f0 = f_f0.astype(np.float32, copy=False)
    f_f0 = np.where(f_f0 == 0, 1.0, f_f0)  # Replace zeroed-out values with 1.0
    return f_f0

# Сomputation I530(t)/I530(t0)
def compute_refl_i_over_i0(refl_images: np.ndarray, rate_noneq: bool = False) -> np.ndarray:
    """
    Normalize reflectance image sequence by baseline (I / I₀).

    Args:
        refl_images (np.ndarray): 3D array of reflectance images with shape (T, H, W).
        rate_noneq (bool): If True, the sequence will be temporally duplicated
            (frames repeated along time axis) to match unequal frame rates (e.g., 2:1 ratio).

    Returns:
        np.ndarray: I/I₀-normalized image array (float32), same or doubled length if `rate_noneq=True`.

    Notes:
        - I₀ is computed as the average across time (axis=0).
        - Epsilon is added to avoid division by zero.
        - Zero values after normalization are replaced with 1.0.
        - Frame repetition (np.repeat) doubles time axis if needed.
    """
    I0 = np.mean(refl_images, axis=0)
    I0_safe = I0 + np.finfo(np.float32).eps
    I_I0 = refl_images / I0_safe
    I_I0 = I_I0.astype(np.float32, copy=False)

    if rate_noneq:
        I_I0 = np.repeat(I_I0, 2, axis=0)

    I_I0 = np.where(I_I0 == 0, 1.0, I_I0)
    return I_I0

# Conversion to hemoglobin concentration

def estimate_pathlength(wave: int, est_path: pd.DataFrame) -> float:
    """
    Estimate the optical pathlength for a given wavelength using direct lookup or interpolation.

    Args:
        wave (int): Target wavelength (in nm).
        est_path (pd.DataFrame): Table containing at least two columns:
            - 'Wavelength': Wavelength values (int)
            - 'X': Corresponding pathlength estimates (float)

    Returns:
        float: Estimated pathlength for the specified wavelength.

    Notes:
        - If the wavelength is even, a direct match is used.
        - If the wavelength is odd, it is interpolated as the average of (wave - 1) and (wave + 1).
        - No extrapolation is performed; both neighboring wavelengths must exist in `est_path`.

    Raises:
        IndexError: If required wavelengths are not present in the `est_path` DataFrame.
    """
    if wave % 2 == 0:
        return est_path.loc[est_path['Wavelength'] == wave, 'X'].values[0]
    else:
        lower = est_path.loc[est_path['Wavelength'] == wave - 1, 'X'].values[0]
        upper = est_path.loc[est_path['Wavelength'] == wave + 1, 'X'].values[0]
        return (lower + upper) / 2

    
#Constant values from 
#https://royalsocietypublishing.org/action/downloadSupplement?doi=10.1098%2Frstb.2015.0360&file=rstb20150360supp1.pdf

def convert_to_dchb(
    refl1: np.ndarray,
    refl2: np.ndarray,
    lambda_1: int,
    lambda_2: int,
    coeff_abs_path: Union[str, Path],
    est_path_path: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert I/I₀ normalized reflectance signals into hemoglobin concentration changes.

    Args:
        refl1 (np.ndarray): Normalized reflectance signal at wavelength `lambda_1` (I/I₀).
        refl2 (np.ndarray): Normalized reflectance signal at wavelength `lambda_2` (I/I₀).
        lambda_1 (int): Wavelength corresponding to `refl1` (in nm).
        lambda_2 (int): Wavelength corresponding to `refl2` (in nm).
        coeff_abs_path (str or Path): Path to CSV file with extinction coefficients
            (must contain columns: 'Wave', 'HbO', 'HbR').
        est_path_path (str or Path): Path to CSV file with estimated pathlengths
            (must contain columns: 'Wavelength', 'X').

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of hemoglobin concentration changes:
            - dcHBO (oxygenated hemoglobin)
            - dcHBR (deoxygenated hemoglobin)
            - dcHBT (total hemoglobin)

    Raises:
        ValueError: If required wavelengths are missing from extinction or pathlength data.

    Notes:
        - Based on modified Beer–Lambert law.
        - Optical pathlengths are scaled by 0.1 to convert from mm to cm.
        - A small epsilon is added to avoid log(0).
    """
    coeff_abs = pd.read_csv(coeff_abs_path)
    est_path = pd.read_csv(est_path_path)

    # === Extinction coefficients ===
    try:
        epsHbO1 = coeff_abs.loc[coeff_abs['Wave'] == lambda_1, 'HbO'].values[0]
        epsHbR1 = coeff_abs.loc[coeff_abs['Wave'] == lambda_1, 'HbR'].values[0]
        epsHbO2 = coeff_abs.loc[coeff_abs['Wave'] == lambda_2, 'HbO'].values[0]
        epsHbR2 = coeff_abs.loc[coeff_abs['Wave'] == lambda_2, 'HbR'].values[0]
    except IndexError:
        raise ValueError(f"Missing extinction coefficient data for wavelengths {lambda_1} or {lambda_2}.")

    # === Optical pathlengths ===
    X1 = estimate_pathlength(lambda_1, est_path) * 0.1  # Convert mm to cm
    X2 = estimate_pathlength(lambda_2, est_path) * 0.1

    # === Optical density calculation ===
    epsilon = np.finfo(np.float32).eps
    dabs1 = -(1 / X1) * np.log(refl1 + epsilon)
    dabs2 = -(1 / X2) * np.log(refl2 + epsilon)

    # === Hemoglobin concentration change ===
    denomR = epsHbO1 * epsHbR2 - epsHbO2 * epsHbR1
    denomO = epsHbR1 * epsHbO2 - epsHbR2 * epsHbO1

    dcHBR = (epsHbO1 * dabs2 - epsHbO2 * dabs1) / denomR
    dcHBO = (epsHbR1 * dabs2 - epsHbR2 * dabs1) / denomO
    dcHBT = dcHBO + dcHBR

    return dcHBO, dcHBR, dcHBT

# Correction options

def apply_blb_correction(
    dataset: np.ndarray,
    dcHBO: np.ndarray,
    dcHBR: np.ndarray,
    wave_ex: int,
    wave_em: int,
    coeff_abs_path: Union[str, Path],
    Xest_ex: float,
    Xest_em: float
) -> np.ndarray:
    """
    Apply Beer–Lambert Law-based correction to a ΔF/F fluorescence signal
    to compensate for absorption by hemoglobin.

    Args:
        dataset (np.ndarray): ΔF/F data (time x height x width).
        dcHBO (np.ndarray): Oxygenated hemoglobin time series (same shape as dataset).
        dcHBR (np.ndarray): Deoxygenated hemoglobin time series (same shape as dataset).
        wave_ex (int): Excitation wavelength (in nm).
        wave_em (int): Emission wavelength (in nm).
        coeff_abs_path (str or Path): Path to extinction coefficients file
            (must contain columns: 'Wave', 'HbO', 'HbR').
        Xest_ex (float): Estimated optical pathlength for excitation (in cm).
        Xest_em (float): Estimated optical pathlength for emission (in cm).

    Returns:
        np.ndarray: Corrected ΔF/F signal (same shape as input).

    Raises:
        ValueError: If extinction coefficients for the given wavelengths are missing.
    """
    coeff_abs = pd.read_csv(coeff_abs_path, sep=' ')
    waves = coeff_abs['Wave'].values

    if wave_ex not in waves or wave_em not in waves:
        raise ValueError(f"Missing extinction data for wave_ex={wave_ex} or wave_em={wave_em}.")

    # Extinction coefficients
    epsHbO_ex = coeff_abs.loc[coeff_abs['Wave'] == wave_ex, 'HbO'].values[0]
    epsHbR_ex = coeff_abs.loc[coeff_abs['Wave'] == wave_ex, 'HbR'].values[0]
    epsHbO_em = coeff_abs.loc[coeff_abs['Wave'] == wave_em, 'HbO'].values[0]
    epsHbR_em = coeff_abs.loc[coeff_abs['Wave'] == wave_em, 'HbR'].values[0]

    # Differential absorption at excitation and emission wavelengths
    dabs_ex = epsHbO_ex * dcHBO + epsHbR_ex * dcHBR
    dabs_em = epsHbO_em * dcHBO + epsHbR_em * dcHBR

    # Apply exponential correction
    corrected = dataset * np.exp(dabs_ex * Xest_ex + dabs_em * Xest_em)

    return corrected

def apply_lstm_correction_2d(
    f_f0: np.ndarray,
    hemo_signal: np.ndarray
) -> np.ndarray:
    """
    Apply voxel-wise linear regression correction to ΔF/F signal
    using a hemodynamic trace (e.g., dcHBO). Inspired by LSTM decontamination logic.

    Args:
        f_f0 (np.ndarray): ΔF/F fluorescence data of shape (T, H, W).
        hemo_signal (np.ndarray): Hemodynamic signal (e.g., dcHBO), same shape as f_f0.

    Returns:
        np.ndarray: Corrected fluorescence data with shape (T, H, W).

    Notes:
        - For each pixel, fits a linear model: y = β₀ + β₁·x
        - Subtracts the predicted hemodynamic component ŷ from the original y.
        - Pixels with invalid values (NaNs, infs) are skipped silently.
    """
    data_corr = np.zeros_like(f_f0)

    for r in range(f_f0.shape[1]):
        for c in range(f_f0.shape[2]):
            try:
                y = f_f0[:, r, c]
                x = hemo_signal[:, r, c]

                if np.any(np.isnan(x)) or np.any(np.isnan(y)):
                    continue

                X = np.column_stack((np.ones_like(x), x))  # Add intercept
                b = np.linalg.lstsq(X, y, rcond=None)[0]
                y_hat = X @ b
                data_corr[:, r, c] = y - y_hat

            except Exception:
                # Silently skip bad fits
                continue

    return data_corr

def show_figures(
    df_list: List[np.ndarray],
    df_list_names: List[str],
    im_n: int,
    mask: np.ndarray
) -> None:
    """
    Display multiple masked 2D images side-by-side with consistent colormap scaling.

    Args:
        df_list (List[np.ndarray]): List of 3D arrays (e.g., time × height × width) containing image data.
        df_list_names (List[str]): List of names for each image to use as subplot titles.
        im_n (int): Index of the frame to display from each dataset.
        mask (np.ndarray): 2D boolean array to apply as a mask (True = include pixel, False = mask out).

    Returns:
        None. The function displays the figures using matplotlib.

    Notes:
        - Masked-out pixels are set to NaN and excluded from percentile-based color scaling.
        - Uses jet colormap with 2nd–98th percentile scaling for better contrast.
    """
    subplot_n = len(df_list)
    plt.figure(figsize=(4 * subplot_n, 4))

    for i, df in enumerate(df_list):
        # Extract image frame and apply mask
        mdata = np.ma.masked_array(df[im_n], mask=~mask.astype(bool))
        mdata = np.ma.filled(mdata.astype(np.float32), np.nan)

        plt.subplot(1, subplot_n, i + 1)
        plt.imshow(
            mdata,
            cmap='jet',
            vmin=np.nanpercentile(mdata, 2),
            vmax=np.nanpercentile(mdata, 98)
        )
        plt.title(df_list_names[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def remove_global_signal(
    fluo_df_f: np.ndarray,
    mask: np.ndarray,
    bregma: int,
    smooth_win: int = 5,
    plot: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove the global fluorescence signal from ΔF/F data using a vessel-excluded mask.

    Args:
        fluo_df_f (np.ndarray): 3D fluorescence data (T, H, W), ΔF/F signal.
        mask (np.ndarray): 2D binary mask of the brain region (H, W).
        bregma (int): X-coordinate of the bregma point (used to exclude vessel region).
        smooth_win (int, optional): Window size for smoothing the global signal. Default is 5.
        plot (bool, optional): If True, plots the raw and smoothed global signal.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - corrected (np.ndarray): ΔF/F data after global signal removal.
            - global_signal_smooth (np.ndarray): Smoothed global signal used for correction.

    Notes:
        - Excludes central vertical stripe (bregma ± 5px) to avoid midline vessel influence.
        - Global signal is computed as the mean over the remaining masked pixels per frame.
    """
    # 1. Exclude vessel strip near bregma from the mask
    mask_vessel_excluded = mask.copy()
    x_min = max(0, bregma - 5)
    x_max = min(mask.shape[1], bregma + 6)
    mask_vessel_excluded[:, x_min:x_max] = 0

    # 2. Convert to boolean mask
    mask_bool = mask_vessel_excluded.astype(bool)

    # 3. Compute global signal over non-vessel pixels
    global_signal = np.mean(fluo_df_f[:, mask_bool], axis=1)

    # 4. Smooth the signal
    global_signal_smooth = uniform_filter1d(global_signal, size=smooth_win)

    # 5. Optional plot
    if plot:
        plt.figure(figsize=(8, 4))
        plt.plot(global_signal, label='Raw Global Signal', alpha=0.4)
        plt.plot(global_signal_smooth, label='Smoothed', linewidth=2)
        plt.title('Global Signal (vessel-excluded mask)')
        plt.xlabel('Frame')
        plt.ylabel('Signal')
        plt.legend()
        plt.tight_layout()
        plt.show()

    # 6. Subtract the global signal from each frame
    corrected = fluo_df_f - global_signal_smooth[:, np.newaxis, np.newaxis]
    return corrected, global_signal_smooth

def imgs_convertion_2waves_IOS_GCaMP(
    fluo_images: np.ndarray,
    refl1_images: np.ndarray,
    refl2_images: Union[np.ndarray, str],
    mask: np.ndarray,
    res_dir: Union[str, Path],
    info_dir: Union[str, Path],
    protocol: str,
    state: str,
    exp: str,
    mouse: str,
    lambdaF_ex: int,
    lambdaF_em: int,
    lambdaR_1: int,
    lambdaR_2: int,
    Ca_corr_method: str,
    rate_noneq: bool,
    Xest_ex: float,
    Xest_em: float,
    dir_coeff_abs: Union[str, Path],
    dir_est_path: Union[str, Path],
    rate: int = 20,
    bregma: int = 44,
    Save_conv: bool = True,
    Save_format: str = "zarr",
    plot_control: bool = True,
    im_n: int = 10,
    Remove_global: bool = False,
    global_smooth_win: int = 5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full pipeline for processing IOS + GCaMP imaging data:
    performs ΔF/F computation, hemoglobin correction (LSTM or BLB),
    global signal removal, Gaussian smoothing, and saves results.

    Parameters
    ----------
    fluo_images : np.ndarray
        3D fluorescence image array (T, H, W).
    refl1_images : np.ndarray
        3D reflectance images at wavelength 1.
    refl2_images : np.ndarray or str
        3D reflectance images at wavelength 2, or 'NaN' if not available.
    mask : np.ndarray
        Binary mask of valid pixels.
    res_dir : str or Path
        Directory to save converted data.
    info_dir : str or Path
        Metadata or logging directory (may be unused).
    protocol : str
        Protocol label.
    state : str
        State label (e.g., Awake/Anesthetized).
    exp : str
        Experiment identifier.
    mouse : str
        Mouse ID.
    lambdaF_ex : int
        Excitation wavelength for GCaMP.
    lambdaF_em : int
        Emission wavelength for GCaMP.
    lambdaR_1 : int
        Wavelength of reflectance channel 1.
    lambdaR_2 : int
        Wavelength of reflectance channel 2.
    Ca_corr_method : str
        Method for hemoglobin correction ('LSTM' or 'BLB').
    rate_noneq : bool
        Whether frame rates of channels differ.
    Xest_ex : float
        Estimated pathlength for excitation.
    Xest_em : float
        Estimated pathlength for emission.
    dir_coeff_abs : str or Path
        Path to extinction coefficient CSV file.
    dir_est_path : str or Path
        Path to estimated pathlength CSV file.
    rate : int, optional
        Frame rate of recording, by default 20.
    bregma : int, optional
        X-position of bregma for global signal removal, by default 44.
    Save_conv : bool, optional
        Whether to save converted results, by default True.
    Save_format : str, optional
        Format to save ('hdf5' or 'zarr'), by default 'zarr'.
    plot_control : bool, optional
        If True, plots intermediate figures, by default True.
    im_n : int, optional
        Frame index for visualization, by default 10.
    Remove_global : bool, optional
        Whether to remove global signal, by default False.
    global_smooth_win : int, optional
        Window size for smoothing global signal, by default 5.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - Ca_corr_gauss : np.ndarray
            Hemoglobin-corrected, smoothed ΔF/F data.
        - dcHBO_gauss : np.ndarray
            Smoothed oxygenated hemoglobin signal.
        - dcHBR_gauss : np.ndarray
            Smoothed deoxygenated hemoglobin signal.
        - dcHBT_gauss : np.ndarray
            Smoothed total hemoglobin signal.
        - moved_img : np.ndarray
            Example preprocessed image (first frame).
    """

    moved_img = fluo_images[0]

    # -------------------- ΔF/F --------------------
    f_f0 = compute_fluo_f_over_f0(fluo_images)

    if plot_control:
        show_figures([fluo_images, f_f0], ['Fluo raw', 'Fluo ΔF/F'], im_n, mask)

    # -------------------- Reflectance Normalization --------------------
    I_I0_refl1 = compute_refl_i_over_i0(refl1_images, rate_noneq=rate_noneq)
    if isinstance(refl2_images, np.ndarray):
        I_I0_refl2 = compute_refl_i_over_i0(refl2_images, rate_noneq=rate_noneq)
        # -------------------- Hemoglobin Signals --------------------
        dcHBO, dcHBR, dcHBT = convert_to_dchb(
            I_I0_refl1, I_I0_refl2,
            lambdaR_1, lambdaR_2,
            dir_coeff_abs, dir_est_path
        )

            # -------------------- Calcium Correction --------------------
        if Ca_corr_method == 'LSTM':
            Ca_corr = apply_lstm_correction_2d(f_f0, dcHBT)
        elif Ca_corr_method == 'BLB':
            Ca_corr = apply_blb_correction(
                f_f0, dcHBO, dcHBR,
                lambdaF_ex, lambdaF_em,
                dir_coeff_abs, Xest_ex, Xest_em
            )
        else:
            raise ValueError(f"Unknown Ca_corr_method: {Ca_corr_method}")

            # -------------------- Global Signal --------------------
        
        
        if Remove_global:
            
            Ca_corr, Ca_corr_global = remove_global_signal(Ca_corr, mask, bregma=bregma, smooth_win=global_smooth_win, plot=plot_control)
            dcHBO, dcHBO_global = remove_global_signal(dcHBO, mask, bregma, smooth_win=global_smooth_win, plot=plot_control)
            dcHBR, dcHBR_global = remove_global_signal(dcHBR, mask, bregma, smooth_win=global_smooth_win, plot=plot_control)
            dcHBT, dcHBR_global = remove_global_signal(dcHBT, mask, bregma, smooth_win=global_smooth_win, plot=plot_control)
            f_f0, f_f0_global = remove_global_signal(f_f0, mask, bregma=bregma, smooth_win=global_smooth_win, plot=plot_control)
        
        
        # -------------------- Smoothing --------------------
        f_f0_gauss = ndimage.gaussian_filter(f_f0, sigma=1)
        Ca_corr_gauss = ndimage.gaussian_filter(Ca_corr, sigma=1)
        dcHBO_gauss = ndimage.gaussian_filter(dcHBO, sigma=1)
        dcHBR_gauss = ndimage.gaussian_filter(dcHBR, sigma=1)
        dcHBT_gauss = ndimage.gaussian_filter(dcHBT, sigma=1)

        if plot_control:
            show_figures(
                [f_f0, f_f0_gauss, Ca_corr, Ca_corr_gauss],
                ['ΔF/F', 'ΔF/F (gauss)', 'Ca_corr raw', 'Ca_corr (gauss)'],
                im_n, mask
            )

        # -------------------- Saving --------------------
        if Save_conv:
            save_path = res_dir / 'Conv_signal'
            save_path.mkdir(parents=True, exist_ok=True)
            filename = f'{mouse}_{exp}_{protocol}_{state}_conv'
    
            if Save_format.lower() == "hdf5":
                with h5py.File(save_path / f'{filename}.hdf5', 'w') as f:
                    f.create_dataset('Ca_corr', data=Ca_corr_gauss)
                    f.create_dataset('Ca_wocorr', data=f_f0_gauss)
                    f.create_dataset('dcHBO_gauss', data=dcHBO_gauss)
                    f.create_dataset('dcHBR_gauss', data=dcHBR_gauss)
                    f.create_dataset('dcHBT_gauss', data=dcHBT_gauss)
    
            elif Save_format.lower() == "zarr":
                root = zarr.open(str(save_path / f'{filename}.zarr'), mode='w')
                root.create_dataset('Ca_corr', data=Ca_corr_gauss, shape=Ca_corr_gauss.shape, dtype='float32')
                root.create_dataset('Ca_wocorr', data=f_f0_gauss, shape=f_f0_gauss.shape, dtype='float32')
                root.create_dataset('dcHBO_gauss', data=dcHBO_gauss, shape=dcHBO_gauss.shape, dtype='float32')
                root.create_dataset('dcHBR_gauss', data=dcHBR_gauss, shape=dcHBR_gauss.shape, dtype='float32')
                root.create_dataset('dcHBT_gauss', data=dcHBT_gauss, shape=dcHBT_gauss.shape, dtype='float32')
            else:
                raise ValueError(f"Unsupported Save_format: {Save_format}")
    
        if plot_control:
            show_figures(
                [dcHBO_gauss, dcHBR_gauss, dcHBT_gauss, Ca_corr_gauss],
                ['dcHBO', 'dcHBR', 'dcHBT', 'Ca_corr'], im_n, mask
            )
            show_figures(
                [dcHBT, dcHBT_gauss],
                ['dcHBT (raw)', 'dcHBT (gauss)'], im_n, mask
            )
    else:
        dcHBT = - I_I0_refl1
        Ca_corr = apply_lstm_correction_2d(f_f0, dcHBT)
        
        # -------------------- Global Signal --------------------
         
        if Remove_global:
            Ca_corr, Ca_corr_global = remove_global_signal(Ca_corr, mask, bregma=bregma, smooth_win=global_smooth_win, plot=plot_control)
            dcHBT, dcHBR_global = remove_global_signal(dcHBT, mask, bregma, smooth_win=global_smooth_win, plot=plot_control)
            f_f0, f_f0_global = remove_global_signal(f_f0, mask, bregma=bregma, smooth_win=global_smooth_win, plot=plot_control)
        
        
        # -------------------- Smoothing --------------------
        f_f0_gauss = ndimage.gaussian_filter(f_f0, sigma=1)
        Ca_corr_gauss = ndimage.gaussian_filter(Ca_corr, sigma=1)
        dcHBT_gauss = ndimage.gaussian_filter(dcHBT, sigma=1)

        if plot_control:
            show_figures(
                [f_f0, f_f0_gauss, Ca_corr, Ca_corr_gauss],
                ['ΔF/F', 'ΔF/F (gauss)', 'Ca_corr raw', 'Ca_corr (gauss)'],
                im_n, mask
            )

        dcHBR_gauss = np.array([np.nan])
        dcHBO_gauss = np.array([np.nan])
        # -------------------- Saving --------------------
        if Save_conv:
            save_path = res_dir / 'Conv_signal'
            save_path.mkdir(parents=True, exist_ok=True)
            filename = f'{mouse}_{exp}_{protocol}_{state}_conv'
    
            if Save_format.lower() == "hdf5":
                with h5py.File(save_path / f'{filename}.hdf5', 'w') as f:
                    f.create_dataset('Ca_corr', data=Ca_corr_gauss)
                    f.create_dataset('Ca_wocorr', data=f_f0_gauss)
                    f.create_dataset('dcHBO_gauss', data=dcHBO_gauss)
                    f.create_dataset('dcHBR_gauss', data=dcHBR_gauss)
                    f.create_dataset('dcHBT_gauss', data=dcHBT_gauss)
    
            elif Save_format.lower() == "zarr":
                root = zarr.open(str(save_path / f'{filename}.zarr'), mode='w')
                root.create_dataset('Ca_corr', data=Ca_corr_gauss, shape=Ca_corr_gauss.shape, dtype='float32')
                root.create_dataset('Ca_wocorr', data=f_f0_gauss, shape=f_f0_gauss.shape, dtype='float32')
                root.create_dataset('dcHBO_gauss', data=dcHBO_gauss, shape=dcHBO_gauss.shape, dtype='float32')
                root.create_dataset('dcHBR_gauss', data=dcHBR_gauss, shape=dcHBR_gauss.shape, dtype='float32')
                root.create_dataset('dcHBT_gauss', data=dcHBT_gauss, shape=dcHBT_gauss.shape, dtype='float32')
            else:
                raise ValueError(f"Unsupported Save_format: {Save_format}")
    
        if plot_control:
            show_figures(
                [dcHBT_gauss, Ca_corr_gauss],
                ['dcHBT', 'Ca_corr'], im_n, mask
            )
            show_figures(
                [dcHBT, dcHBT_gauss],
                ['dcHBT (raw)', 'dcHBT (gauss)'], im_n, mask
            )

        
    return Ca_corr_gauss, dcHBO_gauss, dcHBR_gauss, dcHBT_gauss, moved_img


def load_aligned_images(result_path: Path, save_format: str) -> Tuple[Union[np.ndarray, str], Union[np.ndarray, str], Union[np.ndarray, str]]:
    """
    Load aligned image stacks (fluorescence and/or reflectance) from a result file
    in HDF5 or Zarr format.

    Parameters
    ----------
    result_path : Path
        Path to the result file (.hdf5 or .zarr).
    save_format : str
        Format of the result file: either 'hdf5' or 'zarr'.

    Returns
    -------
    tuple of np.ndarray or str
        A tuple containing:
        - moved_fluo_images : np.ndarray or 'NaN'
        - moved_refl1_images : np.ndarray or 'NaN'
        - moved_refl2_images : np.ndarray or 'NaN'

    Raises
    ------
    FileNotFoundError
        If the file at result_path does not exist.
    ValueError
        If the given save_format is unsupported.
    """

    if not result_path.exists():
        raise FileNotFoundError(f"{save_format.upper()} file not found: {result_path}")

    fluo = 'NaN'
    refl1 = 'NaN'
    refl2 = 'NaN'

    if save_format.lower() == "hdf5":
        import h5py
        with h5py.File(result_path, 'r') as f:
            if 'moved_fluo_images' in f:
                fluo = np.array(f['moved_fluo_images'])
            if 'moved_refl1_images' in f:
                refl1 = np.array(f['moved_refl1_images'])
            if 'moved_refl2_images' in f:
                refl2 = np.array(f['moved_refl2_images'])

    elif save_format.lower() == "zarr":
        import zarr
        root = zarr.open(str(result_path), mode='r')
        if 'moved_fluo_images' in root:
            fluo = root['moved_fluo_images'][()]
        if 'moved_refl1_images' in root:
            refl1 = root['moved_refl1_images'][()]
        if 'moved_refl2_images' in root:
            refl2 = root['moved_refl2_images'][()]

    else:
        raise ValueError(f"Unknown save format: {save_format}")

    return fluo, refl1, refl2

def load_mask(
    res_dir: Path,
    mouse: str,
    exp: str,
    protocol: str,
    state: str,
    config: Dict[str, any]
) -> np.ndarray:
    """
    Loads and resizes a binary brain mask from HDF5 or image format.

    Parameters
    ----------
    res_dir : Path
        Base results directory.
    mouse : str
        Mouse identifier.
    exp : str
        Experiment identifier.
    protocol : str
        Protocol or condition name.
    state : str
        State of the animal (e.g., 'Awake').
    config : dict
        Configuration dictionary with keys:
        - 'mask_format': Format of the mask file (e.g., 'hdf5', 'png', 'tiff').
        - 'scalingfactor': Resize factor in percent.

    Returns
    -------
    np.ndarray
        Binary brain mask as a boolean array (True = brain, False = background).

    Raises
    ------
    ValueError
        If the specified mask format is unsupported.
    """
    base_name = f"{mouse}_{exp}_{protocol}_{state}_mask_draw"
    mask_path = res_dir / 'Masks' / 'Mask_draw'

    format = config['mask_format'].lower()
    if format == "hdf5":
        file_path = mask_path / f"{base_name}.hdf5"
        with h5py.File(file_path, "r") as f:
            mask = np.array(f['Mask_draw'])
    elif format in ["png", "tiff", "bmp"]:
        file_path = mask_path / f"{base_name}.{format}"
        mask = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
    else:
        raise ValueError(f"Unsupported mask format: {config['mask_format']}")

    mask = resize(mask, config["scalingfactor"])
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return binary_mask.astype(bool)

def process_mouse_experiment(
    img_dir: Path,
    exp: str,
    mouse: str,
    state: str,
    protocol: str,
    config: dict,
    res_dir: Path,
    demonstr_dir: Path,
    info_dir: Path,
    info_moving_dir: Path,
    log_df: pd.DataFrame,
    log_path: Path
) -> pd.DataFrame:
    """
    Process a single mouse experiment, including alignment and GCaMP signal conversion.

    This function performs the full processing pipeline for one experiment: it checks whether
    alignment and conversion were already done, performs image alignment if needed, loads the
    appropriate brain mask, computes ΔF/F and corrected GCaMP signals, and saves the results.
    Processing status is logged to a CSV file.

    Args:
        img_dir (Path): Directory containing raw image frames for the experiment.
        exp (str): Experiment identifier.
        mouse (str): Mouse identifier.
        state (str): State of the mouse (e.g., "Awake", "Anesthetized").
        protocol (str): Protocol or stimulus name.
        config (dict): Configuration dictionary with processing parameters.
        res_dir (Path): Directory where results will be saved.
        demonstr_dir (Path): Directory for saving demonstration preview images.
        info_dir (Path): Directory for metadata (unused).
        info_moving_dir (Path): Path to movement and alignment CSV metadata.
        log_df (pd.DataFrame): DataFrame containing the processing log.
        log_path (Path): Path to the CSV file for logging experiment status.

    Returns:
        pd.DataFrame: Updated processing log DataFrame with current experiment status.
    """
    print(f"[DEBUG] Processing: {mouse} / {exp} / {protocol} / {state}")
    if log_df is None:
        raise ValueError("[ERROR] log_df is None at entry to function.")

    result_filename = f"{mouse}_{exp}_{protocol}_{state}.{config['save_format']}"

    # Check if experiment already fully processed
    condition = (
        (log_df["mouse"] == mouse) &
        (log_df["exp"] == exp) &
        (log_df["protocol"] == protocol) &
        (log_df["state"] == state)
    )
    already_done = (
        not log_df[condition].empty and
        log_df.loc[condition, 'conv_status'].iloc[0] == 'success'
    )
    if already_done:
        print(f"[SKIP] Already fully processed: {mouse}, {exp}, {protocol}, {state}")
        return log_df

    # Load aligned images if already processed
    if already_processed(log_df, mouse, exp, protocol, state):
        print(f"[LOAD] Alignment already done: {mouse}, {exp}, {protocol}, {state}")
        try:
            moved_fluo_images, moved_refl1_images, moved_refl2_images = load_aligned_images(
                res_dir / "Moved_data" / result_filename,
                config["save_format"]
            )
        except Exception as e:
            print(f"[ERROR] Failed to load aligned result: {e}")
            return log_df
    else:
        # Run alignment step
        print(f"[PROCESSING] Alignment: {mouse}, {exp}, {protocol}, {state}")
        try:
            moved_fluo_images, moved_refl1_images, moved_refl2_images, bregma, img_size, err = nparrayFromimages_2waves_IOS_GCAMP(
                img_dir=img_dir,
                res_dir=res_dir,
                demonstr_dir=demonstr_dir,
                info_dir=info_dir,
                info_moving_dir=info_moving_dir,
                protocol=protocol,
                state=state,
                exp=exp,
                mouse=mouse,
                img_number=config["img_number"],
                rotation=config["rotation"],
                scalingfactor=config["scalingfactor"],
                br_y=config["br_y"],
                Save_format=config["save_format"],
                ex_ptrn=config.get('lambdaF_ex_pattern'),
                r1_ptrn=config.get('lambdaR_1_pattern'),
                r2_ptrn=config.get('lambdaR_2_pattern'))
        except Exception as e:
            print(f"[ERROR] Alignment step failed: {e}")
            return log_df

        log_df = log_result(
            log_path, log_df,
            mouse, exp, protocol, state,
            result_filename, move_status=err
        )
        if log_df is None:
            raise ValueError("[ERROR] log_df became None after log_result() call.")

        if err != "Ok":
            print(f"[STATUS] Alignment failed: {err}")
            return log_df

    # Plot raw image with mask (optional)
    if config.get("plot_control", False):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(moved_fluo_images[0], cmap='gray')
        plt.title("Fluorescence Image (raw)")
        plt.axis('off')

    # Load binary brain mask
    try:
        mask = load_mask(res_dir, mouse, exp, protocol, state, config)
        if config.get("plot_control", False):
            import numpy.ma as ma
            masked_img = ma.masked_array(moved_fluo_images[0], mask=~mask)
            plt.subplot(1, 2, 2)
            plt.imshow(masked_img, cmap='gray')
            plt.title("Fluorescence + Mask")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
    except Exception as e:
        print(f"[ERROR] Failed to load mask: {e}")
        return log_df

    # Run signal conversion to Ca2+ and hemoglobin
    try:
        print(f"[DEBUG] Starting GCaMP conversion for {mouse} / {exp}")
        img_size = moved_fluo_images[0].shape
        bregma = math.ceil(img_size[1] / 2)

        Ca_corr, dcHBO_gauss, dcHBR_gauss, dcHBT_gauss, moved_img = imgs_convertion_2waves_IOS_GCaMP(
            moved_fluo_images, moved_refl1_images, moved_refl2_images,
            mask, res_dir, info_dir, protocol, state, exp, mouse,
            config['lambdaF_ex'], config['lambdaF_em'], config['lambdaR_1'], config.get('lambdaR_2'),
            config['Ca_corr_method'], config['rate_noneq'],
            config['Xest_ex'], config['Xest_em'],
            config['dir_coeff_abs'], config['dir_est_path'],
            rate=config['rate'], bregma=bregma,
            Save_conv=config['save_conv'], Save_format=config['save_format'],
            plot_control=config['plot_control'], im_n=config['im_n'],
            Remove_global=config['Remove_global'], global_smooth_win=config['global_smooth_win']
        )

        idx = log_df[
            (log_df['mouse'] == mouse) &
            (log_df['exp'] == exp) &
            (log_df['protocol'] == protocol) &
            (log_df['state'] == state)
        ].index

        if not idx.empty:
            log_df.loc[idx, 'conv_status'] = 'success'
            log_df.to_csv(log_path, index=False)
        print(f"[DEBUG] Conversion success logged for {mouse}, {exp}")
    except Exception as e:
        print(f"[ERROR] GCaMP conversion failed: {e}")
        idx = log_df[
            (log_df['mouse'] == mouse) &
            (log_df['exp'] == exp) &
            (log_df['protocol'] == protocol) &
            (log_df['state'] == state)
        ].index
        if not idx.empty:
            log_df.loc[idx, 'conv_status'] = 'failed'
            log_df.to_csv(log_path, index=False)

    print(f"[DEBUG] Finished process for {mouse} / {exp}")
    return log_df