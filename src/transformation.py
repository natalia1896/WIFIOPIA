import os
import math
import cv2
import imutils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Union

from common import sortByNumberofFrame, load_or_create_csv

def move_vertical(data: np.ndarray, shift: int) -> np.ndarray:
    """
    Vertically shifts an image or stack of images along the Y-axis.

    Args:
        data (np.ndarray): A 2D array (H, W) or 3D array (D, H, W) representing a single image
            or a sequence of images (frames).
        shift (int): Number of pixels to shift along the Y-axis.
            - Positive value shifts upward (toward lower indices)
            - Negative value shifts downward (toward higher indices)

    Returns:
        np.ndarray: The vertically shifted array, same shape as the input.
    
    Notes:
        - Areas exposed by the shift are zero-filled.
        - If input is 2D, it is temporarily expanded to 3D for uniform processing and then returned back as 2D.
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # Convert to 3D for consistent handling
        squeeze = True
    else:
        squeeze = False

    h = data.shape[1]
    shifted = np.zeros_like(data)
    if shift > 0:
        shifted[:, :h-shift, :] = data[:, shift:, :]
    elif shift < 0:
        shifted[:, -shift:, :] = data[:, :h+shift, :]
    else:
        shifted = data.copy()

    return shifted[0] if squeeze else shifted


def move_horizontal(data: np.ndarray, shift: int) -> np.ndarray:
    """
    Horizontally shifts an image or a stack of images along the X-axis.

    Args:
        data (np.ndarray): A 2D array (H, W) or 3D array (D, H, W) representing a single image
            or a sequence of images (frames).
        shift (int): Number of pixels to shift along the X-axis.
            - Positive value shifts left (toward lower indices)
            - Negative value shifts right (toward higher indices)

    Returns:
        np.ndarray: The horizontally shifted array, same shape as the input.
    
    Notes:
        - Areas revealed by the shift are zero-filled.
        - If the input is 2D, it is temporarily expanded to 3D during processing and returned back as 2D.
    """
    if data.ndim == 2:
        data = data[np.newaxis, ...]  # Convert to 3D for uniform processing
        squeeze = True
    else:
        squeeze = False

    w = data.shape[2]
    shifted = np.zeros_like(data)
    if shift > 0:
        shifted[:, :, :w-shift] = data[:, :, shift:]
    elif shift < 0:
        shifted[:, :, -shift:] = data[:, :, :w+shift]
    else:
        shifted = data.copy()

    return shifted[0] if squeeze else shifted

def process_single_experiment(
    img_dir: Path,
    exp: str,
    mouse: str,
    protocol: str,
    state: str,
    config: dict[str, Any],
    info_dir: Path,
    demonstr_dir: Path,
    image: int,
    br_y: float
) -> None:
    """
    Process a single experiment: mark anatomical points, rotate and shift brain image,
    save transformed image and record experiment info.

    Args:
        img_dir (Path): Path to directory with input image frames.
        exp (str): Name or ID of the experiment day.
        mouse (str): Mouse folder name.
        protocol (str): Protocol name.
        state (str): State condition name (e.g., 'Stim', 'Baseline').
        config (dict): Configuration dictionary (must contain "rotation" and optional "mark_Tvt").
        info_dir (Path): Directory where transformation metadata (`info_moving.csv`) is stored.
        demonstr_dir (Path): Directory to save processed (moved) image for demonstration.
        image (int): Index of the frame in the folder to use for alignment and marking.
        br_y (float): Relative vertical position of bregma in original image (0–1, top to bottom).

    Returns:
        None. Saves transformed image and updates info CSV.

    Raises:
        FileNotFoundError: If input image file is missing.
        ValueError: If user input is malformed.
        KeyError: If required config keys are missing.

    Notes:
        - Skips processing if the experiment is already listed in `info_moving.csv`.
        - Requires user to click points via `matplotlib.ginput()`.
        - All outputs are saved automatically (image and CSV row).
    """
    fname_info_moving = info_dir / 'info_moving.csv'
    columns = [
        'mouse#', 'protocol', 'exp', 'state', 'U_x', 'U_y', 'D_x', 'D_y',
        'angle', 'Br_x', 'Br_y', 'Brt_x', 'Brt_y', 'Lbt_x', 'Lbt_y',
        'Tvt_x', 'Tvt_y', 'move_vert', 'move_horis', 'invert', 'date'
    ]
    old_info_df = load_or_create_csv(fname_info_moving, columns)

    # Skip if already processed
    if not old_info_df[
        (old_info_df['mouse#'] == mouse) &
        (old_info_df['exp'] == exp) &
        (old_info_df['protocol'] == protocol)
    ].empty:
        return

    # Read and sort frames
    path_for_frames = os.listdir(img_dir)
    if 'Thumbs.db' in path_for_frames:
        path_for_frames.remove('Thumbs.db')
    path_for_frames.sort(key=sortByNumberofFrame)
    
    date = path_for_frames[1].split('_')[2][:6]

    # Load and rotate image
    img_path = img_dir / path_for_frames[image]
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    img_rotated = imutils.rotate_bound(img, config["rotation"])

    # User input: suture points
    plt.imshow(img_rotated, cmap='gray')
    plt.title("2 points on suture: Up and Down")
    coord = plt.ginput(2, timeout=-1)
    plt.close()

    U_x, U_y = map(round, coord[0])
    D_x, D_y = map(round, coord[1])
    tang = abs(U_x - D_x) / abs(D_y - U_y)
    angle = np.arctan(tang) * 180 / np.pi * (-1 if U_x - D_x <= 0 else 1)

    # Rotate based on suture
    rotated_img = imutils.rotate(img_rotated, angle)
    plt.imshow(rotated_img, cmap='gray')
    plt.title("Bregma")
    Br = plt.ginput(1, timeout=-1)
    plt.close()

    move_vert_var = round(Br[0][1]) - math.ceil(br_y * img.shape[0])
    move_horis_var = round(Br[0][0]) - math.ceil(img.shape[1] / 2)

    # Apply transformations
    moved_brain = move_vertical(
        move_horizontal(rotated_img, move_horis_var), move_vert_var
    )
    fname_img = demonstr_dir / f'{mouse}_{exp}_{protocol}_{state}_moved.tiff'
    plt.imsave(fname_img, moved_brain, cmap='gray')

    # Mark additional landmarks
    plt.imshow(moved_brain, cmap='gray')
    plt.title("New_Bregma, Lambda")
    ap = plt.ginput(2, timeout=-1)
    plt.close()

    # Optional TVT point
    Tvt_x = Tvt_y = 0
    if config.get("mark_Tvt"):
        plt.imshow(moved_brain, cmap='gray')
        plt.title("Mark Tvt point")
        Tvt = plt.ginput(1, timeout=-1)
        plt.close()
        Tvt_x, Tvt_y = map(round, Tvt[0])

    invert = input("Enter 1 if the wavelengths are mixed up, or 0 otherwise: ")

    # Save transformation metadata
    info = {
        'mouse#': mouse, 'protocol': protocol, 'exp': exp, 'state': state,
        'U_x': U_x, 'U_y': U_y, 'D_x': D_x, 'D_y': D_y,
        'angle': angle, 'Br_x': round(Br[0][0]), 'Br_y': round(Br[0][1]),
        'Brt_x': round(ap[0][0]), 'Brt_y': round(ap[0][1]),
        'Lbt_x': round(ap[1][0]), 'Lbt_y': round(ap[1][1]),
        'Tvt_x': Tvt_x, 'Tvt_y': Tvt_y,
        'move_vert': move_vert_var, 'move_horis': move_horis_var,
        'invert': invert, 'date': date
    }

    info_df = pd.DataFrame([info])
    updated_df = pd.concat([old_info_df, info_df], ignore_index=True)
    updated_df.to_csv(fname_info_moving, index=False)

def resize(image: Union[str, Path, np.ndarray], scale: Union[float, int]) -> np.ndarray:
    """
    Resize an image while maintaining its proportions.

    Args:
        image (str, Path, or np.ndarray): Path to an image file (grayscale),
            or a preloaded grayscale image as a NumPy array.
        scale (float or int): Scaling factor as a percentage of the original size
            (e.g., 25 means 25%).

    Returns:
        np.ndarray: Scaled image as a 2D NumPy array.

    Raises:
        FileNotFoundError: If the provided image path is invalid.
        ValueError: If the image array has invalid shape or the scale is non-positive.
        TypeError: If input type is not supported.

    Notes:
        - The image is loaded in grayscale mode if a path is provided.
        - Uses `cv2.INTER_AREA` interpolation (suitable for downscaling).
    """
    def _scale(dim: int, s: float) -> int:
        return int(dim * s / 100)

    # Load image from file path
    if isinstance(image, (str, Path)):
        image = Path(image)
        if not image.exists():
            raise FileNotFoundError(f"Image file not found: {image}")
        im = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise ValueError(f"Unable to load image from: {image}")
    elif isinstance(image, np.ndarray):
        im = image
    else:
        raise TypeError("image must be a path (str or Path) or a NumPy array")

    if scale <= 0:
        raise ValueError("Scale must be a positive number")

    if im.ndim != 2:
        raise ValueError("Expected a 2D grayscale image")

    height, width = im.shape
    new_width = _scale(width, scale)
    new_height = _scale(height, scale)
    new_dim = (new_width, new_height)

    return cv2.resize(src=im, dsize=new_dim, interpolation=cv2.INTER_AREA)