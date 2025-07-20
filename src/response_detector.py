import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import pandas as pd
from PIL import Image
from skimage import measure
from matplotlib.colors import Normalize

from common import (
    load_conv_results)
from signal_converter import (
    load_mask)

def Response_mask(crop_img_L: np.ndarray, threshold: int) -> np.ndarray:
    """
    Extracts the largest binary component from a thresholded image.

    Applies Gaussian blur, binary thresholding, erosion, dilation, and connected component
    labeling to isolate the largest structure in the binary image.

    Args:
        crop_img_L (np.ndarray): Grayscale input image (2D).
        threshold (int): Threshold value to binarize the image (0–255).

    Returns:
        np.ndarray: Binary mask of the largest connected component (1 = object, 0 = background).
    """
    # Apply Gaussian blur
    img_b_L = cv2.GaussianBlur(crop_img_L, (3, 3), cv2.BORDER_DEFAULT)

    # Apply threshold
    _, thresh = cv2.threshold(img_b_L, threshold, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)

    # Apply erosion
    eroded_image = cv2.erode(thresh, kernel, iterations=0)

    kernel2 = np.ones((2, 2), np.uint8)

    # Apply dilation
    dilated_image = cv2.dilate(eroded_image, kernel2, iterations=1)

    # Connected components
    labels_img = measure.label(dilated_image, background=0, return_num=False, connectivity=1)

    # Loop over the unique components
    num_pixel = []
    for label in np.unique(labels_img):
        # if this is the background label, ignore it
        if label == 0:
            num_pixel.append(0)
        else:
            # Construct the label mask and count the number of pixels
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels_img == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            num_pixel.append(numPixels)

        # if numPixels > 3:
        #     mask = cv2.add(mask, labelMask)

    if num_pixel == [0]:
        labelMask = np.zeros(thresh.shape, dtype="uint8")
    else:
        num_pixel_arr = np.array(num_pixel)
        max_label = np.unique(labels_img)[np.argmax(num_pixel_arr)]
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels_img == max_label] = 1
        sq = cv2.countNonZero(labelMask)  # Размер области

    return labelMask

def From_sum_to_mask_all(
    sum_Ca_total: np.ndarray,
    mask: np.ndarray,
    stim_type: str,
    stim_type_dict: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates binary response masks for left and right stimulus areas based on the calcium sum image.

    This function performs normalization of the calcium response image, extracts relevant regions 
    using predefined stimulus rectangles, calculates a dynamic threshold, and generates binary masks 
    representing the most prominent response areas in left and right hemispheres.

    Args:
        sum_Ca_total (np.ndarray): 2D array of the total calcium response.
        mask (np.ndarray): Binary brain mask (True for brain region).
        stim_type (str): Stimulus type identifier (e.g., 'L1', 'R2').
        stim_type_dict (dict): Dictionary mapping stimulus types to bounding box coordinates 
            in the form (y1, y2, x1, x2).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: 
            - mask_L: Binary response mask for the left hemisphere.
            - mask_R: Binary response mask for the right hemisphere.
            - sum_Ca_total: Normalized and scaled calcium image (values in 0–255).
    """
    # Mask outside the brain
    mdata_Ca = np.ma.masked_array(sum_Ca_total, mask=~np.array(mask, dtype=bool))
    mdata_Ca = np.ma.filled(mdata_Ca, np.nan)

    # Normalize the calcium signal using percentiles
    sum_Ca_total = (sum_Ca_total - np.nanpercentile(mdata_Ca, 2)) / (
        np.nanpercentile(mdata_Ca, 99) - np.nanpercentile(mdata_Ca, 1)
    )
    sum_Ca_total *= 255
    sum_Ca_total[sum_Ca_total > 255] = 255
    sum_Ca_total[sum_Ca_total < 0] = 0

    # Create a region-of-interest (ROI) mask from stim_type
    y1, y2, x1, x2 = stim_type_dict[stim_type]
    new_mask = np.zeros(sum_Ca_total.shape)
    new_mask[y1:y2, x1:x2] = 1

    # Combine brain mask with stimulus ROI
    merge_mask = new_mask + mask
    merge_mask = np.where(merge_mask == 2, 1, 0)

    # Compute threshold from maximum within ROI
    mdata_b = np.ma.masked_array(sum_Ca_total, mask=~np.array(merge_mask, dtype=bool))
    mdata_b = np.ma.filled(mdata_b, np.nan)
    threshold = np.nanmax(mdata_b) * 0.86

    # Convert to 8-bit image
    sum_Ca_total_8b = sum_Ca_total.astype(np.uint8)
    img = sum_Ca_total_8b

    # Extract left and right response regions using symmetric stimulus definitions
    y1_L, y2_L, x1_L, x2_L = stim_type_dict['R' + stim_type[1:]]
    y1_R, y2_R, x1_R, x2_R = stim_type_dict['L' + stim_type[1:]]

    crop_img_L = img[y1_L:y2_L, x1_L:x2_L]
    crop_img_R = img[y1_R:y2_R, x1_R:x2_R]

    # Generate response masks for both sides
    mask_L = Response_mask(crop_img_L, threshold)
    mask_R = Response_mask(crop_img_R, threshold)

    return mask_L, mask_R, sum_Ca_total

def draw_edge(mask_Ca_R: np.ndarray) -> np.ndarray:
    """
    Detects the edges of a binary calcium response mask using the Canny edge detector.

    Args:
        mask_Ca_R (np.ndarray): Binary mask (1 for region of interest, 0 elsewhere).

    Returns:
        np.ndarray: Binary edge image (255 on edges, 0 elsewhere).
    """
    # Copy and scale binary mask to 8-bit range
    mask = mask_Ca_R.copy()
    mask = (mask * 255).astype(np.uint8)

    # Apply Canny edge detection
    edged = cv2.Canny(mask, 30, 200)

    return edged

def resize_mask(
    mask_Ca_L: np.ndarray,
    mask_Ca_R: np.ndarray,
    img_size: tuple[int, int],
    stim_type: str,
    stim_type_dict: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Places the cropped left and right calcium response masks back into full-size image masks.

    Args:
        mask_Ca_L (np.ndarray): Cropped mask for the left hemisphere.
        mask_Ca_R (np.ndarray): Cropped mask for the right hemisphere.
        img_size (tuple[int, int]): Size of the full image (height, width).
        stim_type (str): Stimulus label indicating the current stimulation type (e.g., 'Lpaw').
        stim_type_dict (dict): Dictionary mapping stimulation types to bounding boxes (y1, y2, x1, x2).

    Returns:
        tuple[np.ndarray, np.ndarray]: Full-size left and right hemisphere masks.
    """
    # Determine bounding boxes by flipping hemisphere labels
    y1_L, y2_L, x1_L, x2_L = stim_type_dict['R' + stim_type[1:]]
    y1_R, y2_R, x1_R, x2_R = stim_type_dict['L' + stim_type[1:]]

    # Allocate new full-size arrays and insert cropped masks into correct regions
    new_mask_Ca_L = np.zeros(img_size)
    new_mask_Ca_L[y1_L:y2_L, x1_L:x2_L] = mask_Ca_L

    new_mask_Ca_R = np.zeros(img_size)
    new_mask_Ca_R[y1_R:y2_R, x1_R:x2_R] = mask_Ca_R

    return new_mask_Ca_L, new_mask_Ca_R

def Center_detector(img: np.ndarray) -> tuple[int, int]:
    """
    Detects the center (centroid) of a binary object in a grayscale image using image moments.

    Args:
        img (np.ndarray): Input grayscale image where the object is assumed to be white on black background (float or uint8, values in range 0–1 or 0–255).

    Returns:
        tuple[int, int]: Coordinates of the object's center (cX, cY).
    
    Raises:
        ZeroDivisionError: If the object has zero area (M["m00"] == 0), causing division by zero.
    """
    # Scale the grayscale image to 0–255 range
    thresh = img * 255

    # Compute spatial moments of the image
    M = cv2.moments(thresh)

    # Compute centroid from moments
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY

def Create_ROI_dict(br_y_coeff: float, img_size: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    """
    Creates a dictionary of predefined region-of-interest (ROI) coordinates for various stimulus types,
    based on image size and bregma-relative vertical position.

    Args:
        br_y_coeff (float): Vertical coefficient (0–1) representing the relative position of bregma along the y-axis.
        img_size (tuple[int, int]): Image dimensions as (height, width).

    Returns:
        dict[str, tuple[int, int, int, int]]: Dictionary mapping stimulus labels to rectangular ROI coordinates
            in the format (y_start, y_end, x_start, x_end).
    
    Example:
        >>> Create_ROI_dict(0.6, (96, 128))
        {
            'RHL': (51, 80, 35, 60),
            'LHL': (51, 80, 68, 93),
            ...
        }
    """
    y, x = img_size
    br_x = int(x / 2)                # Bregma x-coordinate (center of image width)
    br_y = int(br_y_coeff * y)       # Bregma y-coordinate (based on provided coefficient)

    stim_type_dict = {
        'RHL': (br_y - 6, br_y + 23, br_x - 29, br_x - 4),
        'LHL': (br_y - 6, br_y + 23, br_x + 4, br_x + 29),
        'RE':  (br_y + 17, br_y + 38, br_x - 30, br_x - 4),
        'LE':  (br_y + 17, br_y + 38, br_x + 4, br_x + 30),
        'RV':  (br_y + 7,  br_y + 28, br_x - 40, br_x - 25),
        'LV':  (br_y + 7,  br_y + 28, br_x + 25, br_x + 40),
    }

    return stim_type_dict

def stim_sum_flexible(
    dataset: np.ndarray,
    stim_count: int = 21,
    stim_start_frame: int = 320,
    stim_step: int = 400,
    stim_window: tuple[int, int] = (0, 100),
    subtr_baseline: str = 'prestart_mean',
    subtr_baseline_interval: int = 20,
    substim_count: int | None = None,
    substim_interval: int | None = None,
    substim_window: tuple[int, int] = (1, 4)
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Flexibly sums stimulus-related responses from a 3D time-series dataset.

    This function allows extraction of response windows across repeated stimuli,
    with optional baseline subtraction and sub-stimulus decomposition.

    Args:
        dataset (np.ndarray): 3D array of shape (T, Y, X) representing a time-series image stack.
        stim_count (int): Number of stimulus repetitions.
        stim_start_frame (int): Frame index of the first stimulus onset.
        stim_step (int): Interval in frames between consecutive stimuli.
        stim_window (tuple[int, int]): Time window (start, end) relative to stimulus onset to extract data.
        subtr_baseline (str): Type of baseline subtraction:
            - 'mean': use mean over entire dataset
            - 'prestart_mean': use mean of frames before each stimulus
            - 'no': no baseline subtraction
        subtr_baseline_interval (int): Number of frames before stimulus used for 'prestart_mean' baseline.
        substim_count (int or None): If set, divides each stimulus period into sub-stimuli.
        substim_interval (int or None): Total frame span of each stimulus to divide into sub-stimuli.
        substim_window (tuple[int, int]): Time window relative to sub-stimulus for summing signal.

    Returns:
        tuple:
            dict[str, np.ndarray]: Dictionary of stimulus responses with keys 'st1', 'st2', ..., each of shape (window, Y, X).
            np.ndarray: Summed response image across all stimuli (Y, X).
    """
    stim_dict: dict[str, np.ndarray] = {}
    sum_img: np.ndarray = np.zeros(dataset.shape[1:])

    for i in range(stim_count):
        stim_base = stim_start_frame + i * stim_step
        stim_start = stim_base + stim_window[0]
        stim_end = stim_base + stim_window[1]

        if subtr_baseline == 'mean':
            baseline = np.mean(dataset, axis=0)
        elif subtr_baseline == 'prestart_mean':
            base_start = stim_base - subtr_baseline_interval
            base_end = stim_base
            baseline = np.mean(dataset[base_start:base_end], axis=0)
        elif subtr_baseline == 'no':
            baseline = 0
        else:
            raise ValueError(f"Unknown baseline type: {subtr_baseline}")

        stim_data = dataset[stim_start:stim_end] - baseline
        stim_dict[f'st{i+1}'] = stim_data

        if substim_count is None:
            sum_img += np.sum(stim_data, axis=0)
        else:
            for j in range(substim_count):
                s_start = j * (substim_interval // substim_count) + substim_window[0]
                s_end = j * (substim_interval // substim_count) + substim_window[1]
                sum_img += np.sum(stim_data[s_start:s_end], axis=0)

    return stim_dict, sum_img

def plot_overlay(
    data: np.ndarray,
    mask_overlay: np.ndarray,
    base_mask: np.ndarray,
    title: str
) -> None:
    """
    Display an overlay of masked data and a base mask for visualization.

    Pixels where `mask_overlay == 255` are set to zero in `data` before masking.
    The result is then masked using `base_mask` and displayed as an image.

    Args:
        data (np.ndarray): 2D array of image data (e.g., activation or intensity values).
        mask_overlay (np.ndarray): 2D binary mask array where values of 255 mark areas to zero out in `data`.
        base_mask (np.ndarray): Boolean array where `True` defines the visible region for plotting.
        title (str): Title for the plot.

    Returns:
        None
    """
    # Zero out areas defined by mask_overlay
    img = np.where(mask_overlay == 255, 0, data)

    # Apply base mask (e.g., brain mask)
    mdata = np.ma.masked_array(img, mask=~np.array(base_mask, dtype=bool))
    mdata = np.ma.filled(mdata, np.nan)

    # Display with percentiles for contrast stretching
    plt.imshow(
        mdata,
        cmap='jet',
        vmin=np.nanpercentile(mdata, 2),
        vmax=np.nanpercentile(mdata, 98)
    )
    plt.axis('off')
    plt.title(title)

def save_masks(resize_masks_dict, save_path: Path, base_name: str, formats: list):
    """
    Save masks in selected formats (hdf5, png, tiff, bmp)

    Args:
        resize_masks_dict (dict): dict like {'Ca_L': np.array, ...}
        save_path (Path): folder to save
        base_name (str): base file name, without extension
        formats (list): list of formats to save ['hdf5', 'png', 'tiff', 'bmp']
    """
    save_path.mkdir(parents=True, exist_ok=True)

    if 'hdf5' in formats:
        hdf5_path = save_path / f'{base_name}_mask_stim.hdf5'
        with h5py.File(hdf5_path, 'w') as f:
            for name, arr in resize_masks_dict.items():
                f.create_dataset(f'Response_mask_{name}', data=arr)

    for ext in {'png', 'tiff', 'bmp'} & set(formats):
        for name, arr in resize_masks_dict.items():
            img = Image.fromarray((arr * 255).astype('uint8'))
            img.save(save_path / f'{base_name}_{name}_mask.{ext}')
    
def save_overlay_image(data, mask_overlay, base_mask, title: str, save_path: Path):
    """
    Generate and save overlay image (signal + mask) as PNG.

    Args:
        data (np.ndarray): 2D signal (e.g., sum_Ca_total_norm)
        mask_overlay (np.ndarray): binary mask overlay (255 for edges)
        base_mask (np.ndarray): base inclusion mask
        title (str): 'Ca' or 'HBT'
        save_path (Path): full PNG path to save
    """
    # Prepare masked data
    img = np.where(mask_overlay == 255, 0, data)
    mdata = np.ma.masked_array(img, mask=~np.array(base_mask, dtype=bool))
    mdata = np.ma.filled(mdata, np.nan)

    # Create figure
    plt.figure(figsize=(5, 5))
    norm = Normalize(vmin=np.nanpercentile(mdata, 2), vmax=np.nanpercentile(mdata, 98))
    plt.imshow(mdata, cmap='jet', norm=norm)
    plt.axis('off')
    plt.title(title)

    # Save as PNG
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.close()

def process_stim_experiment(
    img_dir: Path,
    exp: str,
    mouse: str,
    state: str,
    protocol: str,
    config: dict,
    stim_config: dict,
    res_dir: Path,
    info_dir: Path
) -> None:
    """
    Processes a stimulation experiment to generate response-based ROI masks for calcium and hemodynamic signals.

    This function loads preprocessed fluorescence and reflectance signals, extracts stimulation-related activity
    windows, computes activation masks, overlays, and center coordinates, and saves results in specified formats.

    Args:
        img_dir (Path): Directory containing raw or processed image data.
        exp (str): Experiment name or identifier.
        mouse (str): Mouse ID.
        state (str): State label (e.g., 'awake', 'anesthetized').
        protocol (str): Full protocol string, from which stimulation type is parsed.
        config (dict): General processing configuration dictionary.
        stim_config (dict): Stimulation-specific configuration, e.g. frame windows, baseline settings, save formats.
        res_dir (Path): Directory where results (e.g., masks, signals) are stored.
        info_dir (Path): Directory where ROI metadata is saved.

    Returns:
        None
    """
    try:
        stim_type = protocol.split('-')[0]
        mask_path = res_dir / f'Mask_stim_{stim_type}'
        mask_path.mkdir(parents=True, exist_ok=True)

        base_name = f'{mouse}_{exp}_{protocol}_{state}'
        full_mask_hdf5 = mask_path / f'{base_name}_mask_stim.hdf5'

        # Skip if already processed
        if 'hdf5' in stim_config.get("save_formats", []) and full_mask_hdf5.exists():
            print(f"Skipping {img_dir} — HDF5 mask already exists.")
            return

        # --- Load data ---
        filename = f'{mouse}_{exp}_{protocol}_{state}_conv'
        Ca_corr, dcHBO, dcHBR, dcHBT = load_conv_results(
            save_path=res_dir / "Conv_signal",
            filename=filename,
            save_format=config['save_format']
        )

        mask = load_mask(res_dir, mouse, exp, protocol, state, config)
        img_size = Ca_corr.shape[1:]
        stim_type_dict = Create_ROI_dict(config['br_y'], img_size)

        # --- Ca signal ---
        stim_dict_Ca, sum_Ca = stim_sum_flexible(
            dataset=Ca_corr,
            stim_count=stim_config["stim_count"],
            stim_start_frame=stim_config["stim_start_frame"],
            stim_step=stim_config["stim_step"],
            stim_window=tuple(stim_config["stim_window_ca"]),
            subtr_baseline=stim_config["subtr_baseline"],
            subtr_baseline_interval=stim_config["subtr_baseline_interval"],
            substim_count=stim_config.get("substim_count"),
            substim_interval=stim_config.get("substim_interval"),
            substim_window=tuple(stim_config.get("substim_window", (0, 0)))
        )

        # --- HBT signal ---
        stim_dict_HBT, sum_HBT = stim_sum_flexible(
            dataset=dcHBT,
            stim_count=stim_config["stim_count"],
            stim_start_frame=stim_config["stim_start_frame"],
            stim_step=stim_config["stim_step"],
            stim_window=tuple(stim_config["stim_window_hbt"]),
            subtr_baseline=stim_config["subtr_baseline"],
            subtr_baseline_interval=stim_config["subtr_baseline_interval"]
        )

        # --- Create masks ---
        mask_Ca_L, mask_Ca_R, sum_Ca_total_norm = From_sum_to_mask_all(sum_Ca, mask, stim_type, stim_type_dict)
        mask_HBT_L, mask_HBT_R, sum_HBT_total_norm = From_sum_to_mask_all(sum_HBT, mask, stim_type, stim_type_dict)

        y1_L, y2_L, x1_L, x2_L = stim_type_dict['R' + stim_type[1:]]
        y1_R, y2_R, x1_R, x2_R = stim_type_dict['L' + stim_type[1:]]

        new_mask_Ca = mask.copy().astype(np.uint8)
        new_mask_Ca[y1_L:y2_L, x1_L:x2_L] = draw_edge(mask_Ca_L)
        new_mask_Ca[y1_R:y2_R, x1_R:x2_R] = draw_edge(mask_Ca_R)

        new_mask_HBT = mask.copy().astype(np.uint8)
        new_mask_HBT[y1_L:y2_L, x1_L:x2_L] = draw_edge(mask_HBT_L)
        new_mask_HBT[y1_R:y2_R, x1_R:x2_R] = draw_edge(mask_HBT_R)

        # --- Resize for saving ---
        resize_mask_Ca_L, resize_mask_Ca_R = resize_mask(mask_Ca_L, mask_Ca_R, img_size, stim_type, stim_type_dict)
        resize_mask_HBT_L, resize_mask_HBT_R = resize_mask(mask_HBT_L, mask_HBT_R, img_size, stim_type, stim_type_dict)

        # --- Save masks in selected formats ---
        resize_masks_dict = {
            'Ca_L': resize_mask_Ca_L,
            'Ca_R': resize_mask_Ca_R,
            'HBT_L': resize_mask_HBT_L,
            'HBT_R': resize_mask_HBT_R
        }

        save_masks(
            resize_masks_dict=resize_masks_dict,
            save_path=mask_path,
            base_name=base_name,
            formats=stim_config.get("save_formats", ["hdf5"])
        )

        # --- Save overlay PNGs ---
        if "overlay" in stim_config.get("save_formats", []):
            save_overlay_image(
                data=sum_Ca_total_norm,
                mask_overlay=new_mask_Ca,
                base_mask=mask,
                title='Ca',
                save_path=mask_path / f'{base_name}_overlay_Ca.png'
            )
            save_overlay_image(
                data=sum_HBT_total_norm,
                mask_overlay=new_mask_HBT,
                base_mask=mask,
                title='HBT',
                save_path=mask_path / f'{base_name}_overlay_HBT.png'
            )

        # --- Center detection ---
        cX_L = cY_L = cX_R = cY_R = 0
        try:
            if stim_type.startswith('R'):
                cX_L, cY_L = Center_detector(resize_mask_Ca_L)
            elif stim_type.startswith('L'):
                cX_R, cY_R = Center_detector(resize_mask_Ca_R)
        except Exception as e:
            print(f"Center detection failed: {e}")

        # --- Preview (optional) ---
        plt.figure(figsize=(10, 5))
        plt.subplot(121)
        plot_overlay(sum_Ca_total_norm, new_mask_Ca, mask, "Ca")
        plt.subplot(122)
        plot_overlay(sum_HBT_total_norm, new_mask_HBT, mask, "HBT")
        plt.suptitle(f'{mouse}_{exp}_{protocol}_{state}')
        plt.show()
        plt.close()

        # --- Save info with manual review ---
        Q = int(input("Is the ROI correct? [1/0]: "))
        info_coord = {
            'mouse#': mouse, 'exp': exp,  'protocol': protocol, 'state': state, 'stim_type': stim_type,
            'cX_L': cX_L, 'cX_R': cX_R, 'cY_L': cY_L, 'cY_R': cY_R, 'Q': Q
        }
        info_roi_df = pd.DataFrame([info_coord])
        fname_info = info_dir / 'info_roi_coord.csv'

        if not fname_info.exists():
            info_roi_df.to_csv(fname_info, index=False)
        else:
            existing = pd.read_csv(fname_info)
            pd.concat([existing, info_roi_df], axis=0).to_csv(fname_info, index=False)

    except Exception as e:
        print(f"Error processing {img_dir}: {e}")