# Import
import os
import yaml
import h5py
import zarr
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Union, Tuple, Callable, Generator, Sequence, Iterable, Optional

# Common functions used in multiple modules

def load_config(path: str | Path = "../config.yaml") -> dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        path (str or Path, optional): Path to the YAML config file. Defaults to "../config.yaml".

    Returns:
        dict[str, Any]: Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If the YAML content is invalid.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)

def def_folders(directory: Union[str, Path]) -> Tuple[Path, Path, Path, Path, Path]:
    """
    Load an existing project folder structure.

    Args:
        directory (str or Path): Path to the main analysis directory.

    Returns:
        tuple[Path, Path, Path, Path, Path]: Paths to the following subdirectories:
            - res_dir: Main analysis directory
            - info_dir: 'Info' subdirectory
            - demonstr_dir: 'Demonstr' subdirectory
            - result_directory_stim: 'Stim' subdirectory
            - result_directory_masks: 'Masks' subdirectory
    """
    directory = Path(directory)

    res_dir = directory
    info_dir = directory / 'Info'
    demonstr_dir = directory / 'Demonstr'
    result_directory_stim = directory / 'Stim'
    result_directory_masks = res_dir / 'Masks'

    return (res_dir, info_dir, demonstr_dir,
            result_directory_stim, result_directory_masks)


def create_folders(directory_to_save: Union[str, Path]) -> Tuple[Path, Path, Path, Path, Path]:
    """
    Create a new timestamped main analysis folder with a standard subfolder structure.

    Args:
        directory_to_save (str or Path): Base path where the new project folder will be created.

    Returns:
        tuple[Path, Path, Path, Path, Path]: Paths to the following subdirectories:
            - res_dir: Created main analysis directory (with timestamp)
            - info_dir: 'Info' subdirectory
            - demonstr_dir: 'Demonstr' subdirectory
            - result_directory_stim: 'Stim' subdirectory
            - result_directory_masks: 'Masks' subdirectory
    """
    directory_to_save = Path(directory_to_save)
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    # Main analysis folder
    res_dir = directory_to_save / f'Main_analysis_{today}'
    res_dir.mkdir(exist_ok=True)

    # Subfolders
    subfolders = ['Info', 'Demonstr', 'Stim', 'Masks']
    for sub in subfolders:
        (res_dir / sub).mkdir(exist_ok=True)

    # Paths to return
    info_dir = res_dir / 'Info'
    demonstr_dir = res_dir / 'Demonstr'
    result_directory_stim = res_dir / 'Stim'
    result_directory_masks = res_dir / 'Masks'

    return (res_dir, info_dir, demonstr_dir,
            result_directory_stim, result_directory_masks)


def setup_or_load_project(
    config_path: str | Path,
    create_func: Callable[[Path], tuple],
    def_func: Callable[[Path], tuple]
) -> tuple:
    """
    Initialize or load project folders based on a YAML config file.

    Args:
        config_path (str or Path): Path to the configuration YAML file.
        create_func (Callable[[Path], tuple]): Function to create new project folders. 
            Should take the base directory as argument and return a tuple of Paths.
        def_func (Callable[[Path], tuple]): Function to load existing project folders.
            Should take the saved main analysis directory and return a tuple of Paths.

    Returns:
        tuple: Result of `create_func(...)` or `def_func(...)`, depending on the config contents.
    """
    config_path = Path(config_path)
    config = load_config(config_path)

    directory_to_save_analysis = config_path.parent

    if "directory_to_save_main_analysis" not in config:
        result = create_func(directory_to_save_analysis)

        config["directory_to_save_main_analysis"] = str(result[0])
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        return result
    else:
        directory = Path(config["directory_to_save_main_analysis"])
        return def_func(directory)


def iterate_experiments(
    base_dir: str | Path,
    skip_mouse: str = "m#",
    verbose: bool = True
) -> Generator[tuple[Path, str, str, str, str], None, None]:
    """
    Generator that recursively iterates through experimental folders:
    experiment_day → mouse_dir → state_folder → protocol_folder = img_dir

    Args:
        base_dir (str or Path): Root directory containing experiment days.
        skip_mouse (str, optional): Folder name to skip (e.g., 'm#'). Defaults to "m#".
        verbose (bool, optional): If True, prints progress info. Defaults to True.

    Yields:
        tuple[Path, str, str, str, str]: For each protocol folder found:
            - img_dir (Path): Path to the image directory (deepest folder)
            - exp (str): Experiment day folder name
            - mouse (str): Mouse folder name
            - state (str): State folder name (e.g., 'Baseline', 'Stim')
            - protocol (str): Protocol name inside the state folder
    """
    base_path = Path(base_dir)

    for exp_day in os.listdir(base_path):
        day_path = base_path / exp_day
        if not day_path.is_dir():
            continue

        if verbose:
            print(f"Processing experiment day: {exp_day}")

        for mouse_dir in os.listdir(day_path):
            if mouse_dir == skip_mouse:
                continue

            mouse_path = day_path / mouse_dir
            if not mouse_path.is_dir():
                continue

            for state_folder in os.listdir(mouse_path):
                state_path = mouse_path / state_folder
                if not state_path.is_dir():
                    continue

                for protocol_folder in os.listdir(state_path):
                    img_dir = state_path / protocol_folder
                    if not img_dir.is_dir():
                        continue

                    exp = exp_day
                    mouse = mouse_dir
                    state = state_folder
                    protocol = protocol_folder

                    if verbose:
                        print(f"Found: exp={exp}, mouse={mouse}, state={state}, protocol={protocol}")

                    yield img_dir, exp, mouse, state, protocol

def load_or_create_csv(path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    """
    Load a CSV file if it exists, otherwise create it with the specified columns.

    Args:
        path (str or Path): Path to the CSV file.
        columns (Sequence[str]): List of column names to use if the file does not exist.

    Returns:
        pd.DataFrame: Loaded or newly created DataFrame.
    """
    if os.path.isfile(path):
        return pd.read_csv(path)
    else:
        df = pd.DataFrame(columns=columns)
        df.to_csv(path, index=False)
        return df

def sortByNumberofFrame(p: str | Path) -> int:
    """
    Extracts a numeric frame index from a filename or Path object.

    Expected format: something like 'prefix_123_suffix.ext' → will extract 123.

    Args:
        p (str or Path): Filename or Path containing an underscore and a number.

    Returns:
        int: The number extracted from the second part of the filename after splitting by '_'.

    Raises:
        ValueError: If the filename format does not contain a number in the expected place.
    """
    if isinstance(p, Path):
        p = p.name
    return int(p.split('_')[1])

def load_conv_results(
    save_path: Path,
    filename: str,
    save_format: str,
    return_wocorr: bool = False
) -> Union[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Load converted signal data (Ca_corr, dcHBO_gauss, dcHBR_gauss, dcHBT_gauss, optionally Ca_wocorr) from a file (HDF5 or Zarr format).

    Args:
        save_path (Path): Directory containing the saved conversion results.
        filename (str): Base name of the file (without extension).
        save_format (str): File format to load from. Must be either 'hdf5' or 'zarr' (case-insensitive).
        return_wocorr (bool, optional): If True, also return the Ca_wocorr array. Defaults to False.

    Returns:
        tuple: Always returns:
            - ca_corr (np.ndarray): Corrected calcium signal
            - dcHBO_gauss (np.ndarray): Smoothed oxyhemoglobin signal
            - dcHBR_gauss (np.ndarray): Smoothed deoxyhemoglobin signal
            - dcHBT_gauss (np.ndarray): Smoothed total hemoglobin signal  
          Additionally, if `return_wocorr` is True:
            - ca_wocorr (np.ndarray): Uncorrected calcium signal

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is unsupported.
        KeyError: If 'Ca_wocorr' is requested but not present in the file.
    """
    save_format = save_format.lower()

    if save_format == "hdf5":
        file_path = save_path / f"{filename}.hdf5"
        if not file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        
        with h5py.File(file_path, 'r') as f:
            ca_corr = np.array(f['Ca_corr'])
            dcHBO_gauss = np.array(f['dcHBO_gauss'])
            dcHBR_gauss = np.array(f['dcHBR_gauss'])
            dcHBT_gauss = np.array(f['dcHBT_gauss'])

            if return_wocorr:
                if 'Ca_wocorr' in f:
                    ca_wocorr = np.array(f['Ca_wocorr'])
                else:
                    raise KeyError(f"'Ca_wocorr' not found in {file_path}")

    elif save_format == "zarr":
        file_path = save_path / f"{filename}.zarr"
        if not file_path.exists():
            raise FileNotFoundError(f"Zarr file not found: {file_path}")

        root = zarr.open(str(file_path), mode='r')
        ca_corr = root['Ca_corr'][()]
        dcHBO_gauss = root['dcHBO_gauss'][()]
        dcHBR_gauss = root['dcHBR_gauss'][()]
        dcHBT_gauss = root['dcHBT_gauss'][()]

        if return_wocorr:
            if 'Ca_wocorr' in root:
                ca_wocorr = root['Ca_wocorr'][()]
            else:
                raise KeyError(f"'Ca_wocorr' not found in {file_path}")

    else:
        raise ValueError(f"Unsupported format: {save_format}")

    if return_wocorr:
        return ca_corr, dcHBO_gauss, dcHBR_gauss, dcHBT_gauss, ca_wocorr
    else:
        return ca_corr, dcHBO_gauss, dcHBR_gauss, dcHBT_gauss

def iterate_experiments_filt(
    base_dir: Union[str, Path],
    skip_mouse: str = "m#",
    verbose: bool = True,
    mice_to_include: Optional[Iterable[str]] = None,
    protocols_to_include: Optional[Iterable[str]] = None,
    state_to_include: Optional[Iterable[str]] = None,
    exp_to_include: Optional[Iterable[str]] = None
) -> Generator[tuple[Path, str, str, str, str], None, None]:
    """
    Recursively iterate through experimental folders (with optional filters):
    experiment_day → mouse → state → protocol → image directory.

    Args:
        base_dir (str or Path): Root directory containing experiment days.
        skip_mouse (str, optional): Mouse folder name to skip. Defaults to "m#".
        verbose (bool, optional): If True, prints progress info. Defaults to True.
        mice_to_include (Iterable[str] or None, optional): List of mouse folder names to include.
        protocols_to_include (Iterable[str] or None, optional): List of protocol folder names to include.
        state_to_include (Iterable[str] or None, optional): List of state folder names to include.
        exp_to_include (Iterable[str] or None, optional): List of experiment day folder names to include.

    Yields:
        tuple[Path, str, str, str, str]: A tuple containing:
            - img_dir (Path): Path to the protocol/image directory.
            - exp (str): Experiment day folder name.
            - mouse (str): Mouse folder name.
            - state (str): State folder name.
            - protocol (str): Protocol folder name.
    """
    base_path = Path(base_dir)

    for exp_day in os.listdir(base_path):
        if exp_to_include is not None and exp_day not in exp_to_include:
            continue

        day_path = base_path / exp_day
        if not day_path.is_dir():
            continue

        if verbose:
            print(f"Processing experiment day: {exp_day}")

        for mouse_dir in os.listdir(day_path):
            if mouse_dir == skip_mouse:
                continue
            if mice_to_include is not None and mouse_dir not in mice_to_include:
                continue

            mouse_path = day_path / mouse_dir
            if not mouse_path.is_dir():
                continue

            for state_folder in os.listdir(mouse_path):
                if state_to_include is not None and state_folder not in state_to_include:
                    continue

                state_path = mouse_path / state_folder
                if not state_path.is_dir():
                    continue

                for protocol_folder in os.listdir(state_path):
                    if protocols_to_include is not None and protocol_folder not in protocols_to_include:
                        continue

                    img_dir = state_path / protocol_folder
                    if not img_dir.is_dir():
                        continue

                    if verbose:
                        print(f"Found: exp={exp_day}, mouse={mouse_dir}, state={state_folder}, protocol={protocol_folder}")

                    yield img_dir, exp_day, mouse_dir, state_folder, protocol_folder