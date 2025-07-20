# Analysis pipeline for Wide-Field Optical Imaging (WFOI)

This project provides a complete pipeline for processing widefield imaging data combining GCaMP fluorescence (calcium signals) and intrinsic optical signal (IOS) recordings. It is designed for neuroscientific experiments in mice and provides a full pipeline to convert raw imaging data into physiologically meaningful signals. It also includes tools for sensory stimulation analysis, automated and manual ROI detection, response visualization, and statistical evaluation.

The pipeline supports:
- Structured data management across multiple animals, states, and protocols
- Correction of the fluorescence signal for hemoglobin absorption (Beer–Lambert-based methods or regression)
- Stimulus response detection and visualization
- Automated and manual ROI selection using thresholding or Napari-based tools
- Metadata logging and tracking of all steps
- Flexible configuration through YAML files
- Exportable results in .csv, .bmp, .zarr, or .hdf5 formats
- Optional behavior tracking integration for advanced analysis

## Article Reference
This repository is a supplementary resource for the research article where the wide-field optical imaging protocol and animal preparation techniques are described in detail:
"Wide-field cranial window by skull thinning with gel polish coating and protocol for mesoscopic wide-field optical imaging in mice"

📖 Available at: _
> If you use this pipeline for data analysis or signal processing, please cite the original publication.

## Project Structure
The repository includes:
- Example imaging data and extinction/pathlength tables in /data
- Jupyter notebooks for each step of the pipeline
- Modular Python scripts in /src
- YAML configuration templates
- Installation instructions using conda and requirements.txt


# Installation and Project Setup Guide

## 1. Install Anaconda and Set Up the Environment

Follow these steps to set up the project on your local machine.

#### 1. Install Anaconda

Download and install Anaconda from the official website:  
[https://www.anaconda.com/download](https://www.anaconda.com/download)

#### 2. Open Anaconda Prompt

On Windows, search for **Anaconda Prompt** in the Start menu and open it.

#### 3. Create a New Environment

You can name it as you like (e.g., `myenv`):
```bash
conda create -n myenv python=3.12
```

#### 4. Activate the Environment

```bash
conda activate myenv
```

#### 5. Download the Project

You have two options:

##### Option 1: Clone with Git 

(Requires Git to be installed)

```bash
git clone https://github.com/your-username/your-project.git
cd your-project
```

> If Git is not installed, you can download it from: [https://git-scm.com/downloads](https://git-scm.com/downloads)

##### Option 2: Download ZIP Archive

1. Go to the project page on GitHub.
2. Click the green **"Code"** button → **"Download ZIP"**.
3. Extract the archive and open the folder in your terminal.

```bash
cd the-path\where-you-saved-the-unzipped-project
```

#### 6. Install Required Libraries

From the project directory (where `requirements.txt` is located), run

```bash
pip install -r requirements.txt
```

## 2. Prerequisites

For the code to work correctly, input data must follow a specific folder structure:

```
base_path/
└── experiment_day/
    └── mouse_id/
        └── state/
            └── protocol/
                ├── image_1.bmp
                ├── image_2.bmp
                └── ...
```

**Important**:  
Folder names **must not contain underscores (`_`)**, as this may cause errors in the code execution.
These folder names are later used to generate metadata and log files to track processing progress.

### Example

For the directory:

```
D:\Test_exp\Day1\m8\Awake\RHL-90mkA\
```

The following metadata will be generated:

| exp_day | mouse_id | state | protocol     |
|---------|----------|-------|--------------|
| Day1    | m8       | Awake | RHL-90mkA    |

This folder should contain brain cortex images.


## 3. Image Format Requirements

The `protocol/` folder must contain images in formats supported by `cv2.imread`, such as:

- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.ppm`, `.pgm`, `.pbm`, `.tiff`, `.tif`

The code uses `cv2.imread(path_to_img)` to load images — make sure it **does not return `None`** for your files.


## 4. Filename Convention

Each image filename must include a number in the following format:

```
<anything>_<image_number>_<anything>.<ext>
```

When splitting the filename by the underscore (`_`), the image number **must appear in the second position**  
(i.e., index 1 in Python, since indexing starts at 0).


> You can use the example notebook **"Bonus_functions.ipynb"** to:
> - Rename your files in batch  
> - Restructure your folders automatically


---

## Configuration of `config.yaml`

To run the analysis code, you need to configure the `config.yaml` file and set up a folder to save the results.

1. Create a folder where the results will be stored.
2. Copy the `config.yaml` file from the base project into this folder.
3. Open and edit the file using a text editor to match your experiment settings.

Below is a description of each parameter in `config.yaml`:

| Parameter               | Description                                                                                      | Input Type                                               | Default         |
|------------------------|--------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|-----------------|
| `all_exp_dir`          | Base directory where experiment folders are stored                                               | path to the folder                                                     | `D:\Test_exp`          |
| `dir_coeff_abs`        | Path to the ε_Hb(λ) absorption coefficients file                                                  | path to the file                          
| `C:\NL\Ext_hb_coeff.csv` |
| `dir_est_path`         | Path to the X(λ) estimation file                                                                  | path to the file                                        | `C:\NL\Est_pathlenght.csv` |
| `br_y`                 | The position of the bregma along the y-axis in the aligned images is specified as a fraction of the image height, measured from the top edge. For images sized 64×87, if br_y = 0.35, the bregma will be located at a distance of 0.35 × 64 = 22 (rounded to the nearest integer) pixels from the top edge.         | Float (0.0–1.0)                                                             | 0.35            |
| `rotation`             | The angle in degrees by which all images should be rotated before starting the analysis (counterclockwise rotation). The angle can be specified as a negative value.                              | Integer                                                            | 0               |
| `mark_Tvt`             | Is it necessary to label the intersection point of the Superior sagittal sinus and the Inferior cerebral vein?                                              | `true` / `false`                                                   | `false`         |
| `scalingfactor`        | Percentage for image downscaling to speed up analysis. For example, with a value of 25: original image → (348×256); downscaled → (348×0.25 × 256×0.25) = (87×64)                                                                     | Integer                                                            | 25              |
| `img_number`           | Expected number of frames (to discard incomplete recordings). There may be multiple values if the experiment includes recordings with different frame counts.                                     | List (`[8800, 17600]` or `- 8800`)                                 | `[8800,17600]`  |
| `image`                | Frame index used for anatomical annotation                                                        | Integer (0-based)                                                  | 1               |
| `im_n`                 | Frame index (after channel separation) that will be used to demonstrate the processing steps                                            | Integer (0-based)                | 323             |
| `lambdaF_em`           | Fluorescence emission wavelength                                                                  | Integer / leave empty if none                                      | 514             |
| `lambdaF_ex`           | Fluorescence excitation wavelength                                                                | Integer / leave empty if none                                      | 470             |
| `lambdaR_1`            | Scattered light channel 1 wavelength                                                              | Integer / leave empty if none                                      | 530             |
| `lambdaR_2`            | Scattered light channel 2 wavelength                                                              | Integer / leave empty if none                                      | 656             |
| `lambdaF_ex_pattern`   | Frame pattern for fluorescence excitation channel (specifies how frames alternate in the shared folder, e.g.,`[0, 2]` means every second frame starting from frame 0)         | List (`[0, 2]`)                                                    | `[0, 2]`        |
| `lambdaR_1_pattern`    | Frame pattern for scatter channel 1                                                               | List                                                               | `[1, 4]`        |
| `lambdaR_2_pattern`    | Frame pattern for scatter channel 2                                                               | List                                                               | `[3, 4]`        |
| `rate`                 | Acquisition rate for main channel (fluorescence), in fps                                          | Integer                                                            | 20              |
| `rate_noneq`           | Whether to duplicate frames in the light-scattering channels (Required if their acquisition rate is 2 times lower).         | `true` / `false`                                                   | `true`          |
| `Ca_corr_method`       | Correction method for calcium signal: `LSTM` (regression on HbT) or `BLB` (Beer–Lambert law)      | `LSTM` / `BLB`                                                     | `LSTM`          |
| `Xest_em`              | Estimated photon pathlength value for emission (by default, the value for GCaMP6f from Ma et al., 2016 is used).                                                        | Float                                                              | 0.57            |
| `Xest_ex`              | Estimated photon pathlength value for excitation by default, the value for GCaMP6f from Ma et al., 2016 is used).                                                        | Float                                                              | 0.56            |
| `Remove_global`        | Subtract global brain signal (midsagittal sinus excluded)                                         | `true` / `false`                                                   | `true`          |
| `global_smooth_win`    | Window size for smoothing global signal                                                           | Integer                                                            | 20              |
| `save_conv`            | Save converted signal files (e.g., ∆[Ca], ∆[HbT], Raw F/F₀)                                       | `true` / `false`                                                   | `true`          |
| `save_format`          | File format for saving converted data                                                             | `hdf5` / `zarr`                                                    | `zarr`          |
| `mask_format`          | File format for brain masks (used if masks were made with other tools)                            | `bmp`, `png`, `tiff`, `hdf5`                                       | `bmp`           |
| `plot_control`         | Whether to show plots during analysis                                                             | `true` / `false`                                                   | `true`          |
| `exp_to_include`       | Experiments to include (analyze all if left blank)                                                | List                                                               | —               |
| `state_to_include`     | States to include (analyze all if left blank)                                                     | List                                                               | —               |
| `protocols_to_include` | Protocols to include (analyze all if left blank)                                                  | List                                                               | —               |
| `mice_to_include`      | Mice IDs to include (analyze all if left blank)                                                   | List                                                               | —               |

> **Note:**  
> - You can use helper code from the notebook "Bonus_functions.ipynb" to generate lists of available protocols, mice, states, and experiments.
> - The files Est_pathlenght.csv and Ext_hb_coeff.csv used in our experiments are located in the data folder of the project.

# Running the Analysis

## Terminal Setup

1. Open a terminal (e.g., Anaconda Prompt).
2. Activate the environment:
   
   ```bash
   conda activate YourNameForEnvironment
   ```
3. Navigate to the project directory:
   
   ```bash
   cd Your:\Path\To\Folder\WIFIOPIA
   ```
4. Launch Jupyter Notebook:
   
   ```bash
   jupyter notebook
   ```
5. This will open a browser window with the Jupyter interface displaying files from WIFIOPIA.
6. Navigate to the notebooks folder.


## Step 1 - Landmark Annotation
> Manual annotation of anatomical landmarks for image alignment

1. Launch the notebook named “Step1_LandmarkAnnotation.ipynb”.
2. Run the first cell (Select the cell and press Ctrl+Enter) to import libraries and modules.
3. In the second cell, update the path:
   ```python
   config_path=r"Your:\Path_to_your_config\config.yaml"
   ```
   This is the path to the modified config.yaml file, not the default one.
4. Run the cell. Then run the third cell.
   Interactive windows will appear showing images from folders iterated via all_exp_dir in config.yaml.
   The image index is set by `image` parameter in config.yaml.
   You'll be prompted to select anatomical points.

<details> <summary>Image Alignment Instructions</summary>
    
<br>

**First image preview**  
- Place two points at the **top and bottom** of the sagittal suture.  
- The program will draw a line and calculate the rotation angle so the suture becomes **vertical**.
<br>

**Second image preview**  
- The image is now rotated.  
- Mark the **bregma** point.    
<br>

**Third image preview**  
- The image is shifted to center the bregma **horizontally**.  
- The **vertical** position is defined by `br_y` in `config.yaml`.    
<br>

You will be asked:
  
```text
Enter 1 if the wavelengths are mixed up, or 0 otherwise:
```
<br>

This step helps filter out experiments with incorrect illumination patterns due to acquisition issues.
</details>

5. Outputs of this step:
   1. Project folder ‘Main_analysis_yyyy-mm-dd’ with subfolders:
      - Info
      - Demonstr
      - Stim
      - Masks
   2. info_moving.csv file in Info with landmark data.
   3. config.yaml will gain the entry:
      directory_to_save_main_analysis: Your:\Path_to_your_config\Main_analysis_yyyy-mm-dd
   4. Demonstr folder will contain rotated and shifted images like m1_Exp-day1_LE_Awake_moved.tiff


## Step 1-1 - Create Masks using GUI
> GUI-assisted brain boundary delineation to create binary masks

1. Launch “Step1-1_BrainMaskDrawingGUI.ipynb”.
2. Run the first cell.
3. A window will prompt you to select the Demonstr folder.
4. An image will appear; draw a brain outline and save the binary mask.
5. Masks will be saved in the Mask_draw folder with names like:
   mouseid_exp_protocol_state_mask_draw.format

Alternative: You can create masks with external software (e.g., Ilastik, PaintNet).
- Use Demonstr images (already aligned).
- Format: hdf5, png, tiff, or bmp (set in config.yaml).
- Binary mask (0 = non-brain, 255 = brain).
- Naming must follow: mouseid_exp_protocol_state_mask_draw.format
- Move files to Mask_draw/Mask folder.


## Step 2 - Run the Main Analysis
> Conversion of raw intensity traces into physiologically relevant signals

1. Open “Step2_SignalProcessing.ipynb”.
2. Run the first cell to import libraries.
3. In the second cell, update:
   ```python
   config_path=r"Your:\Path_to_your_config\config.yaml"
   ```
4. Run the third cell. If plot_control: true, intermediate steps will be visualized.
   If Remove_global: true, global signal before and after smoothing is also shown.
5. Outputs of this step:
   1. Folder Moved_files with sorted and aligned 3D image arrays in zarr or hdf5 format.
   2. Folder Conv_files with ∆F/F0, corrected ∆F/F0, ∆[HbT], ∆[HHb], ∆[HbO], or ∆I/I0 in zarr or hdf5 format.
   3. processed_log.csv in Info to track analysis progress.

> Already processed files will not be recalculated. Analysis can be resumed anytime.


## Step 3 – Stimulus Response Masking
> Identification of stimulus-evoked response regions in the cortex

**Important**  
The rest of the code is currently compatible only with protocols that include both calcium and HbT signals.

---

### Configuring `config_stim.yaml`

1. Copy `config_stim.yaml` from the base project into the same folder where your edited `config.yaml` is located.
2. Edit `config_stim.yaml` according to your stimulation parameters.

#### Parameters in `config_stim.yaml`

| Parameter | Description | Input | Default |
|----------|-------------|-------|---------|
| `stim_count` | Number of stimulation repeats per session | Integer | 21 |
| `stim_start_frame` | Frame number where first stimulus is applied | Integer | 320 |
| `stim_step` | Frame interval between stimuli | Integer | 400 |
| `stim_window_ca` | Calcium averaging window after stimulus start (frames) | List: `[start, stop]` | `[0, 100]` |
| `stim_window_hbt` | HbT averaging window after stimulus start (frames) | List: `[start, stop]` | `[20, 100]` |
| `subtr_baseline` | Baseline normalization method | One of the following options can be used: `'mean'` – subtracts the mean over the entire stimulation segment, `'prestart_mean'` – subtracts the mean over a defined interval before the stimulation starts, `'no'` – no subtraction is applied. | `'prestart_mean'` |
| `subtr_baseline_interval` | Interval length used for baseline averaging if `prestart_mean` is selected | Integer | 20 |
| `substim_count` | Number of pulses per main stimulus (optional) | Integer / empty | 4 |
| `substim_interval` | Frame interval between pulses (optional) | Integer / empty | 20 |
| `substim_window` | Start and stop frames for the pulses window | List: `[start, stop]` | `[1, 4]` |
| `save_formats` | Output formats for saving response masks | List (any of: `'hdf5'`, `'png'`, `'tiff'`, `'bmp'`, `'overlay'`) | `['bmp', 'overlay']` |

---

### Option A – Automatic ROI Detection

Automatic ROI detection uses pre-defined spatial templates. These regions are defined in the `Create_ROI_dict` function inside `response_detector.py`:

```python
stim_type_dict = {
    'RHL': (br_y-6, br_y+23, br_x-29, br_x-4),
    'LHL': (br_y-6, br_y+23, br_x+4, br_x+29),
    'RE':  (br_y+17, br_y+38, br_x-30, br_x-4),
    'LE':  (br_y+17, br_y+38, br_x+4, br_x+30),
    'RV':  (br_y+7, br_y+28, br_x-40, br_x-20),
    'LV':  (br_y+7, br_y+28, br_x+20, br_x+40),
}
```

> Note: The protocol name must contain one of the above keys *before* the dash.  
> For example: `RHL-90mkA`, `LHL-test`, etc.

**Steps:**

1. Run `Step3a_ResponseDetection.ipynb`
2. Run the first cell to import libraries.
3. In the second cell, set:
```python
config_path = r"Your:\Path\to\config.yaml"
config_stim_path = r"Your:\Path\to\config_stim.yaml"
```
4. Run the third cell. You will see overlays of detected masks over mean Ca/HbT responses. Confirm whether the ROI is correct (1 = yes, 0 = no).
5. **Results:**
   - `Mask_stim_<stimtype>` directories will be created inside `Main_analysis_yyyy-mm-dd`, each containing 4 masks: left/right for Ca and HbT.
   - `Info/info_roi_coord.csv` will contain ROI center coordinates and quality labels.

---

### Option B – Manual ROI Drawing with Napari

1. Run `Step3b_ResponseDetection.ipynb`
2. Run the first cell to import libraries.
3. In the second cell, set:
```python
config_path = r"Your:\Path\to\config.yaml"
config_stim_path = r"Your:\Path\to\config_stim.yaml"
```
4. Run the third cell. A list of available experiments with their indices will be displayed.
The user will be prompted to enter the index of the experiment for which the response mask should be drawn.
5. Run the cell under the heading “Draw Mask”. A napari window will appear with a 3D array of mean calcium responses. Use the polygon tool to draw the ROI on the most responsive frame. Close napari once the mask is visible.
6. Run the next cell to save the mask.
7. Optionally, run two cells under “Visualize Response” to plot and save the average response trace.
8. Repeat step 4 for each new experiment.

**Results:**
- A folder `ResponseMasks_Drawn` will be created in `Main_analysis_yyyy-mm-dd`, storing one mask per experiment (contralateral only).
- A file `Info/roi_mask_log.csv` will log whether a mask was drawn (1 = yes, 0 = no).
- If step 7 was executed, the ROI response plot will be saved in the `Stim` folder.


## Step 4 – Creating a Summary Table of ROI-Averaged Values per Stimulus
>Tabulation of extracted responses for quantitative analysis

Depending on how the response masks were created, follow the appropriate path.

### Option A – If Masks Were Created Automatically

1. Launch the notebook `Step4a_ResponseTableGeneration.ipynb`
2. Run the first cell to import libraries and modules.
3. In the second cell, update the paths and run:

```python
config_path = r"Your:\Path_to_your_config\config.yaml"
config_stim_path = r"Your:\Path_to_your_config\config_stim.yaml"
```

4. Run the third cell.
5. The following CSV files will be saved in the `Stim` folder:
   - `Ca_stim_info.csv`
   - `Ca_wocorr_info.csv`
   - `dcHbO_stim_info.csv`
   - `dcHHb_stim_info.csv`
   - `dcHbR_stim_info.csv`

Each file will contain time-series signal values averaged over the automatically detected ROI, e.g.:

| signal | stim_type | hemisphere | state | exp          | protocol       | mouse | stim# | 0 s       | 0.05 s    | ... |
|--------|-----------|------------|-------|--------------|----------------|-------|-------|---------|---------|-----|
| Ca     | V         | L          | Awake | Stim-pulseexp | RV-pulseflow09 | m3923 | st1   | 0.00359 | 0.00365 | ... |
| Ca     | V         | L          | Awake | Stim-pulseexp | RV-pulseflow09 | m3923 | st2   |         | ...     | ...   |

Each entry includes data for both hemispheres, but only the **contralateral** response is used in visualizations and analysis.

---

### Option B – If Masks Were Drawn in Napari

1. Launch the notebook `Step4b_ResponseTableGeneration.ipynb`
2. Run the first cell to import libraries and modules.
3. In the second cell, update the paths and run:

```python
config_path = r"Your:\Path_to_your_config\config.yaml"
config_stim_path = r"Your:\Path_to_your_config\config_stim.yaml"
```

4. Run the third cell. It will define filters for low-pass filtering the Hb signal. You can modify the cutoff frequency with the `fcutoff = 1.5` parameter.
5. Run the fourth cell. To **disable** filtering, simply comment out the following lines:

```python
# signal_dhbt = signal.filtfilt(filter_kernel, 1, signal_dhbt, axis = 0)
# signal_dhbo = signal.filtfilt(filter_kernel, 1, signal_dhbo, axis = 0)
# signal_dhbr = signal.filtfilt(filter_kernel, 1, signal_dhbr, axis = 0)
```

6. Resulting CSV files saved to the `Stim` folder:
   - `Ca_stim_info.csv`
   - `Ca_wocorr_info.csv`
   - `dcHbO_stim_info.csv`
   - `dcHHb_stim_info.csv`
   - `dcHbR_stim_info.csv`

These files will include time-averaged ROI values from manually drawn masks in napari.

Only one hemisphere’s ROI is saved, **expected to be contralateral** to the stimulated side.

---

## Step 5 – Visualization and Analysis of Stimulus Responses
> Statistical analysis and visualization of brain responses

You can analyze the files created above (`*_stim_info.csv`) in any suitable program. A Jupyter notebook `Step5_ResponseAnalysis.ipynb` is included to assist with visualization and analysis using Python.

> Note: This step requires basic familiarity with Python/Jupyter Notebook or time to customize the plotting for your needs.

This step also requires a file with behavioral annotations, such as `beh_example.xlsx` from the `/data/` folder.

Behavior codes:
- `s` = sitting
- `r` = running

These annotations are optional. 
