#import libraries
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import scipy
import pathlib
import h5py
import datetime
from pathlib import Path

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


def merge_behavior_with_stim_data(
    stim_df: pd.DataFrame,
    behavior_file: Path,
    behavior_sheet: str = None
) -> pd.DataFrame:
    """
    Merges stimulus-aligned DataFrame with behavioral annotations from an Excel file.
    If behavior is missing and state != 'Awake', the state value is assigned to behavior.

    Parameters
    ----------
    stim_df : pd.DataFrame
        DataFrame with stimulus data. Must include columns:
        'mouse', 'exp', 'protocol', 'state', and 'stim#'.
    
    behavior_file : Path
        Path to the Excel file containing behavioral annotations.
    
    behavior_sheet : str, optional
        Name of the Excel sheet to read. Defaults to the first sheet if None.

    Returns
    -------
    pd.DataFrame
        The merged DataFrame with an added 'behavior' column.
    """
    if not isinstance(behavior_file, Path):
        raise TypeError("behavior_file must be a pathlib.Path object")
    
    if not behavior_file.exists():
        raise FileNotFoundError(f"Behavior file not found: {behavior_file}")

    # Load behavioral annotations
    behavior_df = pd.read_excel(behavior_file, sheet_name=behavior_sheet)

    # Convert behavior table from wide to long format
    behavior_long = behavior_df.melt(
        id_vars=["mouse", "exp", "protocol", "state"],
        var_name="stim_id",
        value_name="behavior"
    )
    behavior_long["stim#"] = "st" + behavior_long["stim_id"].astype(str)
    behavior_long.drop(columns=["stim_id"], inplace=True)

    # If stim_df contains non-Awake states, force state values into behavior_long
    if not (stim_df["state"] == "Awake").all():
        unique_states = stim_df[["mouse", "exp", "protocol", "state"]].drop_duplicates()
        behavior_long = pd.merge(
            behavior_long.drop(columns=["state"]),
            unique_states,
            on=["mouse", "exp", "protocol"],
            how="left"
        )

    # Merge behavior with stimulus data
    merged_df = pd.merge(
        stim_df,
        behavior_long,
        on=["mouse", "exp", "protocol", "state", "stim#"],
        how="left"
    )

    # If behavior is missing and state is not 'Awake' — fill with state name
    condition = merged_df["behavior"].isna() & (merged_df["state"] != "Awake")
    merged_df.loc[condition, "behavior"] = merged_df.loc[condition, "state"]

    return merged_df

def answer_side(df: pd.DataFrame) -> None:
    """
    Assign ipsilateral/contralateral answer side based on protocol and hemisphere.
    """
    conditions = [
        (df['hemisphere'] == 'L') & df['protocol'].str.startswith('L'),
        (df['hemisphere'] == 'L') & df['protocol'].str.startswith('R'),
        (df['hemisphere'] == 'R') & df['protocol'].str.startswith('R'),
        (df['hemisphere'] == 'R') & df['protocol'].str.startswith('L')
    ]
    choices = ['I_L', 'C_L', 'I_R', 'C_R']
    df['answ_side'] = np.select(conditions, choices, default='Unknown')

def group_signal(
    df: pd.DataFrame, stim_type: str, state: str = 'Awake', answer_side_start: str = 'C'
) -> pd.DataFrame:
    """
    Filters and groups signal data for specific stim and state.
    """
    df_filtered = df[
        (df['answ_side'].str.startswith(answer_side_start)) &
        (df['stim_type'] == stim_type) &
        (df['state'] == state)
    ].reset_index(drop=True)

    df_filtered['sstim_id'] = df_filtered.index + 1
    return df_filtered.groupby(['stim_type', 'mouse', 'sstim_id']).mean(numeric_only=True).copy()

def data_for_plot(df: pd.DataFrame, stype: str = 'HbT') -> pd.DataFrame:
    """
    Prepares data in long format for plotting with detrending and scaling.
    """
    time = df.T.iloc[:, 0].index.astype(np.float32)
    all_data = []

    for (stim_type, mouse, sstim_id) in df.index:
        signal = df.T.loc[:, (stim_type, mouse, sstim_id)].values
        signal = scipy.signal.detrend(signal, type='constant')
        signal *= 1e6 if 'Hb' in stype else 100
        signal -= np.mean(signal[:5])

        all_data.append(pd.DataFrame({
            'Time': time,
            'Signal': signal,
            'S_type': stype,
            'stim_type': stim_type,
            'mouse': mouse,
            'sstim_id': sstim_id
        }))

    return pd.concat(all_data, ignore_index=True)

def align_yaxis(ax1, v1, ax2, v2):
    """Align v1 in ax1 to match v2 in ax2 visually."""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2, (y1 - y2) / 2, v2)
    adjust_yaxis(ax1, (y2 - y1) / 2, v1)

def adjust_yaxis(ax, ydiff, v):
    """Adjust y-limits to maintain a fixed data point location."""
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

def plot_behavior_or_simple(
    ca_df,
    cawocorr_df,
    hbt_df,
    stim_type,
    state,
    dchbo_df=None,
    dchbr_df=None,
    side='C',
    lang='ru',
    save_path: Path = None
):
    try:
        has_behavior = 'behavior' in ca_df.columns and state == 'Awake'

        if lang == 'ru':
            titles = {'s': 'Без локомоции', '!s': 'С локомоцией'}
            y_ca = 'ΔF/F₀ Ca²⁺, % (GCaMP6f)'
            y_hb = 'Δс(Hb), мкМ'
            stim_label = 'Стимул'
            main_title = 'Общий график'
            x_label = 'Время, с'
        else:
            titles = {'s': 'No locomotion', '!s': 'Locomotion'}
            y_ca = 'ΔF/F₀ Ca²⁺, % (GCaMP6f)'
            y_hb = 'Δ[Hb], μM'
            stim_label = 'Stimulus'
            main_title = 'Combined plot'
            x_label = 'Time, s'

        if has_behavior:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10), sharex=True)
            axes2 = []

            def safe_filter(df, key):
                if df is None or df.empty or 'behavior' not in df.columns:
                    return df
                return df[df['behavior'] == key] if key == 's' else df[df['behavior'] != 's']

            for idx, (key, subtitle) in enumerate(titles.items()):
                ax = axes[idx]
                ax2 = ax.twinx()
                axes2.append(ax2)

                ca_sel = safe_filter(ca_df, key)
                cawocorr_sel = safe_filter(cawocorr_df, key)
                hbt_sel = safe_filter(hbt_df, key)
                dchbo_sel = safe_filter(dchbo_df, key) if dchbo_df is not None else None
                dchbr_sel = safe_filter(dchbr_df, key) if dchbr_df is not None else None

                _plot_single_panel(
                    ca_sel, cawocorr_sel, hbt_sel,
                    stim_type, state, ax, ax2,
                    title=f"{subtitle} — {stim_type}, {state}",
                    dchbo_df=dchbo_sel,
                    dchbr_df=dchbr_sel,
                    y_ca=y_ca, y_hb=y_hb,
                    stim_label=stim_label,
                    x_label=x_label,
                    collect_legend=False
                )

            # Общая легенда снизу
            handles, labels = [], []
            for ax, ax2 in zip(axes, axes2):
                h1, l1 = ax.get_legend_handles_labels()
                h2, l2 = ax2.get_legend_handles_labels()
                handles += h1 + h2
                labels += l1 + l2

            combined = dict(zip(labels, handles))
            fig.legend(
                combined.values(), combined.keys(),
                loc='lower center',
                bbox_to_anchor=(0.5, 0.01),
                ncol=len(combined),
                frameon=False,
                fontsize='medium'
            )

            plt.tight_layout(rect=[0, 0.08, 1, 1])
            if save_path:
                save_file = save_path / f"{stim_type}_{state}_split_behavior.png"
                plt.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.show()

        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax2 = ax.twinx()

            _plot_single_panel(
                ca_df, cawocorr_df, hbt_df,
                stim_type, state, ax, ax2,
                title=f"{main_title} — {stim_type}, {state}",
                dchbo_df=dchbo_df,
                dchbr_df=dchbr_df,
                y_ca=y_ca, y_hb=y_hb,
                stim_label=stim_label,
                x_label=x_label,
                collect_legend=True
            )

            plt.tight_layout(rect=[0, 0.14, 1, 1])
            if save_path:
                save_file = save_path / f"{stim_type}_{state}_plot.png"
                plt.savefig(save_file, dpi=300, bbox_inches='tight')
            plt.show()

    except Exception as e:
        print(f"[ERROR] Failed to plot for {stim_type}, {state}: {e}")
        plt.close('all')


def _plot_single_panel(
    ca_df, cawocorr_df, hbt_df,
    stim_type, state, ax, ax2, title,
    dchbo_df=None, dchbr_df=None,
    y_ca='ΔF/F₀ Ca²⁺, %', y_hb='Δс(Hb), мкМ',
    stim_label='Стимул',
    x_label='Time, s',
    collect_legend=True
):
    if ca_df.empty and cawocorr_df.empty and hbt_df.empty and \
       (dchbo_df is None or dchbo_df.empty) and (dchbr_df is None or dchbr_df.empty):
        print(f"[WARNING] All inputs empty for {stim_type}, {state} — skipping panel.")
        plt.close(ax.figure)
        return

    ca_plot = data_for_plot(group_signal(ca_df, stim_type, state, 'C'), stype='Ca')
    cawocorr_plot = data_for_plot(group_signal(cawocorr_df, stim_type, state, 'C'), stype='Cawocorr')
    hbt_plot = data_for_plot(group_signal(hbt_df, stim_type, state, 'C'), stype='HbT')

    sns.lineplot(data=ca_plot, x="Time", y="Signal", errorbar='se', color='g', ax=ax, label='Ca', legend = False)
    sns.lineplot(data=cawocorr_plot, x="Time", y="Signal", errorbar='se', color='k', ax=ax, label='Cawocorr', legend = False)
    sns.lineplot(data=hbt_plot, x="Time", y="Signal", errorbar='se', color='m', ax=ax2, label='HbT', legend = False)

    if dchbo_df is not None and not dchbo_df.empty:
        dchbo_plot = data_for_plot(group_signal(dchbo_df, stim_type, state, 'C'), stype='HbO')
        sns.lineplot(data=dchbo_plot, x="Time", y="Signal", errorbar='se', color='r', ax=ax2, label='HbO', legend = False)

    if dchbr_df is not None and not dchbr_df.empty:
        dchbr_plot = data_for_plot(group_signal(dchbr_df, stim_type, state, 'C'), stype='HbR')
        sns.lineplot(data=dchbr_plot, x="Time", y="Signal", errorbar='se', color='b', ax=ax2, label='HbR', legend = False)

    for stim_time in [1, 2, 3, 4]:
        ax.axvspan(stim_time, stim_time + 0.2, facecolor='gray', alpha=0.2, label=stim_label if stim_time == 1 else None)

    ax.axhline(y=0.0, color='black', linestyle='--')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_ca, color='green')
    ax2.set_ylabel(y_hb, color='red')

    ax.tick_params(axis='y', colors='green')
    ax2.spines['right'].set_color('red')
    ax2.tick_params(axis='y', colors='red')
    ax2.yaxis.label.set_color('red')
    ax2.yaxis.set_label_coords(1.08, 0.5)

    align_yaxis(ax, 0, ax2, 0)
    ax.set_title(title)

    if collect_legend:
        handles1, labels1 = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        combined = dict(zip(labels1 + labels2, handles1 + handles2))
        ax.figure.legend(
            combined.values(), combined.keys(),
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(combined),
            frameon=False,
            fontsize='medium'
        )