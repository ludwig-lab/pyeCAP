import datetime as dt
import os
from glob import glob

import numpy as np
import openpyxl
import pandas as pd
import warnings
import xlrd

from pathlib import Path
import pandas as pd
from collections import defaultdict

# TDT_SAMPLE_DELAY = 21  # TDT introduces a delay between stim and recording. This is the sample size of that delay
# PULSE_WIDTH_DELAY = .20e-3


def electrode_distances(log_path):
    wb = openpyxl.load_workbook(log_path)
    ws = wb.active
    num_header_cells = 1
    num_recordings = 0
    cols = tuple(ws.columns)
    target_col = "E"
    target_col_num = ord(target_col.lower()) - 96
    max_length = len(cols[target_col_num - 1])
    distances = []

    for row in range(1, max_length + 1):
        if ws.cell(row=row, column=target_col_num).value is None:
            num_recordings = row - 2
            break
    if ws.cell(row=max_length, column=target_col_num).value is not None:
        num_recordings = max_length - 1

    for i in range(num_recordings):
        distances.append(ws.cell(i + num_header_cells + 1, target_col_num).value)

    return distances


def check_make_dir(input_dir):
    if not os.path.isdir(input_dir):
        os.makedirs(input_dir)


def get_exp(input_dir):
    pass


def create_experimental_log(file_path):
    # First, we'll get all tdt tanks and store them in tdt_tanks
    tdt_tanks = []

    directories = [x[0] for x in os.walk(file_path)]

    # TDT tanks are a directory with a number of files. In particular any stream data is contained in a *.tev file.
    # Let's use that as in indicator of a complete tdt tank.
    tev_search = "/*.tev"
    for d in directories:
        tev_files = glob(d + tev_search)
        # Append
        if len(tev_files) > 0:
            tdt_tanks.append(d)

    wb = openpyxl.Workbook()
    ws = wb.active

    # We'll create some headers in the log
    headers = ["Location", "Stimulation Contact", "Stimulation Description", "Experimental Condition",
               "Distances: Recording to Stimulating Electrode (cm)"]
    for idx, val in enumerate(headers):
        _cell = ws.cell(row=1, column=idx + 1, value=val)
        _cell.font = openpyxl.styles.Font(bold=True)

    for idx, val in enumerate(tdt_tanks):
        ws.cell(row=idx + 2, column=1, value=val)

    # This next bit of code "autosizes" the cell widths
    dims = {}
    for row in ws.rows:
        for cell in row:
            if cell.value:
                dims[cell.column_letter] = max((dims.get(cell.column_letter, 0), len(str(cell.value))))
    for col, value in dims.items():
        ws.column_dimensions[col].width = value

    experiment_log_path = file_path + "/Experimental Log.xlsx"
    wb.save(filename=experiment_log_path)
    return experiment_log_path


def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum % 1) - dt.timedelta(days=366)
    return day + dayfrac


def configure_local_dask(num_workers=8, threads_per_worker=1, cache_gb=8, blas_threads=1):
    """
        Configure a local distributed Dask cluster optimized for
        NumPy/SciPy heavy workloads (e.g., filtering pipelines).
        """

    import os
    from dask.distributed import Client, LocalCluster

    # Prevent BLAS oversubscription (VERY important)
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(blas_threads)

    # Create distributed cluster
    cluster = LocalCluster(
        n_workers=num_workers,
        threads_per_worker=threads_per_worker,  # 1 is best for numpy-heavy workloads
        processes=True,
        memory_limit=f"{int(cache_gb / num_workers)}GB",  # per worker
    )

    client = Client(cluster)

    return client, cluster

def shutdown_dask(client=None, cluster=None):
    # close in the right order, tolerate "No scheduler connected"
    if client is not None:
        try:
            client.shutdown()
        except Exception:
            pass
        try:
            client.close()
        except Exception:
            pass
    if cluster is not None:
        try:
            cluster.close()
        except Exception:
            pass

def resolve_ephys_paths_and_distances_from_excel(
        excel_path,
        location_col="Location",
        distance_col="Distances: Recording to Stimulating Electrode (cm)",
        prefer="latest",
        raise_on_duplicates=True,
        verbose=True,
):
    """
    Resolve full recording-folder paths and extract distances from an experimental log Excel file.

    Assumed folder structure:
        <date_folder>/
            Notes/<excel_file>
            Data/<experiment_folder>/<recording_folder>

    Parameters
    ----------
    excel_path : str or Path
        Path to the experimental log Excel file.
    location_col : str
        Name of the column containing folder names to resolve.
    distance_col : str
        Name of the column containing recording distances.
    prefer : {"latest", "first"}
        How to resolve duplicates when multiple folders share the same recording folder name.
        "latest" selects the lexicographically last path after sorting, which matches your
        timestamped recording names well in typical cases.
    raise_on_duplicates : bool
        If True, raise an error whenever duplicates are found.
    verbose : bool
        If True, print status messages and duplicate warnings.

    Returns
    -------
    df : pandas.DataFrame
        Original dataframe with the distance column removed and a new 'full_path' column added.
    paths : list[Path]
        Resolved paths in dataframe order, excluding missing rows.
    distances : pandas.Series
        Distance column popped from the dataframe, preserved in original row order.
    """

    excel_path = Path(excel_path)
    base_dir = excel_path.parent.parent / "Data"

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    if not base_dir.exists():
        raise FileNotFoundError(f"Derived Data directory not found: {base_dir}")

    if prefer not in {"latest", "first"}:
        raise ValueError("prefer must be 'latest' or 'first'")

    if verbose:
        print(f"Using Excel log: {excel_path}")
        print(f"Using base_dir : {base_dir}")

    df = pd.read_excel(excel_path)

    if location_col not in df.columns:
        raise ValueError(f"Excel must contain a '{location_col}' column")
    if distance_col not in df.columns:
        raise ValueError(f"Excel must contain a '{distance_col}' column")

    distances = df.pop(distance_col).dropna().to_list()

    if verbose:
        print("Scanning directory tree...")

    dir_map = defaultdict(list)
    for p in base_dir.rglob("*"):
        if p.is_dir():
            dir_map[p.name].append(p)

    resolved_paths = []
    missing = []
    duplicate_report = {}

    for loc in df[location_col]:
        matches = dir_map.get(loc, [])

        if len(matches) == 0:
            missing.append(loc)
            resolved_paths.append(None)
            continue

        if len(matches) == 1:
            resolved_paths.append(matches[0])
            continue

        sorted_matches = sorted(matches, key=lambda x: str(x))

        if prefer == "latest":
            chosen = sorted_matches[-1]
        else:
            chosen = sorted_matches[0]

        duplicate_report[loc] = {
            "chosen": chosen,
            "matches": sorted_matches,
        }
        resolved_paths.append(chosen)

    df["full_path"] = resolved_paths

    if duplicate_report:
        msg_lines = [
            f"Found {len(duplicate_report)} duplicate recording folder name(s)."
        ]
        for loc, info in duplicate_report.items():
            msg_lines.append(f"\n{loc}")
            msg_lines.append(f"  selected -> {info['chosen']}")
            for m in info["matches"]:
                msg_lines.append(f"  candidate -> {m}")
        msg = "\n".join(msg_lines)

        if raise_on_duplicates:
            raise ValueError(msg)
        elif verbose:
            print("\nDuplicate folder names detected:")
            print(msg)

    if missing and verbose:
        print(f"\nMissing {len(missing)} folder(s):")
        for loc in missing[:20]:
            print(f"  {loc}")
        if len(missing) > 20:
            print("  ...")

    if verbose:
        n_resolved = df["full_path"].notna().sum()
        print(f"\nResolved {n_resolved} / {len(df)} paths")

    paths = [p for p in resolved_paths if p is not None]
    return df, paths, distances