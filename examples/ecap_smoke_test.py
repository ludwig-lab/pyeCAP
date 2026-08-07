# --- Imports
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pyeCAP.ephys import Ephys
from pyeCAP.stim import Stim
from pyeCAP.ecap import ECAP
from pyeCAP.utilities.ancillary_functions import configure_local_dask
from pyeCAP.visualization import plot_ecap_surface_plotly

import warnings
import traceback

import dask
from dask.distributed import Client
from dask.diagnostics import ProgressBar


# -------------------------------------------------
# User Inputs
# -------------------------------------------------
client, cluster = configure_local_dask(num_workers=8, cache_gb=8, blas_threads=1)
print(client)
TDT_BLOCK = r"D:\ImThera\data_raw\20191216\Data\Imthera_Pig_Exeriment_25Hz-191216\pnpig191126-191216-141042\\"
animal = Path(TDT_BLOCK).parts[Path(TDT_BLOCK).parts.index("data_raw") + 1]
sample_delay = 24
stores = ['RawE', 'RawG']
rec_ch_names = ['LIFE 1', 'LIFE 2', 'LIFE 3', 'LIFE 4', 'EMG 1', 'EMG 2', 'EMG 3']
rec_ch_types = ['ENG', 'ENG', 'ENG', 'ENG', 'EMG', 'EMG', 'EMG']
pre_ms, post_ms = 1.0, 5.0


# -------------------------------------------------
# Load ephys & stim (lazy, chunked)
# -------------------------------------------------
ephys = Ephys(TDT_BLOCK, stores=stores, sample_delay=sample_delay)
ephys = ephys.remove_ch("RawG 4").set_ch_names(rec_ch_names).set_ch_types(rec_ch_types)
ephys.base_rechunk(time_chunk_samples=2_000_000)
print("Raw rechunk", ephys.data[0].chunks)
stim = Stim(TDT_BLOCK)
stim.set_pulse_amplitude_abs()


# -------------------------------------------------
# Filter Data
# -------------------------------------------------
print("orig", ephys.data[0].chunks)
ephys_f = (
    ephys
    .filter_powerline(frequencies=60, notch_width=None, trans_bandwidth=1, tap_limit=20001)
    .common_reference(method="median", ch_type="ENG")
    .filter_median(btype="highpass", persist_input=False)  # don't checkpoint here
    .filter_gaussian(Wn=4000, btype="lowpass")
)
print("after pipeline", ephys_f.data[0].chunks)

d = ephys_f.data[0]
print("shape:", d.shape)
print("dtype:", d.dtype)
print("chunks:", d.chunks)
print("npartitions (time):", len(d.chunks[1]))
print("scheduler:", dask.config.get("scheduler"))

ephys_f = ephys_f.persist()
print("after persist", ephys_f.data[0].chunks)
# ephys_f   = (ephys
#              .filter_powerline(frequencies=60, notch_width=None, trans_bandwidth=2.0, method="auto")
#              .filter_iir(Wn=(300, 3000), btype="band", order=4, ftype="butter"))


# -------------------------------------------------
# Create ECAP Class
# -------------------------------------------------
distances = [7.7, 8.3, 8.5, 8.6]
ecap_raw = ECAP(ephys, stim, distances)
ecap_filt = ECAP(ephys_f, stim, distances)

# -------------------------------------------------
# Plotting for sanity checks
# -------------------------------------------------
parameter = (0,7)

pulse_scaling_factor = 2
ts_scaling_factor = 2

# data = ecap_filt.epoch_3d(parameter, ecap_filt.neural_channels)
# Dask array: (pulses, channels, samples)

# with ProgressBar():
# #     out = data.compute()

# mean_data = ecap_filt.mean_over_pulses[ecap_filt.parameters_dictionary[parameter],ecap_filt.neural_channels,:].persist()
# median_data = np.median(data, axis=0)
# # shape: (channels, samples)
#
# # #Get largest sample per channel
# # mean_maxes = mean_data.max(axis=1).compute()
# # largest_ch_idx = np.argmax(mean_maxes)
#
#
# n_ch = len(ecap_filt.neural_channels)
#
# fig, ax = plt.subplots(
#     nrows=n_ch, ncols=1,
#     figsize=(12,16),
#     sharex=True,
#     sharey=True,
#     constrained_layout=True,
#     dpi=300)
#
# t = np.asarray(ecap_filt.time_axis(parameter)[::ts_scaling_factor])
#
# for ch_idx, ch in enumerate(ecap_filt.neural_channels):
#
#     # --- Build LineCollection segments of individual pulses---
#     # --- Subsample pulses ---
#     pulses = data[::pulse_scaling_factor, ch_idx, ::ts_scaling_factor].compute()
#     # shape: (n_sel_pulses, n_samples)
#
#     # Each pulse becomes [(t0,y0), (t1,y1), ...]
#     segments = np.stack(
#         [np.column_stack((t, p)) for p in pulses],
#         axis=0
#     )
#
#     lc = LineCollection(
#         segments,
#         colors="gray",
#         linewidths=0.75,
#         alpha=0.8,
#         zorder=0
#     )
#
#     ax[ch_idx].add_collection(lc)
#
#     # --- Mean trace ---
#     mean = mean_data[ch_idx]
#     ax[ch_idx].plot(
#         t,
#         mean[::ts_scaling_factor],
#         label=f"Ch {ch} Mean",
#         color="red",
#         linewidth=2.6,
#         zorder=2
#     )
#
#     # --- Median trace ---
#     median = median_data[ch_idx]
#     ax[ch_idx].plot(
#         t,
#         median[::ts_scaling_factor],
#         label=f"Ch {ch} Mean",
#         color="blue",
#         linewidth=2,
#         zorder=3
#     )
#
#     # Boundary Lines for fiber types
#     bbs = ecap_filt.neural_window_indices[ch_idx,:,:].flatten() / ephys.sample_rate * ts_scaling_factor
#     ax[ch_idx].vlines(
#         x=bbs,
#         ymin=ax[ch_idx].get_ylim()[0],
#         ymax=ax[ch_idx].get_ylim()[1],
#         linestyles="--",
#         linewidth=1,
#         color="black"
#     )
#
#     ax[ch_idx].set_ylabel(f"Ch {ch}")
#
# # Axis limits must be set manually when using LineCollection
# ax[-1].set_xlabel("Time (s)")
# ax[-1].set_ylim(-20e-6,20e-6)
# xmin, _ = ax[-1].get_xlim()
# ax[-1].set_xlim(xmin, t[-1])
#
# legend_handles = [
#     Line2D([0], [0], color="gray", lw=1, alpha=0.3, label="Individual pulses"),
#     Line2D([0], [0], color="red", lw=2.6, label="Mean response"),
#     Line2D([0], [0], color="blue", lw=2, label="Median response"),
# ]
#
# ax[-1].legend(
#     handles=legend_handles,
#     loc="lower right",
#     frameon=True,
#     framealpha=1.0,  # fully opaque background
#     facecolor="white",
#     edgecolor="black"
# )
#
# save_root = Path(r"D:\ImThera\figures\intermediate\qc")
# save_path = save_root / animal
# save_path.mkdir(parents=True, exist_ok=True)
# file_path = save_path / "ecap_epoch_grid.pdf"
# fig.savefig(file_path, dpi=300, bbox_inches="tight")
# plt.close(fig)

### -------------------------------
### TO HERE
### -------------------------------

#
# channel = 0
# pulse = 0
# # ephys.plot_psd()
# # ephys_f.plot_psd()
#
# y0 = np.asarray(ecap_raw.dask_array(parameter=(0, 7))[pulse, channel, ::2])
# y1 = np.asarray(ecap_filt.dask_array(parameter=(0, 7))[pulse, channel, ::2])
# x = np.arange(len(y0))
#
# fig = make_subplots(
#     # rows=2, cols=1,
#     # shared_xaxes=True,
#     # shared_yaxes=True
# )
#
# fig.add_trace(go.Scatter(x=x, y=y0, name="Raw"))
# fig.add_trace(go.Scatter(x=x, y=y1, name="Filtered"))
# colors = ["red", "orange", "yellow", "green", "blue"]
#
# y_min = min([float(min(tr.y)) for tr in fig.data if hasattr(tr, "y") and tr.y is not None])
# y_max = max([float(max(tr.y)) for tr in fig.data if hasattr(tr, "y") and tr.y is not None])
#
# (ecap_filt.neural_window_indices[channel]):




# ymin = min(y0.min(), y1.min())
# ymax = max(y0.max(), y1.max())
#
# fig.update_yaxes(matches="y", row=2, col=1)
#
# fig.update_layout(
#     dragmode="zoom",
#     hovermode="x unified",
# )


#
#
# # --- 4) Show sequence of stim
# param_idx = ecap_raw.parameters_dictionary[stim.parameters['pulse amplitude (μA)'].idxmax()]
# fig, ax = plt.subplots(1,1)
# ax.plot(ecap_raw.mean_over_pulses[param_idx, 0, :])
# ax.plot(ecap_filt.mean_over_pulses[param_idx, 0, :])
# plt.show()
# fig = plot_ecap_surface_interactive(
#         ecap_filt,
#         channel="LIFE 1",
#         x_lim=(0.0, 0.010),
#         absolute=False,
#         downsample=2,
#         renderer=None  # optional: open in your default browser
#     )


# --- 5) Calculate AUC
# window_s = ecap_raw.calculate_neural_window_times()
# params = [(0, 0)]
#
# auc = ecap_raw.calculate_AUC(parameter=params, windows=window_s)


# ecap_raw  = ecap_raw.calculate_AUC(window_type="standard_neural", analysis_method="RMS")
# ecap_raw  = ecap_raw.calculate_AUC(window_type="standard_neural", analysis_method="Peaks")


# # --- 3) Build lazy stacks → mean trace for one channel, then compute
# stack_raw  = ecap_raw.dask_array(param_name)         # (pulses, channels, samples)
# stack_filt = ecap_filt.dask_array(param_name)
#
# ch_idx = 0
# mean_raw  = stack_raw[:, ch_idx, :].mean(axis=0).compute()
# mean_filt = stack_filt[:, ch_idx, :].mean(axis=0).compute()
#
# # --- 4) Plot overlay
# fs = float(ephys.sample_rate)
# t_ms = (np.arange(mean_raw.size) / fs - pre_ms/1000.0) * 1000.0
#
# plt.figure(figsize=(7,4))
# plt.plot(t_ms, mean_raw,  label="Raw mean ECAP",  alpha=0.8)
# plt.plot(t_ms, mean_filt, label="Filtered mean ECAP", linewidth=1.5)
# plt.axvline(0, color="k", linewidth=0.8, alpha=0.5)
# plt.xlim(-pre_ms, post_ms)
# plt.xlabel("Time (ms)"); plt.ylabel("Amplitude (µV)")
# plt.title(f"ECAP — ch {ch_idx} | notch 60 Hz + 300–3000 Hz bandpass")
# plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout(); plt.show()