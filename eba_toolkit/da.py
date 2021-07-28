"""
DA: Descriptive Analysis
Class is intended to help user describe, show and summarize data
1. 3D plot of recording traces with varying stimulation amplitudes
"""

import numpy as np
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import plotly.io as pio
pio.renderers.default = "browser"

class DA:
    def __init__(self, ephys_data, stim_data, ecap_data):
        self.ephys = ephys_data
        self.stim = stim_data
        self.ecap = ecap_data

        if self.ecap.mean_traces.shape[0] > 0:
            self.mean_traces = self.ecap.mean_traces
        else:
            self.ecap.average_data()
            self.mean_traces = self.ecap.mean_traces

        self.ts = np.linspace(0, self.mean_traces.shape[-1] / self.ephys.sample_rate, self.mean_traces.shape[-1])

    def get_max_to_min_amplitude_idx(self, df):
        """
        Returns: dictionary of amplitudes to self.mean_traces index
        """
        # check if amplitudes are negative
        if df['pulse amplitude (μA)'].iloc[0] < 0:
            df['pulse amplitude (μA)'] *= -1
        df_sorted = df.sort_values(by='pulse amplitude (μA)', ascending=False)
        idx_list = df_sorted.index
        mean_idx_list = [self.ecap.parameters_dictionary[i] for i in idx_list]
        d = {df_sorted['pulse amplitude (μA)'].iloc[i]: mean_idx_list[i] for i in range(len(mean_idx_list))}
        return d

    def plot_3d(self):
        channels_to_plot = slice(0, 1)
        condition_to_plot = slice(0,10)
        plotting_df = self.stim.parameters.copy(deep=True)
        plotting_df = plotting_df.iloc[condition_to_plot]
        plot_dict = self.get_max_to_min_amplitude_idx(plotting_df)

        all_z = []
        for i_idx, amp in enumerate(plot_dict.keys()):
            trace_z = self.mean_traces[plot_dict[amp], channels_to_plot]
            for z in trace_z:
                all_z.append(z)

        all_x = [[i]*len(self.ts) for i in plot_dict.keys()]
        all_y = [self.ts] * len(all_z)

        fig = go.Figure(data=[go.Surface(z=all_z, x=all_x, y=all_y)])

        fig.show()
        return fig






