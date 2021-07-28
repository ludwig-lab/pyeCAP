"""
EDA: Exploratory Data Analysis
Class is intended to help user conduct preliminary data analysis
1. Ensure waveforms have no time delay
2. Calculate conduction velocities of fiber types
3. See time windows which will be used to calculate compound action potentials
4. Different visualization options based on unique stimulation parameters
"""

from .base.epoch_data import _EpochData

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

import pandas as pd
import numpy as np

from eba_toolkit import Ephys, Stim


class EDA(_EpochData):
    def __init__(self, ephys_data, stim_data):
        self.ephys = ephys_data
        self.stim = stim_data
        self.df_max_stim_parameters = None
        self.df_max_stim_unique_parameters = None

        self.unique_dict = {}

        super().__init__(ephys_data, stim_data, stim_data)

    def show_onset(self, amp_select='max amp'):
        def find_max_amp(amps):
            min_max = [amps.min(), amps.max()]

            # if amplitude has same sign
            if min_max[0] * min_max[1] >= 0:
                # if negative
                if min_max[0] < 0:
                    max_amp = min_max[0]
                else:
                    max_amp = min_max[1]

            # if opposite sign
            else:
                # if negative greater
                if abs(min_max[0]) > abs(min_max[1]):
                    max_amp = min_max[0]
                else:
                    max_amp = min_max[1]
            return max_amp

        def create_non_unique_dict(max_amp):
            sub_df = self.stim.parameters[self.stim.parameters['pulse amplitude (μA)'] == max_amp]
            self.df_max_stim_parameters = sub_df
            self.df_max_stim_unique_parameters = self.df_max_stim_parameters.copy(deep=True)
            sub_df_cols = sub_df.columns

            for col in sub_df_cols:
                if len(self.df_max_stim_unique_parameters[col].unique()) <= 1:
                    self.df_max_stim_unique_parameters.drop(columns=col, inplace=True)

            d = {c: sub_df[c].unique() for c in sub_df_cols if len(sub_df[c].unique()) > 1}
            if len(self.ephys.ch_names) > 1:
                d['Channel Names'] = self.ephys.ch_names
            ch_types = list(set(self.ephys.ch_types))
            if len(ch_types) > 1:
                d['Channel Types'] = ch_types
            return d

        if amp_select == 'max amp':
            amps = self.stim.parameters['pulse amplitude (μA)'].unique()
            max_amp = find_max_amp(amps)
        else:
            max_amp = amp_select

        self.unique_dict = create_non_unique_dict(max_amp)
        all_dims = [*self.unique_dict]
        all_dims.insert(0, "None")

        """
        Initiate plotting
        """
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

        app.layout = html.Div([
            html.Div([

                html.Div([
                    dcc.Dropdown(
                        id='row_facet',
                        options=[{'label': i, 'value': i} for i in all_dims],
                    ),
                ],
                    style={'width': '48%', 'display': 'inline-block'}),

                html.Div([
                    dcc.Dropdown(
                        id='col_facet',
                    ),
                ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
            ]),

            dcc.Graph(id='graphic'),
        ])

        @app.callback(
            Output('col_facet', 'options'),
            Input('row_facet', 'value'))
        def set_col_value(row_facet_value):
            working_dims = all_dims
            if row_facet_value == "Channel Names":
                working_dims.remove("Channel Types")
            elif row_facet_value == "Channel Types":
                working_dims.remove("Channel Names")
            new_col_facet = [{'label': i, 'value': i} for i in working_dims if i != row_facet_value]
            return new_col_facet

        @app.callback(
            Output('graphic', 'figure'),
            Input('row_facet', 'value'),
            Input('col_facet', 'value'),
        )
        def update_output(row_facet, col_facet):
            # if row_facet is None or col_facet is None:
            #     return

            if row_facet == "None" or row_facet is None:
                num_rows = 1
            else:
                num_rows = len(self.unique_dict[row_facet])
            if col_facet == "None" or col_facet is None:
                num_cols = 1
            else:
                num_cols = len(self.unique_dict[col_facet])

            unique_ch_types = self.unique_dict['Channel Types']
            ch_list_by_type = [np.array(self.ephys.ch_names)[self.ephys._ch_type_to_index(i)] for i in unique_ch_types]


            unique_ch_names = self.unique_dict['Channel Names']
            ch_list_by_name = [np.array(self.ephys.ch_names)[self.ephys._ch_to_index(i)][0] for i in
                               unique_ch_names]

            fig = make_subplots(rows=num_rows, cols=num_cols)

            if row_facet is None or col_facet is None:
                return fig

            if row_facet == 'Channel Types' and row_facet != "None":
                num_rows = len(ch_list_by_type)

                if col_facet != "Channel Names" and col_facet != "None":
                    # Get values that make columns unique
                    unique_vals = self.unique_dict[col_facet]
                    
                    # Get the MultiIndex of parameters that meet unique value criteria
                    param_idx_row = [self.df_max_stim_parameters.index[self.df_max_stim_parameters[col_facet] == i][0] for i in
                                     unique_vals]

                    title_str = [col_facet + ": " + str(uv) for uv in unique_vals]
                    fig = make_subplots(rows=num_rows, cols=num_cols,
                                        subplot_titles=title_str,
                                        shared_xaxes='all', shared_yaxes='all',
                                        vertical_spacing=0.02)

                    for r_idx, r in enumerate(range(num_rows)):
                        for c_idx, c in enumerate(range(num_cols)):
                            trace_y = self.mean(param_idx_row[c_idx], tuple(ch_list_by_type[r_idx])).compute()
                            trace_ts = np.linspace(0, trace_y.shape[-1] / self.ephys.sample_rate, trace_y.shape[-1])

                            for t_idx, t in enumerate(trace_y):
                                fig.append_trace(go.Scatter(x=trace_ts, y=t, name=ch_list_by_type[r_idx][t_idx]), row=r_idx+1, col=c_idx+1)
                            
                            fig.update_xaxes(title_text="Time (s)")

                        title_str = 'Channel Type: ' + unique_ch_types[r_idx]
                        fig.update_yaxes(title_text=title_str, row=r_idx+1, col=1)

                    fig.update_layout(height=600*num_rows, width=600*num_cols, showlegend=False)

            elif row_facet == 'Channel Names':
                unique_vals = self.ephys.ch_names

                param_idx_row = self.df_max_stim_parameters.index

                title_str = [col_facet + ": " + str(uv) for uv in unique_vals]

                fig = make_subplots(rows=num_rows, cols=num_cols,
                                    subplot_titles=title_str,
                                    shared_xaxes='all', shared_yaxes='all',
                                    vertical_spacing=0.02)

                for r_idx, r in enumerate(range(num_rows)):
                    for c_idx, c in enumerate(range(num_cols)):

                        for p in param_idx_row:
                            trace_y = self.mean(p, unique_vals[c_idx]).compute()
                            trace_ts = np.linspace(0, trace_y.shape[-1] / self.ephys.sample_rate, trace_y.shape[-1])

                            for t in trace_y:
                                fig.append_trace(go.Scatter(x=trace_ts, y=t,
                                                            name=str(self.df_max_stim_unique_parameters.loc[p])),
                                                 row=r_idx + 1, col=c_idx + 1)

                        fig.update_xaxes(title_text="Time (s)")

                    title_str = 'Channel Type: ' + unique_ch_types[r_idx]
                    fig.update_yaxes(title_text=title_str, row=r_idx + 1, col=1)

                fig.update_layout(height=600 * num_rows, width=600 * num_cols, showlegend=False)
                    


            return fig

        return app
