"""
EDA: Exploratory Data Analysis
Class is intended to help user conduct preliminary data analysis
1. Ensure waveforms have no time delay
2. Calculate conduction velocities of fiber types
3. See time windows which will be used to calculate compound action potentials
4. Different visualization options based on unique stimulation parameters
"""

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly

import pandas as pd
import numpy as np



from eba_toolkit import Ephys, Stim


class EDA:
    def __init__(self, ephys_data, stim_data):
        self.ephys = ephys_data
        self.stim = stim_data

        def create_non_unique_dict(max_amp):
            sub_df = self.stim.parameters[self.stim.parameters['pulse amplitude (μA)'] == max_amp]
            sub_df_cols = sub_df.columns
            d = {c: sub_df[c].unique() for c in sub_df_cols if len(sub_df[c].unique() > 1)}
            return d

        self.unique_dict = {}

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
            sub_df_cols = sub_df.columns
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


        """
        Initiate plotting
        """
        app = dash.Dash()

        app.layout = html.Div([
            html.H6("Change the value in the text box to see callbacks in action!"),
            html.Div(["Input: ",
                      dcc.Input(id='my-input', value='initial value', type='text')]),
            html.Br(),
            html.Div(id='my-output'),

        ])

        @app.callback(
            Output(component_id='my-output', component_property='children'),
            Input(component_id='my-input', component_property='value')
        )
        def update_output_div(input_value):
            return 'Output: {}'.format(input_value)
        all_dims = [*self.unique_dict]

        return app














