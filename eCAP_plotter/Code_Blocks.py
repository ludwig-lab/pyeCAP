# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 21:28:59 2023

@author: Rex Chin-Hao Chen
"""
import pyeCAP as pyCAP

def preprocess_data(Dir, fileind, exp_no, stores='CAPc', common_ref=0, pl_on=0, hp_on=0, lp_on=0, hp_ks='500', lpff='5000', sample_delay='24', filter_hp_type='median', filter_lp_type='gaussian'):
    """
    Preprocesses the data specified by the file index.

    Parameters:
    fileind (int): Index of the file to be preprocessed.
    exp_no (int): Experiment number.
    stores (str): Store name for pyCAP.Ephys. Default is 'CAPc'.
    common_ref (str or int): Common reference method, 0 if not used. Default is 0.
    pl_on (bool): Apply 60 Hz powerline filter if True. Default is False.
    hp_on (bool): Apply highpass filter if True. Default is False.
    lp_on (bool): Apply lowpass filter if True. Default is False.
    hp_ks (int): Kernel size for highpass median filter. Default is 500 samples.
    lpff (int): Cutoff frequency for lowpass Gaussian filter. Default is 5000Hz.
    sample_delay (int): Sample delay for pyCAP.Ephys. Default is 24 samples.
    filter_hp_type (str, optional): Type of highpass filter ('median' or 'gaussian'). Default is 'median'.
    filter_lp_type (str, optional): Type of lowpass filter ('gaussian'). Default is 'gaussian'.

    Returns:
    ecap_data_CAP_filt (pyCAP.ECAP): Filtered data with stimulus information.
    Amp (list): Pulse amplitudes extracted from the stimulus data.
    Pulses (list): Pulse counts extracted from the stimulus data.
    """

    # Read the data from the specified file using pyCAP.Ephys
    data_CAP = pyCAP.Ephys(Dir[int(exp_no) - 1, int(fileind)], stores=[stores], sample_delay=sample_delay)

    # Apply the common reference correction if specified
    if isinstance(common_ref, str) and (not common_ref == 0):
        data_CAP = data_CAP.common_reference(method=common_ref)

    # Extract the stimulus information using pyCAP.Stim
    stim_data = pyCAP.Stim(Dir[int(exp_no) - 1, int(fileind)])

    # Apply a 60 Hz powerline filter if pl_on flag is set to True
    if pl_on:
        data_CAP = data_CAP.filter_powerline()

    # Apply a highpass filter if hp_on flag is set to True
    if hp_on:
        if filter_hp_type == 'median':
            data_CAP = data_CAP.filter_median(btype='highpass', kernel_size=hp_ks)
        elif filter_hp_type == 'gaussian':
            data_CAP = data_CAP.filter_gaussian(btype='highpass', fcut=hp_ks)
        else:
            raise ValueError("Invalid filter_hp_type. Must be 'median' or 'gaussian'.")

    # Apply a lowpass filter if lp_on flag is set to True
    if lp_on:
        if filter_lp_type == 'gaussian':
            data_CAP = data_CAP.filter_gaussian(lpff, btype='lowpass')
        else:
            raise ValueError("Invalid filter_lp_type. Must be 'gaussian'.")

    # Apply pyCAP.ECAP to the filtered data and stimulus information
    ecap_data_CAP_filt = pyCAP.ECAP(data_CAP, stim_data)

    # Extract pulse amplitudes and counts from the stimulus data
    Amp = pd.DataFrame(stim_data.parameters, columns=['pulse amplitude (μA)']).to_numpy().tolist()
    Pulses = pd.DataFrame(stim_data.parameters, columns=['pulse count']).to_numpy().tolist()

    return ecap_data_CAP_filt, Amp, Pulses



# Function description for compute_traces(ecap_data, index_ar, pulse_indices, channels, flip_polarity=False, recording_type='monopolar'):
# This function computes the traces for the specified channels and stimulus indices using the provided data. The ecap_data argument is the output from pyCAP.ECAP, index_ar is a tuple specifying the start and end indices of the data to be used, pulse_indices is an array of stimulus indices, channels is an array of channel numbers, flip_polarity specifies whether to invert the polarity of the signal, and recording_type specifies the recording configuration (monopolar, bipolar, or tripolar). The function returns an array of traces for each channel.

def compute_traces(ecap_data, index_ar, pulse_indices, channels, flip_polarity=False, recording_type='monopolar', Subtract_SA=False, SA_ch=None):

    traces = []
    num_channels = len(channels)
    if Subtract_SA and SA_ch is not None:
        sa_trace = ecap_data.array(index_ar)[pulse_indices, SA_ch-1]

    for i in range(num_channels):

        ch1_trace = ecap_data.array(index_ar)[pulse_indices, channels[i]-1]

        if Subtract_SA and SA_ch is not None:
            max_ch1 = np.max(ch1_trace)
            min_ch1 = np.min(ch1_trace)
            max_sa = np.max(sa_trace)
            min_sa = np.min(sa_trace)

            minmax_sa = max_sa if abs(max_sa) > abs(min_sa) else min_sa
            minmax_ch1 = max_ch1 if abs(max_ch1) > abs(min_ch1) else min_ch1
            ratio = minmax_ch1 / minmax_sa

            ch1_trace = ch1_trace - (sa_trace * ratio)

        if recording_type == 'monopolar':
            if flip_polarity:
                trace = -ch1_trace
            else:
                trace = ch1_trace
        elif recording_type == 'bipolar':
            if i == num_channels - 1:
                break
            ch2_trace = ecap_data.array(index_ar)[pulse_indices, channels[i + 1]-1]

            if flip_polarity:
                trace = ch2_trace - ch1_trace
            else:
                trace = ch1_trace - ch2_trace
        elif recording_type == 'tripolar':
            if i == 0 or i == num_channels - 1:
                continue
            ch2_trace = ecap_data.array(index_ar)[pulse_indices, channels[i + 1]-1]
            ch0_trace = ecap_data.array(index_ar)[pulse_indices, channels[i - 1]-1]

            if flip_polarity:
                trace = ch1_trace - (ch0_trace + ch2_trace) / 2
            else:
                trace = (ch0_trace + ch2_trace) / 2 - ch1_trace
        else:
            raise ValueError("Invalid recording_type. Must be 'monopolar', 'bipolar', or 'tripolar'.")

        traces.append(trace)

    return traces


# Function description for plot_traces(traces, time_vector, t_ylim, t_xlim, use_median=True, plot_individual_traces=False, title=''):
# This function plots the traces provided in traces against the time vector provided in time_vector. The t_ylim and t_xlim arguments specify the y-axis and x-axis limits of the plot, respectively. The use_median flag specifies whether to plot the median or mean trace for each channel. If the plot_individual_traces flag is set to True, it also plots each individual trace. The title argument allows for a title to be added to the plot. The function returns a plot of the traces.
import matplotlib.pyplot as plt
import numpy as np

def plot_traces(ax, traces, time_vector, t_ylim, t_xlim, auto_ylim=1, y_lim_range=None, y_shift=0, use_median=True, plot_individual_traces=False, title='', legend_labels=None, recording_type=None, trace_alpha=4.5, use_broken_axes=False, broken_axes_ylims=None, lc=4, show_legend=True):
    if use_broken_axes and broken_axes_ylims is None:
        print("Missing broken_axes_ylims. Please provide ylims as a tuple of tuples.")

    # if use_broken_axes:
        # bax = brokenaxes(subplot_spec=ax.get_subplotspec(), ylims=broken_axes_ylims, hspace=0.05)
        # ax_ = bax
    # else:
        # ax_ = ax
    ax_ = ax
        
    colors = plt.cm.tab10(range(0, len(traces)))
    ax_.set_title(title)
    for ch_idx in range(len(traces)):
        ch_traces = traces[ch_idx] * 1e6 + y_shift
        if plot_individual_traces:
            for trace in ch_traces:
                ax_.plot(time_vector, trace, color=colors[ch_idx], alpha=trace_alpha /ch_traces.shape[0])

    for ch_idx in range(len(traces)):
        ch_traces = traces[ch_idx] * 1e6 + y_shift
        label = legend_labels[ch_idx] if legend_labels else f'{recording_type} Ch {ch_idx + 1}'
        if use_median:
            median_trace = np.median(ch_traces, axis=0)
            ax_.plot(time_vector, median_trace, color=colors[ch_idx], label=label)
        else:
            mean_trace = np.mean(ch_traces, axis=0)
            ax_.plot(time_vector, mean_trace, color=colors[ch_idx], label=label)

    # Add a dashed horizontal line at y=0 + y_shift for each parameter set
    ax_.axhline(y=y_shift, linestyle='--', color='gray', alpha=0.5)

    start_index = np.searchsorted(time_vector, t_ylim[0])
    end_index = np.searchsorted(time_vector, t_ylim[1])

    mean_traces = np.mean(traces, axis=1)
    min_value = np.min(mean_traces[:, start_index:end_index]) * 1e6
    max_value = np.max(mean_traces[:, start_index:end_index]) * 1e6
    if not use_broken_axes:
        if auto_ylim:
            ax_.set_ylim(min_value * 1.2, max_value * 1.2)
        else:
            ax_.set_ylim(y_lim_range)
    ax_.set_xlim(t_xlim)

    ax_.set_xlabel('Time (ms)')
    ax_.set_ylabel('Amplitude (μV)')
    ax.legend(loc=lc)
    if not show_legend:
        ax.legend().set_visible(False)
    
    
    
import csv
import pandas as pd
import pdb
import ast

def load_parameters_from_csv(csv_file_name):
    parameters = {}
    with open(csv_file_name, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Read through rows of CSV and extract variable name and value pairs
        for row in csv_reader:
            # Check if row is empty or only contains whitespace
            row_=row[:2]
            if not any(row_) or all(cell.isspace() for cell in row_):
                continue

            # Check if we've reached the "Trace Parameters" section and stop if we have
            if row_[0] == 'Trace Parameters':
                break

            # Check if row contains a variable name and value
            if len(row) >= 2 and row[1]:
                var_name = row_[0]
                var_value = row_[1]

                # Convert value to int or float if possible, otherwise leave it as a string
                try:
                    var_value = int(var_value)
                except ValueError:
                    try:
                        var_value = float(var_value)
                    except ValueError:
                        try:
                            if isinstance(var_value, str) and var_value.startswith('range('):
                                var_value = eval(var_value)  # Convert string representation of range to actual range object
                            var_value = ast.literal_eval(var_value)
                            
                        except (ValueError, SyntaxError):
                            pass

                # Add variable name and value to the work space
    #             if isinstance(var_value, str) and var_value.startswith('[') and var_value.endswith(']'):
    #                 try:
    #                     var_value = ast.literal_eval(var_value)
    #                 except ValueError:
    #                     pass

                globals()[var_name] = var_value
                parameters[var_name] = var_value

    return parameters

# =============================================================================
# The differences between the original plot_multi_cond_para_chs function and the new plot_multi_cond_para_chs_param function that takes the params dictionary as input are as follows:
# The new function plot_multi_cond_para_chs_param takes a single input argument params, which is a dictionary containing the parameter names as keys and their values as values. In contrast, the original function plot_multi_cond_para_chs used global variables to access parameter values.
# At the beginning of the new function, parameter values are extracted from the params dictionary by indexing the dictionary with the corresponding parameter names.
# =============================================================================

def plot_multi_cond_para_chs_param(params):

    # Extract the parameters from the params dictionary
    filelist1 = params["filelist1"][0]
    ExpNo = params["ExpNo"][0]
    common_ref = params["common_ref"][0]
    pl_on = params["pl_on"][0]
    hp_on = params["hp_on"][0]
    lp_on = params["lp_on"][0]
    hp_ks = params["hp_ks"][0]
    lpff = params["lpff"][0]
    sample_delay = params["sample_delay"][0]
    stores = params["stores"][0]
    filter_hp_type = params["filter_hp_type"][0]
    filter_lp_type = params["filter_lp_type"][0]
    par_list1_ = params["par_list1_"][0]
    ch_eCAP = params["ch_eCAP"][0]
    flip_polarity = params["flip_polarity"][0]
    recording_type = params["recording_type"][0]
    Subtract_SA = params["Subtract_SA"][0]
    SA_ch = params["SA_ch"][0]
    use_median = params["use_median"][0]
    custom_titles = params["custom_titles"][0]
    Subtitle = params["Subtitle"][0]
    use_status = params["use_status"][0]
    figsz = params["figsz"][0]
    lgfnt = params["lgfnt"][0]
    lc = params["lc"][0]
    ax_sub = params["ax_sub"][0]
    PlotAggrePara = params["PlotAggrePara"][0]
    PlotAggreConds = params["PlotAggreConds"][0]
    autoclosefigures = params["autoclosefigures"][0]
    plot_individual_traces = params["plot_individual_traces"][0]
    trace_alpha = params["trace_alpha"][0]
    use_broken_axes = params["use_broken_axes"][0]
    broken_axes_ylims = params["broken_axes_ylims"][0]
    eCAPspacing = params["eCAPspacing"][0]
    auto_ylim = params["auto_ylim"][0]
    y_lim_range = params["y_lim_range"][0]
    t_ylim = params["t_ylim"][0]
    t_xlim = params["t_xlim"][0]
    fileCnt = -1
    for fileind in filelist1:
        fileCnt = fileCnt + 1
        ecap_data_CAP_filt, Amp, pulse_counts = preprocess_data(
            Dir,
            fileind,
            ExpNo,
            common_ref=common_ref,
            pl_on=pl_on,
            hp_on=hp_on,
            lp_on=lp_on,
            hp_ks=hp_ks,
            lpff=lpff,
            sample_delay=sample_delay,
            stores=stores,
            filter_hp_type=filter_hp_type,
            filter_lp_type=filter_lp_type
        )
        stim_data = pyCAP.Stim(Dir[ExpNo-1, fileind])
        print(fileind);display(stim_data.parameters)
        Cnt_par = -1
        Amp_arr_sz = np.shape(Amp)

        if not PlotAggrePara:
            par_list1 = [par_list1_[fileCnt]]
        else:
            par_list1 = par_list1_

        y_shift = 0        
        # Create a new figure for each condition when PlotAggreConds = 0
        if not PlotAggreConds and ax_sub is None:
            fig, ax = plt.subplots(figsize=figsz)
        else:
            ax = ax_sub
        
        for par_ind in par_list1:
            Cnt_par = Cnt_par + 1
            index_ar = (0, par_ind)
            # Compute traces
            traces = compute_traces(ecap_data_CAP_filt, index_ar, Pulse_Ind, ch_eCAP, flip_polarity, recording_type, Subtract_SA, SA_ch)

            # Convert traces to a NumPy array
            traces = np.array(traces)

            # Define time_vector
            time_vector = ecap_data_CAP_filt.time(index_ar) * 1e3

            # Determine the maximum and minimum amplitude values for each channel in the specified time window
            start_index = np.searchsorted(time_vector, t_ylim[0])
            end_index = np.searchsorted(time_vector, t_ylim[1])

            min_values = np.min(traces[:, :, start_index:end_index], axis=1)
            max_values = np.max(traces[:, :, start_index:end_index], axis=1)

            y_shift_step = np.max(max_values - min_values) * 1.2

            # Increment y_shift for next parameter set
            if PlotAggrePara and Cnt_par > 0:
                y_shift += y_shift_step

            # Plot the traces
            if custom_titles is not None:
                title = custom_titles[fileCnt]
            else:
                title = Subtitle
                if use_status:
                    title += ' ' + Status[ExpNo - 1, fileind]
            legend_labels = [f'{recording_type} Ch {i + 1}' for i in range(len(ch_eCAP))]
            plot_traces(ax, traces, time_vector, t_ylim, t_xlim, auto_ylim=auto_ylim, y_lim_range=y_lim_range, y_shift=y_shift, use_median=use_median, plot_individual_traces=plot_individual_traces, title=title, legend_labels=legend_labels, recording_type=recording_type, trace_alpha=trace_alpha, use_broken_axes=use_broken_axes, broken_axes_ylims=broken_axes_ylims)
            # Add text to the right side of each parameter's dashed line
            text_x = t_xlim[1]  # Right end of the x-axis range
            text_y = y_shift
            ax.text(text_x * 1.01, text_y, str(Amp[par_ind][0]) + "uA", fontsize=lgfnt, va='center', ha='left')

            # Reset the legend labels for the next condition when PlotAggrePara = 0
            if not PlotAggrePara:
                legend_labels = [None] * len(ch_eCAP)

        # Show the plot when PlotAggreConds = 0
        if not PlotAggreConds:
            plt.show()

    # Show the plot when PlotAggreConds = 1
    if PlotAggreConds:
        plt.show()