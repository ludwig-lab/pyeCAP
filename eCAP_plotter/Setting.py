import numpy as np

class ExperimentSettings:
    def __init__(self):
        ##########################################
        self.file_ = np.empty((50, 15), dtype=object) 
        self.Dir = np.empty((50, 15), dtype=object) 
        self.Status = np.empty((50, 15), dtype=object) 
        self.Path = np.empty((50), dtype=object) 
        # Experiment settings
        self.ExpNo                 = 15            # Experiment number
        self.filelist1             = [5]             # List of files to process; filelist can also call list of conditions 
        self.par_list1_            = [5]         # List of parameter indices to plot
        self.stores                = 'CAPc'          # Data store

        # Data processing parameters
        self.Pulse_Ind             = range(1, 299)   # Indices of pulses to process and average
        self.ch_eCAP               = [1,2,3,4,5]     # eCAP channels to process
        self.common_ref            = 0               # Common reference channel
        self.pl_on                 = 0               # powerline filter
        self.lp_on                 = 0               # Low-pass filter
        self.hp_on                 = 0               # High-pass filter
        self.filter_hp_type        = 'median'        # High-pass filter type
        self.filter_lp_type        = 'gaussian'      # Low-pass filter type
        self.sample_delay          = 24              # Sample delay
        self.hp_ks                 = 750             # High-pass median filter kernel size
        self.hpfo                  = 1               # High-pass filter order
        self.lpff                  = 5000            # Low-pass filter frequency
        self.flip_polarity         = False           # Flip polarity of the signal
        self.recording_type        = 'monopolar'     # Recording type (e.g. 'monopolar', 'bipolar', 'tripolar')
        self.Subtract_SA           = False           # Subtract Stimulus Artifact using rescaled empty channel
        self.SA_ch                 = 6               # empty channel for Stimulus Artifact
        self.use_median            = True            # use median as ensemble averaging of traces

        # Figure parameters
        self.name_of_figure        = 'Poster_Fig1_emg'              # Name of the figure
        self.custom_titles         = None #["Cond. 1", "Cond. 2", "Cond. 3", "Cond. 4", "Cond. 5", "Cond. 6", "Cond. 7"] # Custom titles for subplots, if not use enter None 
        self.Subtitle              = ''              # Subtitle for the figure
        self.use_status            = True            # Use status in the title
        self.figsz                 = (7.5, 3.5)         # Figure size
        self.show_legend           = False           # Show legend
        self.lgfnt                 = 12              # Legend font size
        self.lc                    = 'upper right'   # Legend location
        self.ax_sub                = None            # Subplot axis (useful for multi-panel figures)
        self.PlotAggrePara         = 0               # Aggregate parameters in the plot
        self.PlotAggreConds        = 0               # Aggregate conditions in the plot
        self.autoclosefigures      = True            # Auto close previous figures
        self.plot_individual_traces= False            # Plot individual traces
        self.trace_alpha           = 1               # Individual trace transparency = trace_alpha/num_traces
        self.use_broken_axes       = True            # (optional) plot with broken axes  
        self.broken_axes_ylims     =  ((-550, -500),(-250, 200), (1900, 1950))  #Set use_broken_axes to True and provide custom ylims when calling plot_multi_cond_para_chs
        self.save_figure           = True             # Save the figure
        self.file_format           = 'png'            # File format (e.g., 'pdf', 'eps', 'png')

        # eCAP plot settings
        self.eCAPspacing           = 500             # eCAP trace spacing
        self.auto_ylim             = False           # Auto Y-axis limits
        self.y_lim_range           = [-175, 75]      # Y-axis limits
        self.t_ylim                = [2, 20]         # Time range for Y-axis limits calculation
        self.t_xlim                = [0, 2]        # Time range for X-axis
    