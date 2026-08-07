from pyeCAP import Ephys, Stim, ECAP
from pyeCAP.visualization import plot_ecap_surface_plotly

if __name__ == "__main__":
    TDT_BLOCK = r"D:\ImThera\data_raw\20191216\Data\Imthera_Pig_Exeriment_25Hz-191216\pnpig191126-191216-141042"
    stores=['RawE', 'RawG']
    rec_ch_names = ['LIFE 1', 'LIFE 2', 'LIFE 3', 'LIFE 4', 'EMG 1', 'EMG 2', 'EMG 3']
    rec_ch_types = ['ENG', 'ENG', 'ENG', 'ENG', 'EMG', 'EMG', 'EMG']
    ephys = Ephys(TDT_BLOCK, stores=stores)
    ephys = ephys.remove_ch("RawG 4")
    ephys = ephys.set_ch_names(rec_ch_names)
    ephys = ephys.set_ch_types(rec_ch_types)
    stim  = Stim(TDT_BLOCK)
    ecap  = ECAP(ephys, stim)
    ecap.epoch_window = (-0.004, 0.050)

    fig = plot_ecap_surface_plotly(
        ecap,
        channel="LIFE 1",
        x_lim=(0.0, 0.010),
        constraints=None,  # start with None; add filters once you see columns
        baseline_s=None,  # optional baseline subtraction
        absolute=False,
        downsample=1,
        renderer=None  # optional: open in your default browser
    )
