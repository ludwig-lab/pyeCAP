# examples/inspect_tdt_block.py
from pyeCAP.acquisition_io.tdt_io import TdtIO, TdtStim
from pyeCAP import Ephys, Stim, ECAP
from pprint import pprint

TDT_BLOCK = r"D:\ImThera\data_raw\20191204\Data\Imthera_Pig_Exp_25Hz_ShortBurst-191204\pnpig191126-191204-152945"  # <-- change to a real TDT tank/block folder

ephys = Ephys(TDT_BLOCK)
stim  = Stim(TDT_BLOCK)  # now succeeds

ecap = ECAP(ephys, stim, trigger_channel="Stim 1", x_lim=(0.0, 0.01))
ecap.persist_ts(time_chunk=int(0.01 * ecap.ts_data.sample_rate * 6))

param = ecap.parameters.parameters.index[0]
print("Lazy array shape:", ecap.dask_array(param).shape)
print("RMS:", ecap.rms_in_window(param, (0.001, 0.003)).compute())