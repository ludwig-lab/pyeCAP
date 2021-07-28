#%%
from eba_toolkit import Ephys, Stim, ECAP
import pandas as pd

data = [r'D:\20191216\TDT\Imthera_Pig_Exeriment_25Hz-191216\pnpig191126-191216-130637',
        r'D:\20191216\TDT\Imthera_Pig_Exeriment_25Hz-191216\pnpig191126-191216-132559']
data_ephys = Ephys(data=data, sample_delay=23)
data_ephys = data_ephys.remove_ch('RawG 4')
data_ephys = data_ephys.set_ch_names(['LIFE 1', 'LIFE 2', 'LIFE 3', 'LIFE 4', 'EMG 1', 'EMG 2', 'EMG 3'])
data_ephys = data_ephys.set_ch_types(['ENG', 'ENG', 'ENG', 'ENG', 'EMG', 'EMG', 'EMG'])
data_stim = Stim(file_path=data)
data_stim.add_series(0, pd.Series(data=["Condition 1"], name="Stimulation Condition"))
data_stim.add_series(1, pd.Series(data=["Condition 2"], name="Stimulation Condition"))
data_ecap = ECAP(data_ephys, data_stim)
data_ecap.average_data()
#%%
from eba_toolkit import DA
data_da = DA(data_ephys, data_stim, data_ecap)
data_da.plot_3d()

# app = data_da.plot_3d()
#
# if __name__ == '__main__':
#     app.run_server(debug=True)
#
# print("Test")