# General
import os

import dask.array as da

# Plotting
import matplotlib.pyplot as plt

# Scientific data handling
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from ...base.event_data import _EventData
from ...base.parameter_data import _ParameterData

# PyNeuroBase imports
from ...base.ts_data import _TsData
from ...base.utils.numeric import _to_numeric_array
from ...base.utils.visualization import _plt_setup_fig_axis, _plt_show_fig

# Electric circuit simulation
# This enables this class to be used as a circuit element in PySpice
# from PySpice.Spice.NgSpice.Shared import NgSpiceShared


class StimElectrical(_TsData):
    def __init__(self, waveform, sample_rate=1e6, cycles=1, stim_type="current"):
        # Validate waveform input
        if not isinstance(waveform, (np.ndarray, da.Array)):
            raise TypeError("Waveform must be a NumPy or Dask array.")
        if waveform.ndim != 1:
            raise ValueError("Waveform must be a 1D array.")

        # Set up stimulation type
        if stim_type in ("voltage", "current"):
            self._stim_type = stim_type
        else:
            raise ValueError("Stimulation type must be 'voltage' or 'current'.")

        self._waveform = waveform
        self._sample_rate = sample_rate
        self._cycles = cycles

        # Metadata dictionary for StimWaveform class
        metadata = {"sample_rate": sample_rate, "start_time": 0.0}

        # Initialize parent class
        _TsData.__init__(
            self,
            [da.tile(self.waveform, self.cycles)[np.newaxis, :]],
            [metadata],
            daskify=isinstance(self.waveform, da.Array),
        )

    def voltage(self):
        return self.waveform if self._stim_type == "voltage" else None

    def current(self):
        return self.waveform if self._stim_type == "current" else None

    @property
    def waveform(self):
        return self._waveform

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def cycles(self):
        return self._cycles

    def plot_waveform(self, *args, axis=None, fig_size=None, show=True, **kwargs):
        fig, ax = _plt_setup_fig_axis(axis=axis, fig_size=fig_size)
        time = self._segment_time()
        ax.plot(time, self.waveform)
        ax.set_xlim(0, time[-1])
        ax.set_ylabel(
            f"{self._stim_type} (amps)"
            if self._stim_type == "current"
            else f"{self._stim_type} (volts)"
        )
        ax.set_xlabel("time (seconds)")
        return _plt_show_fig(fig, ax, show)

    def plot(self, *args, **kwargs):
        _TsData.plot(self, *args, **kwargs)

    def _segment_time(self):
        """Generate a time array for the waveform."""
        time_length = self.waveform.shape[0] / self.sample_rate
        return np.linspace(0, time_length, self.waveform.shape[0])


class SegmentedStim(StimElectrical):
    def __init__(
        self, time, amplitude, *args, sample_rate=1e6, frequency=None, **kwargs
    ):
        time = _to_numeric_array(time)
        amplitude = _to_numeric_array(amplitude)

        if time.shape != amplitude.shape:
            raise ValueError("'time' and 'amplitude' input lengths must match.")
        if time.ndim > 1 or amplitude.ndim > 1:
            raise ValueError("Inputs must be 1D arrays.")

        if time[0] != 0:
            time = np.insert(time, 0, 0.0)
            amplitude = np.insert(amplitude, 0, 0.0)

        if frequency is not None:
            if time[-1] < 1 / frequency:
                time = np.append(time, 1 / frequency)
                amplitude = np.append(amplitude, amplitude[0])
            elif time[-1] > 1 / frequency:
                time[-1] = 1 / frequency
        else:
            frequency = 1 / time[-1]

        # Validate inputs
        if frequency <= 0:
            raise ValueError("Frequency must be a positive value.")

        # if amplitude[0] != amplitude[-1]:
        #     raise ValueError("Start and stop amplitude must be the same.")

        if not np.array_equal(time, np.unique(time)):
            raise ValueError("Time array contains duplicates or is not in order.")

        self._segment_time = time
        self._segment_amplitude = amplitude
        self._sample_rate = sample_rate

        super().__init__(self.waveform, *args, sample_rate=sample_rate, **kwargs)

    def add(self, other):
        """Sum the amplitudes of this stimulus with another segmented stimulus."""
        if not isinstance(other, SegmentedStim):
            raise ValueError("The other object must be an instance of SegmentedStim.")

        # Find unique segment boundaries
        unique_boundaries = np.unique(
            np.concatenate((self._segment_time, other._segment_time))
        )

        # Calculate the amplitude for each segment
        summed_amplitude = []
        for time in unique_boundaries:
            # Calculate amplitude from both stimuli
            amplitude_self = self._amplitude_at(time)
            amplitude_other = other._amplitude_at(time)
            summed_amplitude.append(amplitude_self + amplitude_other)

        print(unique_boundaries)
        print(summed_amplitude)

        # Return a new instance of SegmentedStim
        return SegmentedStim(
            unique_boundaries,
            np.array(summed_amplitude),
            sample_rate=self._sample_rate,
            frequency=1 / unique_boundaries[-1],
        )

    def _amplitude_at(self, time_point):
        """Helper method to get the amplitude of the stimulus at a specific time point."""
        for start, end, amplitude in zip(
            self._segment_time[:-1], self._segment_time[1:], self._segment_amplitude
        ):
            if start <= time_point < end:
                return amplitude
        return 0  # Return 0 if the time_point is not within any segment

    @property
    def frequency(self):
        return 1 / self._segment_time[-1]

    @property
    def waveform(self):
        waveform = []
        sample_time = 1 / self.sample_rate
        for start, stop, value in zip(
            self._segment_time[:-1],
            self._segment_time[1:],
            self._segment_amplitude[:-1],
        ):
            start_sample = int(np.ceil(start / sample_time))
            stop_sample = int(np.floor(stop / sample_time)) + 1
            num_values = stop_sample - start_sample
            waveform.append(np.ones(num_values) * value)  # Using NumPy for simplicity
        return np.concatenate(waveform)

    # Overwrites parent classes plot_waveform function with a more efficient version based on plotting segments
    # instead of sampled data.
    def plot_waveform(self, *args, axis=None, fig_size=None, show=True, **kwargs):
        fig, ax = _plt_setup_fig_axis(axis=axis, fig_size=fig_size)
        ax.step(self._segment_time, self._segment_amplitude, where="post")
        ax.set_xlim(0, self._segment_time[-1])
        if self._stim_type == "current":
            ax.set_ylabel("current (amps)")
        elif self._stim_type == "voltage":
            ax.set_ylabel("voltage (volts)")
        else:
            raise ValueError("Stim type is not recognized.")
        ax.set_xlabel("time (seconds)")
        return _plt_show_fig(fig, ax, show)


class BiphasicStim(SegmentedStim):
    # TODO: Set a reasonable slew rate default
    def __init__(
        self,
        frequency,
        amplitude,
        phase_duration,
        *args,
        delay=0,
        interphase_delay=0,
        charge_balanced=True,
        phase_ratio=1,
        **kwargs,
    ):
        # Make sure frequency is a float and store
        self._frequency = float(frequency)

        # Read in amplitude input and convert to appropriate array
        self._amplitude = _to_numeric_array(amplitude)
        if self._amplitude.shape[0] == 1:
            self._amplitude = np.append(
                self._amplitude, -self._amplitude[0] / phase_ratio
            )
        elif self._amplitude.shape[0] == 2:
            pass
        else:
            raise ValueError(
                "Amplitude input expect a numeric value or an array with length 1 or 2."
            )

        # Read in amplitude input and convert to appropriate array
        self._phase_duration = _to_numeric_array(phase_duration)
        if self._phase_duration.shape[0] == 1:
            self._phase_duration = np.append(
                self._phase_duration, self._phase_duration[0] * phase_ratio
            )
        elif self._phase_duration.shape[0] == 2:
            pass
        else:
            raise ValueError(
                "Phase duration input expect a numeric value or an array with length 1 or 2."
            )
        # If charged balanced adjust duration of second phase to enforce charge balancing
        if charge_balanced:
            self._phase_duration[1] = (
                self._phase_duration[0] * self._amplitude[0] / -self._amplitude[1]
            )

        if interphase_delay > 0:
            time = np.array(
                [
                    delay,
                    delay + self._phase_duration[0],
                    delay + self._phase_duration[0] + interphase_delay,
                    delay
                    + self._phase_duration[0]
                    + self._phase_duration[1]
                    + interphase_delay,
                ]
            )
            amplitude = np.array([self._amplitude[0], 0, self._amplitude[1], 0])
        else:
            time = np.array(
                [
                    delay,
                    delay + self._phase_duration[0],
                    delay + self._phase_duration[0] + self._phase_duration[1],
                ]
            )
            amplitude = np.array([self._amplitude[0], self._amplitude[1], 0])
        print(time)
        print(amplitude)
        super().__init__(time, amplitude, *args, frequency=self._frequency, **kwargs)


class MonophasicStim(SegmentedStim):
    def __init__(self, frequency, amplitude, phase_duration, *args, delay=0, **kwargs):
        self._frequency = float(frequency)
        self._amplitude = _to_numeric_array(amplitude)

        if delay < 0:
            raise ValueError("Delay must be a non-negative value.")

        if self._amplitude.size != 1:
            raise ValueError("Amplitude input should be a single numeric value.")

        self._phase_duration = _to_numeric_array(phase_duration)

        if self._phase_duration.size != 1:
            raise ValueError("Phase duration input should be a single numeric value.")

        if self._phase_duration[0] < 0:
            raise ValueError("Phase duration must be a non-negative value.")

        time = np.array([delay, delay + self._phase_duration[0]])
        amplitude = np.array([self._amplitude[0], 0])

        super().__init__(time, amplitude, *args, frequency=self._frequency, **kwargs)
