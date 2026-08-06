import numpy as np
import dask.array as da
import pytest

# ---- Minimal fakes to exercise the new logic without real IO ----

class FakeTsData:
    def __init__(self, array, fs, start_indices=None):
        # array: (channels, time) numpy
        self._arr = da.from_array(array, chunks=(array.shape[0], max(1, array.shape[1] // 4)))
        self._fs = fs
        self._start_indices = np.array(start_indices if start_indices is not None else [0], dtype=int)

    @property
    def array(self):
        return self._arr

    @property
    def shape(self):
        return self._arr.shape

    @property
    def sample_rate(self):
        return self._fs

    @property
    def start_indices(self):
        return self._start_indices

    def _time_to_index(self, t, remove_gaps=True):
        # accepts scalar or np.ndarray
        t = np.asarray(t, dtype=float)
        return np.rint(t * self._fs).astype(int)


class FakeEventData:
    def __init__(self, ch_names, event_times_by_ch, indicators_by_ch, params_index):
        self._ch_names = ch_names
        self._events = event_times_by_ch
        self._ind = indicators_by_ch
        # emulate a DataFrame index of tuples like [(0,0), (0,1), ...]
        class DummyParams:
            def __init__(self, idx):
                self.index = idx
        self.parameters = DummyParams(params_index)

    @property
    def ch_names(self):
        return self._ch_names

    def events(self, ch, start_times=None):
        # start_times is provided by _EpochData; we ignore it for the fake.
        return np.asarray(self._events[ch], dtype=float)

    def event_indicators(self, ch):
        return np.asarray(self._ind[ch], dtype=int)


class MiniEpoch:
    """
    Tiny harness that holds your refactored methods from _EpochData.
    Only what we need for the test.
    """
    def __init__(self, ts_data, event_data, x_lim='auto'):
        self.ts_data = ts_data
        self.event_data = event_data
        self.x_lim = x_lim
        self.parameters = event_data  # only need .parameters.index for time_axis helper

    # ---- paste your refactored persist_ts() and dask_array() here ----
    def persist_ts(self, time_chunk=None):
        arr = self.ts_data.array
        if time_chunk is not None:
            # BEFORE: arr = arr.rechunk({0: arr.chunks[0], 1: (time_chunk,)})
            arr = arr.rechunk({0: arr.chunks[0], 1: int(time_chunk)})  # <-- int, not tuple
        self._persisted_array = arr.persist()
        return self

    def dask_array(self, parameter):
        """
        Returns a (pulses, channels, samples) Dask array for the given stimulation parameter
        by stacking per-event slices. Deduplicates events across channels by sample index.
        """
        import numpy as np
        import dask.array as da

        fs = self.ts_data.sample_rate
        n_time = self.ts_data.shape[1]

        # -------- gather per-channel events for this parameter, as integer sample indices --------
        idx_lists = []
        for ch in self.event_data.ch_names:
            events = self.event_data.events(
                ch,
                start_times=self.ts_data.start_indices / fs
            )
            indicators = self.event_data.event_indicators(ch)
            mask = np.logical_and(indicators[:, 0] == parameter[0],
                                  indicators[:, 1] == parameter[1])
            if np.any(mask):
                # convert to sample indices *without* any window shift yet
                idx_lists.append(self.ts_data._time_to_index(events[mask]))

        if not idx_lists:
            return da.zeros((0, self.ts_data.shape[0], 0), dtype=self.ts_data.array.dtype)

        # Union & sort: unique event sample indices across all channels
        event_idx = np.unique(np.concatenate(idx_lists, axis=0))

        # -------- pick window length --------
        if self.x_lim is None or self.x_lim == 'auto':
            # Estimate from min inter-event spacing in samples (fallback = 1 sample)
            if event_idx.size > 1:
                min_gap_samples = int(np.clip(np.min(np.diff(event_idx)), 1, None))
            else:
                min_gap_samples = 1
            sample_len = min_gap_samples
            x0_shift = 0.0
        else:
            x0, x1 = self.x_lim
            sample_len = int(np.rint((x1 - x0) * fs))
            x0_shift = float(x0)

        # -------- build (start, stop) windows in samples --------
        # If x_lim has a negative start (e.g., pre-stim), shift starts left by x0_shift
        starts = event_idx + int(np.floor(x0_shift * fs))
        stops = starts + sample_len

        # clip to array bounds & drop invalid
        starts = np.clip(starts, 0, n_time)
        stops = np.clip(stops, 0, n_time)
        valid = stops > starts
        if not np.any(valid):
            return da.zeros((0, self.ts_data.shape[0], 0), dtype=self.ts_data.array.dtype)
        starts = starts[valid]
        stops = stops[valid]

        # -------- choose source array (persisted if available) and rechunk thoughtfully --------
        src = getattr(self, "_persisted_array", None)
        if src is None:
            src = self.ts_data.array  # (channels, time), dask

        # Aim for ~1–2 chunks per window on time axis; use int (not tuple) so Dask tiles correctly.
        # Guard against tiny/zero sizes.
        current_min_tchunk = min(src.chunks[1])
        target_t = max(1, sample_len * 4, current_min_tchunk)
        src = src.rechunk({0: src.chunks[0], 1: int(target_t)})

        # -------- slice per-window, then stack late --------
        windows = [src[:, s:e] for s, e in zip(starts, stops)]
        return da.stack(windows, axis=0)  # (pulses, channels, samples)

    # convenience for the test
    def mean_waveform(self, parameter):
        arr3 = self.dask_array(parameter)
        return da.mean(arr3, axis=0)  # (ch, samples)

    def rms_in_window(self, parameter, window_s):
        fs = self.ts_data.sample_rate
        wf = self.mean_waveform(parameter)  # (ch, samples)
        i0 = int(np.floor(window_s[0] * fs))
        i1 = int(np.ceil( window_s[1] * fs))
        seg = wf[:, i0:i1]
        return da.sqrt(da.mean(seg**2, axis=1))  # (ch,)

    def auc_in_window(self, parameter, window_s):
        fs = self.ts_data.sample_rate
        dt = 1.0 / fs
        wf = self.mean_waveform(parameter)  # (ch, samples)
        i0 = int(np.floor(window_s[0] * fs))
        i1 = int(np.ceil( window_s[1] * fs))
        seg = wf[:, i0:i1]
        left, right = seg[:, :-1], seg[:, 1:]
        return da.sum((left + right) * (0.5 * dt), axis=1)  # (ch,)


# ---- The actual test ----

def test_epoch_slice_stack_vectorized_features():
    fs = 10_000  # 10 kHz
    T = 2.0      # 2 seconds of data
    n = int(T * fs)
    t = np.arange(n) / fs

    # Two channels: ch0 is a clean sine, ch1 is sine + small offset
    ch0 = 0.0 * t
    ch1 = 0.0 * t
    # Stim events every 100 ms in the first second only
    event_times = np.arange(0.100, 1.000, 0.100)
    # Inject a 2 ms, 100 Hz damped pulse response after each event (simple shape)
    def pulse_response(tt, t0):
        win = (tt >= t0) & (tt < t0 + 0.010)  # 10 ms window
        x = np.zeros_like(tt)
        # 2 ms burst at 300 Hz with exponential decay
        x[win] = np.sin(2*np.pi*300*(tt[win]-t0)) * np.exp(-(tt[win]-t0)/0.004)
        return x

    for t0 in event_times:
        ch0 += 0.5 * pulse_response(t, t0)
        ch1 += 0.8 * pulse_response(t, t0) + 0.02  # slightly larger + offset

    data = np.vstack([ch0, ch1])  # (2, n)
    ts = FakeTsData(data, fs=fs)

    # Build event indicators/labels for a single parameter (0,0)
    # indicators rows correspond to each pulse for each channel.
    # For simplicity, both channels share the same events and each event belongs to param (0,0).
    inds = np.column_stack([
        np.zeros(len(event_times), dtype=int),  # dataset index = 0
        np.zeros(len(event_times), dtype=int),  # stim index = 0
    ])
    event_times_by_ch = {
        "RawE 1": event_times,
        "RawE 2": event_times,
    }
    indicators_by_ch = {
        "RawE 1": inds,
        "RawE 2": inds,
    }
    params_index = [(0,0)]  # pretend our parameter table has one parameter tuple

    ev = FakeEventData(["RawE 1","RawE 2"], event_times_by_ch, indicators_by_ch, params_index)
    epoch = MiniEpoch(ts, ev, x_lim=(0.0, 0.010))  # fixed 10 ms window
    epoch.persist_ts(time_chunk=int(0.010*fs*6))   # ~60 ms chunks

    arr3 = epoch.dask_array((0,0))  # (pulses, ch, samples)
    pulses, chs, samples = arr3.shape
    assert pulses == len(event_times)
    assert chs == 2
    assert samples == int(0.010 * fs)

    # Mean waveform should have a clear positive lobe; RMS and AUC higher for ch1
    wf = epoch.mean_waveform((0,0)).compute()        # (2, samples)
    rms = epoch.rms_in_window((0,0), (0.000, 0.010)).compute()
    auc = epoch.auc_in_window((0,0), (0.000, 0.010)).compute()

    # basic sanity checks
    assert wf.shape == (2, samples)
    assert rms.shape == (2,)
    assert auc.shape == (2,)

    # ch1 has greater amplitude → larger RMS and AUC
    assert rms[1] > rms[0]
    assert auc[1] > auc[0]

    # values are non-trivial (not all zeros)
    assert rms.mean() > 0.001
    assert auc.mean() > 0.0
