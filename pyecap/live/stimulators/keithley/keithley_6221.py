import functools
import time
import warnings

import pyvisa


class Keithley6221Controller:
    def __init__(self, address="TCPIP0::192.168.1.2::1394::SOCKET"):
        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.instrument = None
        self.eol = "\r\n"  # Set the end-of-line character
        self.max_samples = 64000  # The maximum number of samples for the 6221
        self.max_sample_rate = 100000  # The maximum sample rate of the 6221
        self.retry_attempts = 3  # Number of retry attempts for connecting

    def retry_decorator(retry_attempts, delay=1, exceptions=(Exception,)):
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                nonlocal retry_attempts
                for attempt in range(retry_attempts):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        print(f"Attempt {attempt+1} failed with error: {e}")
                        time.sleep(delay)
                        if attempt == retry_attempts - 1:
                            raise

            return wrapper

        return decorator

    def ensure_connection(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if not self.check_connection():
                print("Connection lost. Attempting to reconnect...")
                self.connect()
            return func(self, *args, **kwargs)

        return wrapper

    @retry_decorator(retry_attempts=3, delay=2)
    def connect(self):
        try:
            self.instrument = self.rm.open_resource(self.address)
            self.instrument.write_termination = self.eol
            self.instrument.read_termination = "\n"
            self.instrument.write("*IDN?;")
            print(f"Connected to: {self.instrument.read()}")
        except Exception as e:
            print(f"Failed to connect: {e}")

    def disconnect(self):
        if self.instrument is not None:
            self.instrument.close()
            print("Disconnected.")

    def check_connection(self):
        try:
            self.instrument.query("*IDN?")
            return True  # The query was successful, so the connection is good
        except (pyvisa.VisaIOError, pyvisa.errors.VisaIOWarning) as e:
            print(f"Connection check failed: {e}")
        return False  # The query failed, indicating the connection might be lost

    @ensure_connection
    def upload_waveform(self, waveform, freq, amplitude, cycles=1, comp_voltage=10):
        # Set the chunk size for waveform data transmission.
        # This is used to break the waveform data into manageable pieces for transmission.
        chunk_size = 100

        # Normalize the waveform to ensure its amplitude is within the instrument's range.
        # This is done by dividing the waveform by its maximum absolute value.
        waveform_normalized = waveform / max(abs(waveform))

        # Validate the number of samples in the waveform.
        # The waveform length must not exceed the maximum number of samples the instrument can handle.
        if len(waveform_normalized) > self.max_samples:
            raise ValueError(
                f"Waveform length {len(waveform_normalized)} exceeds maximum of {self.max_samples} samples"
            )

        # Calculate the sample rate of the waveform
        sample_rate = len(waveform_normalized) / freq

        # Check if the sample rate is greater than the maximum sample rate of the device
        if sample_rate > self.max_sample_rate:
            from scipy.signal import resample

            # Downsample the waveform to be below the maximum sample rate
            waveform_downsampled = resample(waveform_normalized, self.max_sample_rate)
            # Warn the user about the downsampling
            warnings.warn(
                f"Waveform sample rate {sample_rate} exceeds maximum of {self.max_sample_rate}. Downsampling to {self.max_sample_rate} samples per second."
            )

            # Use the downsampled waveform for the rest of the code
            waveform_normalized = waveform_downsampled

        self.stop_stimulation()
        # Send the initial chunk of the waveform data to the instrument.
        # The data is converted to a comma-separated string for transmission.
        for i in range(0, len(waveform_normalized), chunk_size):
            data_chunk = ",".join(map(str, waveform_normalized[i : i + chunk_size]))
            if i == 0:
                self.instrument.write(f"SOUR:WAVE:ARB:DATA {data_chunk};")
            else:
                self.instrument.write(f"SOUR:WAVE:ARB:APP {data_chunk};")
            # Ensure all waveform data is processed before proceeding
            self.instrument.write(
                "*OPC?"
            )  # Assuming *OPC? is supported for operation complete query
            while not int(self.instrument.read().strip()):
                print("False")
                time.sleep(0.1)  # Wait for the operation to complete

        # Finalize the waveform upload and configure the waveform type to be arbitrary.
        self.instrument.write("SOUR:WAVE:ARB:COPY 1;")
        self.instrument.write("SOUR:WAVE:FUNC ARB1;")

        # Set the waveform frequency.
        self.instrument.write(f"SOUR:WAVE:FREQ {freq};")

        self.instrument.write("*OPC?")
        while not int(self.instrument.read().strip()):
            print("False")
            time.sleep(0.1)  # Wait for the operation to complete
        # Update waveform settings using dedicated methods.
        # These methods encapsulate the logic for setting amplitude, cycles, and compliance voltage.
        self.update_amplitude(amplitude)
        self.update_cycles(cycles)
        self.update_compliance_voltage(comp_voltage)

        # Additional configuration for phase marker and range.
        self.instrument.write("SOUR:WAVE:PMAR:STAT ON;")
        self.instrument.write("SOUR:WAVE:PMAR 0;")
        self.instrument.write("SOUR:WAVE:PMAR:OLIN 1;")
        self.instrument.write("SOUR:WAVE:RANG BEST;")

    @ensure_connection
    def update_amplitude(self, amplitude):
        # Validate amplitude (assuming a certain range, e.g., 0-10V)
        if not 0 <= amplitude <= 10:
            raise ValueError("Amplitude out of range (0-10V).")

        self.instrument.write("SOUR:WAVE:ABOR;")
        self.instrument.write(f"SOUR:WAVE:AMPL {amplitude};")
        self.instrument.write("SOUR:WAVE:RANG BEST;")

    @ensure_connection
    def update_cycles(self, cycles):
        # Validate cycles (must be a positive integer)
        if not isinstance(cycles, int) or cycles < 1:
            raise ValueError("Number of cycles must be a positive integer.")

        self.instrument.write("SOUR:WAVE:ABOR;")
        self.instrument.write(f"SOUR:WAVE:DUR:CYCL {cycles};")

    @ensure_connection
    def update_compliance_voltage(self, comp_voltage):
        # Validate compliance voltage (assuming a certain range, e.g., 0-20V)
        if not 0 <= comp_voltage <= 100:
            raise ValueError("Compliance voltage out of range (0-100V).")

        self.instrument.write("SOUR:WAVE:ABOR;")
        self.instrument.write(f"SOUR:CURR:COMP {comp_voltage};")

    @ensure_connection
    def start_stimulation(self):
        self.instrument.write("SOUR:WAVE:ABOR;")
        self.instrument.write("SOUR:WAVE:FUNC ARB1")
        self.instrument.write("SOUR:WAVE:ARM")
        self.instrument.write("SOUR:WAVE:INIT")

    @ensure_connection
    def stop_stimulation(self):
        self.instrument.write("SOUR:WAVE:ABOR;")
