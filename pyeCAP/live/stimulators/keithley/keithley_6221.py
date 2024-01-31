import time

import pyvisa


class Keithley6221Controller:
    def __init__(self, address="TCPIP0::192.168.1.2::1394::SOCKET"):
        self.address = address
        self.rm = pyvisa.ResourceManager()
        self.instrument = None
        self.eol = "\r\n"  # Set the end-of-line character
        self.max_samples = 64000  # Replace with actual maximum number of samples
        self.max_sample_rate = 100000  # Replace with actual maximum sample rate in Hz

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

        self.stop_stimulation()
        # Send the initial chunk of the waveform data to the instrument.
        # The data is converted to a comma-separated string for transmission.
        data_to_print = ",".join(map(str, waveform_normalized[:chunk_size]))
        self.instrument.write(f"SOUR:WAVE:ARB:DATA {data_to_print};")

        # Iteratively send the remaining waveform data in chunks.
        # This loop handles the transmission of the waveform data beyond the initial chunk.
        for i in range(chunk_size, len(waveform_normalized), chunk_size):
            data_chunk = ",".join(map(str, waveform_normalized[i : i + chunk_size]))
            self.instrument.write(f"SOUR:WAVE:ARB:APP {data_chunk};")
            # A brief pause is included to prevent overloading the instrument's buffer.
            time.sleep(0.1)

        # Finalize the waveform upload and configure the waveform type to be arbitrary.
        self.instrument.write("SOUR:WAVE:ARB:COPY 1;")
        self.instrument.write("SOUR:WAVE:FUNC ARB1;")

        # Set the waveform frequency.
        self.instrument.write(f"SOUR:WAVE:FREQ {freq};")

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

    def update_amplitude(self, amplitude):
        # Validate amplitude (assuming a certain range, e.g., 0-10V)
        if not 0 <= amplitude <= 10:
            raise ValueError("Amplitude out of range (0-10V).")

        self.instrument.write("SOUR:WAVE:ABOR;")
        self.instrument.write(f"SOUR:WAVE:AMPL {amplitude};")
        self.instrument.write("SOUR:WAVE:RANG BEST;")

    def update_cycles(self, cycles):
        # Validate cycles (must be a positive integer)
        if not isinstance(cycles, int) or cycles < 1:
            raise ValueError("Number of cycles must be a positive integer.")

        self.instrument.write("SOUR:WAVE:ABOR;")
        self.instrument.write(f"SOUR:WAVE:DUR:CYCL {cycles};")

    def update_compliance_voltage(self, comp_voltage):
        # Validate compliance voltage (assuming a certain range, e.g., 0-20V)
        if not 0 <= comp_voltage <= 100:
            raise ValueError("Compliance voltage out of range (0-100V).")

        self.instrument.write("SOUR:WAVE:ABOR;")
        self.instrument.write(f"SOUR:CURR:COMP {comp_voltage};")

    def start_stimulation(self):
        self.instrument.write("SOUR:WAVE:ABOR;")
        self.instrument.write("SOUR:WAVE:FUNC ARB1")
        self.instrument.write("SOUR:WAVE:ARM")
        self.instrument.write("SOUR:WAVE:INIT")

    def stop_stimulation(self):
        self.instrument.write("SOUR:WAVE:ABOR;")
