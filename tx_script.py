#!/usr/bin/env python3

import h5py
from gnuradio import blocks
from gnuradio import gr
from gnuradio import uhd

def load_signals(filepath):
    with h5py.File(filepath, "r") as f:
        data = f["X"]
        modulation_onehot = f["Y"]
        snr = f["Z"]
        # Process the data as needed, e.g., feature extraction
        # Return the processed data to be transmitted
        return data, modulation_onehot, snr

class Transmitter(gr.top_block):

    def __init__(self, addr, freq, samp_rate, data):
        gr.top_block.__init__(self)

        # Create a UHD USRP sink block to transmit signals
        self.sink = uhd.usrp_sink(
            ",".join(("", f"addr={addr}")),
            uhd.stream_args(cpu_format="fc32", channels=0),
        )

        # Set the center frequency and sample rate
        self.sink.set_center_freq(freq)
        self.sink.set_samp_rate(samp_rate)

        # Create a signal source block to generate signals from loaded data
        self.source = blocks.vector_source_c(data, False, 1)

        # Connect the source to the sink
        self.connect(self.source, self.sink)

def main():
    # SDR parameters
    sdr_ip_address = "192.168.10.2"  # Replace with the IP address of your SDR
    center_freq = 100e6  # Center frequency in Hz
    sample_rate = 1e6  # Sample rate in samples per second

    # Input HDF5 file containing signals
    filepath = "/home/bendegani/AMC_v2/GOLD_XYZ_OSC.0001_1024.hdf5"

    # Load signals from the HDF5 file
    data, modulation_onehot, snr = load_signals(filepath)

    # Create the transmitter flowgraph and transmit the loaded data
    transmitter = Transmitter(sdr_ip_address, center_freq, sample_rate, data)

    try:
        # Start the flowgraph
        transmitter.start()
        transmitter.wait()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the flowgraph and cleanup
        transmitter.stop()
        transmitter.wait()

if __name__ == '__main__':
    main()
