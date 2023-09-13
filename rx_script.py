#!/usr/bin/env python3
import sys
sys.path.append('/usr/lib/python3/dist-packages')  # Replace with the actual path to GNU Radio

#!/usr/bin/env python3

from gnuradio import blocks
from gnuradio import gr
from gnuradio import uhd
import os
from algorithm_module import cumulant_generation_complex

class SDRReceiver(gr.top_block):

    def __init__(self, addr, freq, samp_rate, outfile):
        gr.top_block.__init__(self)

        # Create a UHD USRP source block to receive signals
        self.source = uhd.usrp_source(
            ",".join(("", f"addr={addr}")),
            uhd.stream_args(cpu_format="fc32", channels=0),
        )
        
        # Set the center frequency and sample rate
        self.source.set_center_freq(freq)
        self.source.set_samp_rate(samp_rate)

        # Create a custom block to process received samples
        self.custom_block = gr.sync_block("PythonBlock", gr.io_signature(1, 1, gr.sizeof_gr_complex), gr.io_signature(0, 0, 0))
        self.custom_block.set_process_packet_func(self.process_samples)

        # Create an empty list to store feature vectors
        self.feature_vectors = []

        # Connect the source to the custom block
        self.connect((self.source, 0), (self.custom_block, 0))

    def process_samples(self, input_items, output_items):
        # Process received samples here
        signal = input_items[0]  # Input samples

        # Example: Check if a signal condition is met
        if condition_met(signal):
            feature_vector = cumulant_generation_complex(signal)  # Call your custom function
            self.feature_vectors.append(feature_vector)  # Store the feature vector

def condition_met(signal):
    # Implement your signal recognition logic here
    # Return True if the signal condition is met, otherwise False
    return False  # Example: Change this condition as needed

def main():
    # SDR parameters
    sdr_ip_address = "10.0.0.83"  # Replace with the IP address of your SDR
    center_freq = 100e6  # Center frequency in Hz
    sample_rate = 1e6  # Sample rate in samples per second

    # Create the SDR receiver flowgraph
    receiver = SDRReceiver(sdr_ip_address, center_freq, sample_rate, None)

    try:
        # Start the flowgraph
        receiver.start()

        # Run the flowgraph until manually stopped
        input("Press Enter to stop...")
    except KeyboardInterrupt:
        pass
    finally:
        # Stop the flowgraph and cleanup
        receiver.stop()
        receiver.wait()
        
        # Save the feature vectors to a file
        save_feature_vectors(receiver.feature_vectors, "feature_vectors.txt")

def save_feature_vectors(feature_vectors, filename):
    # Save the feature vectors to a text file
    with open(filename, "w") as file:
        for vector in feature_vectors:
            file.write(" ".join(map(str, vector)) + "\n")

if __name__ == '__main__':
    main()
