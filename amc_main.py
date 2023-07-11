from setup_module import init_setup
from classification_module import init_classification
import numpy as np

samples, samples_dx, samples_real, samples_imag, labels, snr = init_setup()
sample_iq = np.concatenate((samples_real, samples_imag), 1)
sample_comb = np.concatenate((samples, samples_dx), 1)

print("we are here")
print("classification for cumulants...")
init_classification(samples, labels, snr, "cumulants")
print("classification for cumulants+dx...")
init_classification(sample_comb, labels, snr, "cumulants+dx")
print("classification for cumulants IQ...")
init_classification(sample_iq, labels, snr, "cumulants_iq")


# unique_snr, snr_counts = np.unique(snr, return_counts=True)
# unique_labels, label_counts = np.unique(labels, return_counts=True)

if __name__ == "__main__":
    filepath = "/home/bendegani/AMC_v2/GOLD_XYZ_OSC.0001_1024.hdf5"
