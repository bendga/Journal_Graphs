from setup_module import init_setup
from classification_module import init_classification, combine_accuracy_graphs
import numpy as np

samples, sample_comb, sample_iq, labels, snr = init_setup()


easy_mode = 1
group1 = [
        "OOK",
        "AM-SSB-WC",
        "AM-SSB-SC",
        "AM-DSB-WC",
        "AM-DSB-SC",
        "FM",
        "GMSK",
        "OQPSK",
    ]  # Low Order and Analog
group2 = [
        "4ASK",
        "8ASK",
        "16QAM",
        "32QAM",
        "64QAM",
        "128QAM",
        "256QAM",
    ]  # ASK and QAM
group3 = [
        "BPSK",
        "QPSK",
        "8PSK",
        "16PSK",
        "32PSK",
        "16APSK",
        "32APSK",
        "64APSK",
        "128APSK",
    ]  # PSK and APSK

if easy_mode == 1:
    easy_mods = [
        "OOK",
        "4ASK",
        "BPSK",
        "QPSK",
        "8PSK",
        "16QAM",
        "AM-SSB-SC",
        "AM-DSB-SC",
        "FM",
        "GMSK",
        "OQPSK",
    ]
    easy_mask = np.isin(labels, easy_mods)
    labels = labels[easy_mask]
    snr = snr[easy_mask]
    samples = samples[easy_mask]
    sample_comb = sample_comb[easy_mask]
    sample_iq = sample_iq[easy_mask]
    easy_mask = np.isin(labels, easy_mods)
    labels = labels[easy_mask]
    snr = snr[easy_mask]
    

    group1 = np.array(group1)[np.isin(group1,easy_mods)]
    group2 = np.array(group2)[np.isin(group2,easy_mods)]
    group3 = np.array(group3)[np.isin(group3,easy_mods)]
    
groups = [group1, group2, group3]

print("we are here")
print("classification for cumulants...")
accuracy_data1 = init_classification(samples, labels, snr, "cumulants",easy_mode)
print("classification for cumulants+dx...")
accuracy_data2 = init_classification(sample_comb, labels, snr, "cumulants+dx",easy_mode)
print("classification for cumulants IQ...")
accuracy_data3 = init_classification(sample_iq, labels, snr, "cumulants_iq",easy_mode)

combine_accuracy_graphs(accuracy_data1, accuracy_data2, accuracy_data3)
# unique_snr, snr_counts = np.unique(snr, return_counts=True)
# unique_labels, label_counts = np.unique(labels, return_counts=True)

if __name__ == "__main__":
    filepath = "/home/bendegani/AMC_v2/GOLD_XYZ_OSC.0001_1024.hdf5"