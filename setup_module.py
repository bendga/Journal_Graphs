import h5py
import numpy as np
from numpy import abs as abs
from numpy import log10 as log10

# import openpyxl
import pandas as pd
import pickle

# import matplotlib.pyplot as plt
import os


def init_setup():
    # check if there isnt a pre-prepared data
    if not os.path.exists("data"):
        # Create the folder if it doesn't exist
        os.makedirs("data")

    if not os.listdir("data"):
        filepath = "/home/bendegani/AMC_v2/GOLD_XYZ_OSC.0001_1024.hdf5"
        import_data(filepath)

    idx = 2
    print("loading data")
    # load the data
    cumulants_vec = save_and_load_list(2, "cumulants_vec", idx)
    cum_dx_vec = save_and_load_list(2, "cum_dx_vec", idx)
    cum_real_vec = save_and_load_list(2, "cum_real_vec", idx)
    cum_imag_vec = save_and_load_list(2, "cum_imag_vec", idx)
    modulation = save_and_load_list(2, "labels", idx)
    snr_vec = save_and_load_list(2, "snr_vec", idx)
    print("fixing for absolute value")
    cumulants_vec = cumulant_fix_complex(cumulants_vec)
    cum_dx_vec = cumulant_fix_complex(cum_dx_vec)
    cum_iq_vec = cumulant_fix_iq(cum_real_vec, cum_imag_vec)
    # convert to array
    print("convert data")
    labels = np.array(modulation)
    snr = np.array(snr_vec)
    samples = np.array(cumulants_vec)
    samples_dx = np.array(cum_dx_vec)
    sample_iq = np.array(cum_iq_vec)
    sample_comb = np.concatenate((samples, samples_dx), 1)

    return samples, sample_comb, sample_iq, labels, snr


def import_data(filepath):
    with h5py.File(filepath, "r") as f:
        data = f["X"]
        modulation_onehot = f["Y"]
        snr = f["Z"]
        (
            cumulants_vec,
            cum_dx_vec,
            cum_real_vec,
            cum_imag_vec,
            labels,
            snr_vec,
        ) = process_data(data, modulation_onehot, snr)
    # Save the cumulants, labels, and SNRs to disk
    print("Data generation complete")
    print("Saving.....")
    idx = 1
    save_and_load_list(snr_vec, "snr_vec", idx)
    save_and_load_list(labels, "labels", idx)
    save_and_load_list(cumulants_vec, "cumulants_vec", idx)
    save_and_load_list(cum_real_vec, "cum_real_vec", idx)
    save_and_load_list(cum_imag_vec, "cum_imag_vec", idx)
    save_and_load_list(cum_dx_vec, "cum_dx_vec", idx)

    print("save is done")


def process_data(data, modulation_onehot, snr):
    mods_total = [
        "OOK",
        "4ASK",
        "8ASK",
        "BPSK",
        "QPSK",
        "8PSK",
        "16PSK",
        "32PSK",
        "16APSK",
        "32APSK",
        "64APSK",
        "128APSK",
        "16QAM",
        "32QAM",
        "64QAM",
        "128QAM",
        "256QAM",
        "AM-SSB-WC",
        "AM-SSB-SC",
        "AM-DSB-WC",
        "AM-DSB-SC",
        "FM",
        "GMSK",
        "OQPSK",
    ]  # all labels
    mod_target = mods_total
    # init vectors
    cumulants_vec = []
    cum_dx_vec = []
    cum_real_vec = []
    cum_imag_vec = []
    labels = []
    snr_vec = []

    # Loop through all the samples in the dataset
    for idx in range(snr.size):
        # Get the modulation type for the current sample
        modulation_idx = np.nonzero(modulation_onehot[idx])[0][0]
        current_modulation = mods_total[modulation_idx]
        # Skip the sample if it's below the threshold SNR or if the modulation type is not in the target list
        # if snr[idx][0] < -10 or current_modulation not in mod_target:
        #     continue
        if idx % 10000 == 0:
            print(
                "we have reached ",
                idx,
                " current modulation is ",
                current_modulation,
                " SNR=",
                snr[idx][0],
                "dB",
            )
        signal = data[idx]
        real = signal[:, 0]
        imag = signal[:, 1]
        # Calculate the derivative of the signal
        real_dx = np.diff(real)
        imag_dx = np.diff(imag)
        dx = np.transpose(np.array((real_dx, imag_dx)))
        # Calculate the cumulants of the signal and its derivative
        cumulant = cumulant_generation_complex(signal)
        cum_dx = cumulant_generation_complex(dx)
        cum_real = cumulant_generation_real(real)
        cum_imag = cumulant_generation_real(imag)
        # Append the cumulants, labels, and SNRs to their respective lists
        cumulants_vec.append(cumulant)
        cum_dx_vec.append(cum_dx)
        cum_real_vec.append(cum_real)
        cum_imag_vec.append(cum_imag)
        snr_vec.append(snr[idx][0])
        labels.append(current_modulation)
        # var_vec.append(variance)
        # var_vec_dx.append(var_dx)

    return cumulants_vec, cum_dx_vec, cum_real_vec, cum_imag_vec, labels, snr_vec


def cumulant_generation_real(signal):
    # initiate all the needed signals
    signal_squared = pow(signal, 2)
    signal_cubed = pow(signal, 3)
    # get the moments
    M_20 = np.mean(signal_squared, 0)
    M_40 = np.mean(pow(signal, 4), 0)
    M_60 = np.mean(pow(signal, 6), 0)
    M_80 = np.mean(pow(signal, 8), 0)

    # get the cumulants
    cumulant = np.empty(4)
    cumulant[0] = M_20  # c_20
    cumulant[1] = M_40 - 3 * pow(M_20, 2)  # c_40
    cumulant[2] = M_60 - 15 * M_40 * M_20 + 30 * pow(M_20, 3)  # c_60
    cumulant[3] = (
        M_80
        - 28 * M_60 * M_20
        - 35 * pow(M_40, 2)
        + 420 * M_40 * pow(M_20, 2)
        - 630 * pow(M_20, 4)
    )  # c_80

    return cumulant


def cumulant_generation_complex(input):
    # initiate all the needed signals
    real = input[:, 0]
    imag = input[:, 1]
    signal = real + imag * 1j
    signal_conjugate = np.conjugate(signal)
    signal_squared = pow(signal, 2)
    signal_cubed = pow(signal, 3)
    # get the moments
    M_20 = np.mean(signal_squared, 0)
    M_21 = np.mean(signal * signal_conjugate, 0)
    M_40 = np.mean(pow(signal, 4), 0)
    M_41 = np.mean(signal_cubed * signal_conjugate, 0)
    M_42 = np.mean(signal_squared * pow(signal_conjugate, 2), 0)
    M_60 = np.mean(pow(signal, 6), 0)
    M_63 = np.mean(signal_cubed * pow(signal_conjugate, 3), 0)
    M_80 = np.mean(pow(signal, 8), 0)

    # get the cumulants
    cumulant = np.empty(8, dtype=np.complex128)
    cumulant[0] = M_20  # c_20
    cumulant[1] = M_21  # c_21
    cumulant[2] = M_40 - 3 * pow(M_20, 2)  # c_40
    cumulant[3] = M_41 - 3 * M_21 * M_20  # c_41
    cumulant[4] = M_42 - pow(abs(M_20), 2) - 2 * pow(M_21, 2)  # c_42
    cumulant[5] = M_60 - 15 * M_40 * M_20 + 30 * pow(M_20, 3)  # c_60
    cumulant[6] = M_63 - 9 * cumulant[2] * M_21 - 6 * pow(M_21, 3)  # c_63
    cumulant[7] = (
        M_80
        - 28 * M_60 * M_20
        - 35 * pow(M_40, 2)
        + 420 * M_40 * pow(M_20, 2)
        - 630 * pow(M_20, 4)
    )  # c_80

    return cumulant


def save_and_load_list(variable, name, case):
    # save and load using pickle based on what to do (case)
    # case 1 for save my_list
    # case 2 for load file
    path = "data/"
    if case == 1:
        with open(path + name + ".pkl", "wb") as f:
            pickle.dump(variable, f)
    if case == 2:
        with open(path + name + ".pkl", "rb") as f:
            load_variable = pickle.load(f)
        return load_variable


def cumulant_fix_complex(cumulants):
    # Gets the features vector from the cumulants
    features_vec = []
    for ii in range(len(cumulants)):
        cum_abs = abs(cumulants[ii])
        # initiate output
        feats = []
        for jj in range(len(cum_abs)):
            feats.append((cum_abs[jj]))
        features_vec.append(feats)

    return features_vec


def cumulant_fix_iq(cum_real, cum_imag):
    # Gets the features vector from the cumulants
    features_vec = []
    for ii in range(len(cum_real)):
        real_abs = abs(cum_real[ii])
        imag_abs = abs(cum_imag[ii])
        # initiate output
        feats = []
        for jj in range(len(real_abs)):
            feats.append((real_abs[jj]))
            feats.append((imag_abs[jj]))
        features_vec.append(feats)

    return features_vec
