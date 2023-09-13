import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import confusion_matrix
import seaborn as sns
import h5py
import pandas as pd

# to use with amc_algorithm.ipynb

def start_amc():
    # get the data preprocessed
    dataset = init_setup()
    dataset_prep = prep_data(dataset)
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

    easy_mask = np.isin(dataset_prep['label'], easy_mods)
    labels = dataset_prep['label'][easy_mask]
    snr = dataset_prep['snr'][easy_mask]
    samples = dataset_prep['data'][easy_mask]

    return samples,labels,snr,dataset_prep['features']

def import_data(filpath):
    with h5py.File(filepath, "r") as f:
        data = f["X"]
        modulation_onehot = f["Y"]
        snr = f["Z"]
        # (samples,samples_dx,sample_real,sample_imag,sample_amp,sample_phs,labels,snr,) = process_data(data, modulation_onehot, snr)
        (
            samples,
            samples_dx,
            labels,
            snr,
        ) = process_data(data, modulation_onehot, snr)
    # Save the cumulants, labels, and SNRs to disk
    print("Data generation complete")
    print("Saving.....")
    idx = 1
    save_and_load_list(snr, "snr_vec", idx)
    save_and_load_list(labels, "labels", idx)
    save_and_load_list(samples, "cumulants_vec", idx)
    # save_and_load_list(sample_real, "cum_real_vec", idx)
    # save_and_load_list(sample_imag, "cum_imag_vec", idx)
    save_and_load_list(samples_dx, "cum_dx_vec", idx)
    # save_and_load_list(sample_amp, "cum_amp_vec", idx)
    # save_and_load_list(sample_phs, "cum_phs_vec", idx)
    
    # dataset = {'samples': samples, 'dx': samples_dx, 'real': sample_real, 'imag': sample_imag, 
    #                 'amplitude': sample_amp, 'phase': sample_phs, 'label': labels, 'snr':snr}   
    dataset = {'samples': samples, 'dx': samples_dx, 'label': labels, 'snr':snr}   
    # save_and_load_list(dataset, "dataset", idx)
    print("save is done")
    return dataset

def process_data(data, modulation_onehot, snr):
    
    # receive raw data and gives the cumulants and labels
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
    # cum_real_vec = []
    # cum_imag_vec = []
    # cum_amp_vec = []
    # cum_phs_vec = []
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
        # phase = np.angle(real+1j*imag)
        # amplitude = np.abs(real+1j*imag)
        # Calculate the derivative of the signal
        real_dx = np.diff(real)
        imag_dx = np.diff(imag)
        dx = np.transpose(np.array((real_dx, imag_dx)))
        # Calculate the cumulants of the signal and its derivative
        cumulant = cumulant_generation_complex(signal)
        cum_dx = cumulant_generation_complex(dx)
        # cum_real = cumulant_generation_real(real)
        # cum_imag = cumulant_generation_real(imag)
        # cum_amp = cumulant_generation_real(amplitude)
        # cum_phs = cumulant_generation_real(phase)
        # Append the cumulants, labels, and SNRs to their respective lists
        cumulants_vec.append(cumulant)
        cum_dx_vec.append(cum_dx)
        # cum_real_vec.append(cum_real)
        # cum_imag_vec.append(cum_imag)
        # cum_amp_vec.append(cum_amp)
        # cum_phs_vec.append(cum_phs)
        
        snr_vec.append(snr[idx][0])
        labels.append(current_modulation)
        # var_vec.append(variance)
        # var_vec_dx.append(var_dx)

    # return cumulants_vec, cum_dx_vec, cum_real_vec, cum_imag_vec, cum_amp_vec,cum_phs_vec, labels, snr_vec
    return cumulants_vec, cum_dx_vec, labels, snr_vec

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

def init_setup():
    # initiate the dataset for the algorithm
    if not os.path.exists("data"):
        # Create the folder if it doesn't exist
        os.makedirs("data")
    if not os.listdir("data"):
        filepath = "/home/bendegani/AMC_v2/GOLD_XYZ_OSC.0001_1024.hdf5"
        import_data(filepath)
    idx = 2
    print("loading data")
    # load the data
    samples = save_and_load_list(2, "cumulants_vec", idx)
    samples_dx = save_and_load_list(2, "cum_dx_vec", idx)
    # sample_real = save_and_load_list(2, "cum_real_vec", idx)
    # sample_imag = save_and_load_list(2, "cum_imag_vec", idx)
    # sample_amp = save_and_load_list(2, "cum_amp_vec", idx)
    # sample_phs = save_and_load_list(2, "cum_phs_vec", idx)
    labels = save_and_load_list(2, "labels", idx)
    snr = save_and_load_list(2, "snr_vec", idx)
    # dataset = {'samples': samples, 'dx': samples_dx, 'real': sample_real, 'imag': sample_imag, 
    #             'phase': sample_phs, 'label': labels, 'snr':snr}
    dataset = {'samples': samples, 'dx': samples_dx,'label': labels, 'snr':snr}
    print('finished')
    return dataset

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
    
def prep_data(dataset):
    # get the data ready for classification
    dataset_prep = {}
    dataset_prep['label'] = np.array(dataset['label'])
    dataset_prep['snr'] = np.array(dataset['snr'])
    # samples,feat_labels =cumulant_fix(dataset['samples'],dataset['dx'],dataset['real'],dataset['imag'],dataset['phase'])
    samples,feat_labels =cumulant_fix(dataset['samples'],dataset['dx'])
    dataset_prep['data'] = samples
    dataset_prep['features'] = feat_labels
    return dataset_prep

def cumulant_fix(cum,dx):
# def cumulant_fix(cum,dx,I,Q,phase):
    features_vec = []
    for ii in range(len(cum)):
        cum_abs = abs(cum[ii])
        dx_abs = abs(dx[ii])
        # abs_I = abs(I[ii])
        # abs_Q = abs(Q[ii])
        # cum_ph = abs(phase[ii])
        feats = []
        for jj in range(len(cum_abs)):
            # feats.append(np.log10(cum_abs[jj]/cum_abs[1]))
            feats.append(cum_abs[jj])
            
        for jj in range(2,len(dx_abs)):
            # feats.append(np.log10(dx_abs[jj]))
            feats.append(dx_abs[jj])
            
        # for jj in range(len(abs_I)):
        #     # feats.append(np.log10(abs_I[jj]*0.5 + abs_Q[jj]*0.5))
        #     feats.append(abs_I[jj])
        #     feats.append(abs_Q[jj])
            
        # for jj in range(1,len(cum_ph)):
        #     # feats.append(np.log10(cum_ph[jj]))
        #     feats.append(cum_ph[jj])
        features_vec.append(feats) 
    # feat_labels = ['C20','C21','C40','C41','C42','C60','C63','C80','Cd40','Cd41','Cd42','Cd60','Cd63','Cd80'
    #             ,'Ci2','Cq2','Ci4','Cq4','Ci6','Cq6','Ci8','Cq8','Cp4','Cp6','Cp8']
    feat_labels = ['C20','C21','C40','C41','C42','C60','C63','C80','Cd40','Cd41','Cd42','Cd60','Cd63','Cd80']
    return np.array(features_vec),feat_labels

def init_classification(samples, labels, snr, name, train_tresh, tree_depth):
    # get the accuracy graph per SNR and per label
    labeling = f"Train set SNR>{train_tresh}dB, {name}"
    if train_tresh<0:
        labeling = f"Full Train set, {name}"
    x_train, x_test, y_train, y_test,snr_test = data_spliting(samples, labels, snr, train_tresh)
    classifier = DecisionTreeClassifier(max_depth=tree_depth)
    classifier.fit(x_train, y_train)
    # print(classifier.get_depth())
    try:
        os.mkdir(labeling)
    except FileExistsError:
        print('skip creation')
    print("Classify per SNR")
    accuracy_data = detection_per_snr(x_test, y_test, snr_test, classifier, labeling)
    # accuracy_data =1
    print("Classify per Label")
    # detection_per_label(x_test, y_test, snr_test, classifier, labeling)
    
    return accuracy_data,classifier

def data_spliting(samples, labels ,snr , train_tresh):
    # split data for training using only high SNR
    if train_tresh>0:
        mask = snr>train_tresh
        samples_mask = samples[mask]
        labels_mask = labels[mask]
        snr_mask = snr[mask]
        samples_not = samples[~mask]
        labels_not = labels[~mask]
        x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
        samples_mask,
        labels_mask,
        range(len(samples_mask)),
        test_size=0.3,
        random_state=40,
        stratify=labels_mask,
    )
        x_test=np.concatenate((x_test,samples_not))
        y_test=np.concatenate((y_test,labels_not))
        snr_test = np.concatenate((snr_mask[test_indices],snr[~mask]))
    else:
        x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
        samples,
        labels,
        range(len(samples)),
        test_size=0.2,
        random_state=41,
        stratify=labels,
    )
        snr_test = snr[test_indices]
    return x_train, x_test, y_train, y_test,snr_test

def detection_per_snr(x_test, y_test, snr_test, classifier, name):
    # classification per SNR
    unique_snr = np.unique(snr_test)
    accuracy_list = []
    for snr_val in unique_snr:
        mask = snr_test == snr_val
        x_snr = x_test[mask]
        y_snr = y_test[mask]
        y_pred_snr = classifier.predict(x_snr)
        accuracy_snr = accuracy_score(y_snr, y_pred_snr)
        accuracy_list.append(accuracy_snr)
        if snr_val in range(11):
            plot_confusion_matrix(y_snr, y_pred_snr, np.unique(y_snr), snr_val, name)
    return {"snr": unique_snr, "accuracy": accuracy_list, "name": name}

def detection_per_label(x_test, y_test, snr_test, classifier, name):
    # classify per label
    
    unique_snr = np.unique(snr_test)
    unique_label = np.unique(y_test)
    i=1
    num_plots = i  # Number of plots needed
    fig, axs = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 6 * num_plots))
    axs.set_title(f"Accuracy vs SNR")
    for label in unique_label:
        if label =='others':
            continue
        mask = y_test == label
        x_label = x_test[mask]
        y_label = y_test[mask]
        snr_label = snr_test[mask]
        accuracy_list = []
        for snr_val in unique_snr:
            mask_snr = snr_label == snr_val
            x_snr = x_label[mask_snr]
            y_snr = y_label[mask_snr]
            y_pred_snr = classifier.predict(x_snr)
            accuracy_snr = accuracy_score(y_snr, y_pred_snr)
            accuracy_list.append(accuracy_snr)
        axs.plot(unique_snr, accuracy_list, label=label)
    axs.set_xlabel("SNR")
    axs.set_ylabel("Accuracy")
    axs.legend(loc='lower right')
    axs.yaxis.set_major_locator(MultipleLocator(0.1))
    axs.xaxis.set_major_locator(MultipleLocator(2))
    axs.grid(which='major')
    plt.tight_layout()
    file_path = os.path.join(name, f"label_group_accuracy_{name}.png")
    plt.savefig(file_path)
    print("saved plots")
    plt.clf()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, labels, snr, name):
    # plot confusion matrix for the data
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    accuracy = accuracy_score(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(f"Confusion Matrix for SNR={snr}dB, Acc={accuracy}%")
    file_path = os.path.join(name, f"SNR={snr}.png")
    plt.savefig(file_path)
    plt.clf()
    plt.close()

def combine_accuracy_graphs(*args):
    
    # this take detection probabilities data and put them together on one plot
    # linestyles = ['-', '--',':']  # Different linestyles for each scenario
    linestyles = ['-']
    markes = ['o','s','^','x','*','+']
    fig, ax = plt.subplots()
    for i, accuracy_data in enumerate(args):
        linestyle = linestyles[i % len(linestyles)]
        marker = markes[i % len(markes)]
        ax.plot(
            accuracy_data["snr"], accuracy_data["accuracy"],
            linestyle=linestyle, linewidth=1, marker=marker, markersize=4, label=accuracy_data["name"]
        )
    ax.set_xlabel("SNR")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs SNR")
    ax.legend(loc='lower right')
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.grid(which='major')
    num_files_saved = sum(1 for file in os.listdir('.') if file.startswith('combine_accuracy_graph_'))
    file_path = f'combine_accuracy_graph_{num_files_saved + 1}.png'
    plt.savefig(file_path)
    plt.clf()
    plt.close()

def make_accuracy_table(labels, samples, snr, snr_thresh, feature_list):
    
    # This make an accuracy table of each feature by itself at snr=10dB
    
    unique_label = np.unique(labels)
    accuracy_max = np.zeros((len(samples[0]), len(unique_label)))  # Initialize a matrix to store max accuracies

    for ii in range(len(samples[0])):
        
        feat_list = np.column_stack((samples[:, ii], samples[:, ii]))  # Duplicate each feature column
        x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
            feat_list,
            labels,
            range(len(feat_list)),
            test_size=0.2,
            random_state=40,
            stratify=labels,
        )
        snr_test = snr[test_indices]
        classifier = DecisionTreeClassifier(max_depth=10)
        classifier.fit(x_train, y_train)
        
        for idx, label in enumerate(unique_label):
            mask = y_test == label
            x_label = x_test[mask]
            y_label = y_test[mask]
            snr_label = snr_test[mask]
            mask_snr = snr_label == snr_thresh
            x_snr = x_label[mask_snr]
            y_snr = y_label[mask_snr]
            y_pred_snr = classifier.predict(x_snr)
            accuracy_snr = accuracy_score(y_snr, y_pred_snr)
            accuracy_max[ii, idx] = accuracy_snr  # Store the accuracy in the matrix

    # Create a DataFrame from accuracy_max
    # feature_names = [f"Feature_{ii}" for ii in range(len(samples[0]))]
    label_names = unique_label.tolist()
    accuracy_df = pd.DataFrame(accuracy_max, columns=label_names, index=feature_list)

    # Save the DataFrame to a CSV file
    accuracy_df.to_csv('accuracy_table_snr10.csv')

# split data for training using only high SNR
def threshold_split(samples,labels,snr):
    # get the train set to be only positive SNR
    mask = snr>0
    samples_mask = samples[mask]
    labels_mask = labels[mask]
    snr_mask = snr[mask]
    samples_not = samples[~mask]
    labels_not = labels[~mask]
    x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
        samples_mask,
        labels_mask,
        range(len(samples_mask)),
        test_size=0.3,
        random_state=41,
        stratify=labels_mask,
        )
    x_test=np.concatenate((x_test,samples_not))
    y_test=np.concatenate((y_test,labels_not))
    snr_test = np.concatenate((snr_mask[test_indices],snr[~mask]))
    return x_train, x_test, y_train, y_test,snr_test

def get_accuracy(samples, labels, snr, classifier, proxy_depth):
    # make a confusion matrix and give the detection probability
    unique_snr = np.unique(snr)
    unique_labels = np.unique(labels)
    accuracy_list = []
    for snr_val in unique_snr:
        mask = snr == snr_val
        x_snr = samples[mask]
        y_snr = labels[mask]
        final_labels = classifier.predict(x_snr)
        accuracy_snr = accuracy_score(y_snr, final_labels)
        accuracy_list.append(accuracy_snr)
        if snr_val in [4,8,20]:
            cm = confusion_matrix(y_snr, final_labels, labels=unique_labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_labels, yticklabels=unique_labels)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title(f"Confusion Matrix for SNR={snr_val}dB, Acc={accuracy_snr}%")
            file_path = f"confusion_matrix_SNR={snr_val}.png"
            plt.savefig(file_path)
            plt.clf()
            plt.close()
    accuracy_steps  = {"snr": unique_snr, "accuracy": accuracy_list, "name": f'Our method, proxy depth={proxy_depth}'}
    return accuracy_steps

def label_detection(x_test, y_test, snr_test, name, classifier):
    # shows the detection probability per label
    group1 = ['AM-DSB-SC','AM-SSB-SC','FM','GMSK','BPSK']
    group2 = ['OOK','4ASK','OQPSK','QPSK','8PSK','16QAM']
    groups = [group1, group2]   
    # classify per label
    unique_snr = np.unique(snr_test)
    num_plots = len(groups)  # Number of plots needed
    fig, axs = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 6 * num_plots))
    for i, label_group in enumerate(groups):
        axs[i].set_title(f"Accuracy vs SNR for Label Group {i+1}")
        for label in label_group:
            mask = y_test == label
            x_label = x_test[mask]
            y_label = y_test[mask]
            snr_label = snr_test[mask]
            accuracy_list = []
            for snr_val in unique_snr:
                mask_snr = snr_label == snr_val
                x_snr = x_label[mask_snr]
                y_snr = y_label[mask_snr]
                y_pred_snr = classifier.predict(x_snr)
                accuracy_snr = accuracy_score(y_snr, y_pred_snr)
                accuracy_list.append(accuracy_snr)
            axs[i].plot(unique_snr, accuracy_list, label=label)
        axs[i].set_xlabel("SNR")
        axs[i].set_ylabel("Accuracy")
        axs[i].legend(loc='lower right')
        axs[i].yaxis.set_major_locator(MultipleLocator(0.1))
        axs[i].xaxis.set_major_locator(MultipleLocator(2))
        axs[i].grid(which='major')
    plt.tight_layout()
    file_path =  f"{name}.png"
    plt.savefig(file_path)
    print("saved plots")
    plt.clf()
    plt.close()
    
class SequentialClassifier:
    # the main classifier in my work
    # constructed from steps of regular classifiers in a row
    def __init__(self,groups,feat_pick):
        self.classifiers = []
        self.groups = groups
        self.feat_pick = feat_pick
        
    def add_classifier(self, classifier):
        self.classifiers.append(classifier)
    
    def fit(self, samples, labels):     
        dataset = samples.copy()
        labels_group = labels.copy()
        for idx,group in enumerate(self.groups):
                mask = np.isin(labels_group,group)
                labels_fit = labels_group.copy()
                labels_fit[~mask]='others'
                data = dataset[:,self.feat_pick[idx]]
                self.classifiers[idx].fit(data,labels_fit)
                labels_group = labels_group[~mask]
                dataset = dataset[~mask]

    def predict(self,samples):
        dataset = samples.copy()
        labels_predict = np.full(len(samples),'others', dtype='<U9')
        for idx, group in enumerate(self.groups):
            data = dataset[:,self.feat_pick[idx]]
            step_predict = self.classifiers[idx].predict(data)
            mask = (labels_predict == 'others') & np.isin(step_predict, group)
            labels_predict[mask] = step_predict[mask]
        return labels_predict
    
class DoubleClassifier:        
    # The sub-classifier that helps. contain 2 steps
    # first step is seperating a pair of labels from the rest (based on similar groups)
    # second step is a major classifier between the 2 groups
    
    def __init__(self,depth,label_pair):
        self.classifiers = []
        self.classifiers.append(DecisionTreeClassifier(max_depth=depth))
        self.classifiers.append(DecisionTreeClassifier(max_depth=depth))
        self.groups = label_pair
        # self.feat_pick = feat_pick
        
    def fit(self, samples, labels):     
            dataset = samples.copy()
            labels_group = labels.copy()
            mask = np.isin(labels_group,self.groups)
            labels_group[~mask]='others'
            labels_group[mask]='pair'
            # data = dataset[:,self.feat_pick]
            self.classifiers[0].fit(dataset,labels_group)
            label_svm = labels[mask]
            # label_svm = labels[mask]
            data_svm = dataset[mask]
            self.classifiers[1].fit(data_svm,label_svm)
            print('SVM fit done')

    def predict(self,samples):
        dataset = samples.copy()
        # data = dataset[:,self.feat_pick]
        step_predict = self.classifiers[0].predict(dataset)
        mask = step_predict == 'pair'
        step2_predict = self.classifiers[1].predict(dataset)    
        step_predict[mask] = step2_predict[mask]
        
        return step_predict

def combo_graph(acc_list, x_limits=None, y_limits=None):
    
    # this take detection probabilities data and put them together on one plot
    # linestyles = ['-', '--',':']  # Different linestyles for each scenario
    linestyles = ['-']
    markes = ['o','s','^','x','*','+']
    fig, ax = plt.subplots()
    for i, accuracy_data in enumerate(acc_list):
        linestyle = linestyles[i % len(linestyles)]
        marker = markes[i % len(markes)]
        ax.plot(
            accuracy_data["snr"], accuracy_data["accuracy"],
            linestyle=linestyle, linewidth=1, marker=marker, markersize=4, label=accuracy_data["name"]
        )
    ax.set_xlabel("SNR")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs SNR")
    # Set x-axis limits if provided
    if x_limits:
        plt.xlim(x_limits)
    
    # Set y-axis limits if provided
    if y_limits:
        plt.ylim(y_limits)
    ax.legend(loc='lower right')
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.grid(which='major')
    num_files_saved = sum(1 for file in os.listdir('.') if file.startswith('combine_accuracy_graph_'))
    file_path = f'combine_accuracy_graph_{num_files_saved + 1}.png'
    plt.savefig(file_path)
    plt.clf()
    plt.close()