import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns




def init_classification(samples, labels, snr, name, groups, easy_mode, train_tresh):
    
    # labeling = 'train_for_'+str(train_tresh)+'_SNR_'+name
    print('change')
    labeling = f"train_for_{train_tresh}_SNR_{name}"
    x_train, x_test, y_train, y_test,snr_test = data_spliting(samples,labels,snr,train_tresh)
    
    classifier = DecisionTreeClassifier(max_depth=8)
    classifier.fit(x_train, y_train)
    
    os.mkdir(labeling)
    print("Classify per SNR")
    accuracy_data = detection_per_snr(x_test, y_test, snr_test, classifier, labeling)
    
    print("Classify per Label")
    detection_per_label(x_test, y_test, snr_test, classifier, groups, labeling)
    
    if easy_mode == 1:
        plot_average_easy(samples, snr, labels, labeling)
    else: plot_average_hard(samples, snr, labels, groups, labeling)
    
    return accuracy_data

def data_spliting(samples,labels,snr,train_tresh):
    
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
        test_size=0.2,
        random_state=42,
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
        test_size=0.33,
        random_state=42,
        stratify=labels,
    )
        snr_test = snr[test_indices]
    return x_train, x_test, y_train, y_test,snr_test

def combine_accuracy_graphs(*args):
    fig, ax = plt.subplots()

    for accuracy_data in args:
        ax.plot(accuracy_data["snr"], accuracy_data["accuracy"], label=accuracy_data["name"])

    ax.set_xlabel("SNR")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs SNR")
    ax.legend()
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.grid(which='major')

    num_files_saved = sum(1 for file in os.listdir('.') if file.startswith('combine_accuracy_graph_'))
    file_path = f'combine_accuracy_graph_{num_files_saved + 1}.png'
    plt.savefig(file_path)
    plt.clf()
    plt.close()

def detection_per_snr(x_test, y_test, snr_test, classifier, name):
    
    unique_snr = np.unique(snr_test)
    accuracy_list = []
    
    for snr_val in unique_snr:
        
        mask = snr_test == snr_val
        x_snr = x_test[mask]
        y_snr = y_test[mask]
        y_pred_snr = classifier.predict(x_snr)
        accuracy_snr = accuracy_score(y_snr, y_pred_snr)
        accuracy_list.append(accuracy_snr)
        
        if snr_val in [0,6,10]:
            plot_confusion_matrix(y_snr, y_pred_snr, np.unique(y_snr), snr_val, name)
            
    return {"snr": unique_snr, "accuracy": accuracy_list, "name": name}

def detection_per_label(x_test, y_test, snr_test, classifier, groups, name):
    unique_snr = np.unique(snr_test)
    unique_labels = np.unique(y_test)
    # num_plots = len(unique_labels) // 8  # Number of plots needed
    num_plots = len(groups)  # Number of plots needed

    fig, axs = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 6 * num_plots))
    # for i, label_group in enumerate(np.array_split(unique_labels, num_plots)):
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
        axs[i].legend()
        axs[i].yaxis.set_major_locator(MultipleLocator(0.1))
        axs[i].xaxis.set_major_locator(MultipleLocator(2))
        axs[i].grid(which='major')
    plt.tight_layout()
    file_path = os.path.join(name, f"label_group_accuracy_{name}.png")
    plt.savefig(file_path)
    print("saved plots")
    plt.clf()
    plt.close()

def plot_average_easy(samples, snr, labels, name):
    features = samples.shape[1]  # Number of features
    unique_labels = np.unique(labels)
    cmap = plt.get_cmap('tab20')
    num_colors = 20
    for feature_idx in range(features):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.set_xlabel('SNR')
        ax.set_ylabel('Average Value')
        for label in unique_labels:
            mask = labels == label
            average_values = []
            snr_values = np.unique(snr)
            for snr_val in snr_values:
                feature_values = samples[mask & (snr == snr_val), feature_idx]
                # feature_values = np.log10(feature_values)
                average = np.mean(feature_values)
                variance = np.var(feature_values)
                ratio = (variance)/np.power(average,2)
                ratio = np.log10(ratio)
                average_values.append(ratio)
            color = cmap(i % num_colors)
            ax.plot(snr_values, average_values, label=label, color=color)
        ax.set_title(f'Feature {feature_idx + 1}')
        ax.yaxis.set_major_locator(MultipleLocator(0.25))
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(which='major')
        ax.legend()
        file_path = os.path.join(name, f"feature_{feature_idx + 1}.png")
        plt.savefig(file_path)
        print(f"Saved plot {file_path}")
        plt.clf()
        plt.close()

def plot_average_hard(samples, snr, labels, groups, name):
    features = samples.shape[1]  # Number of features
    snr_values = np.unique(snr)

    # os.makedirs("plots", exist_ok=True)

    for feature_idx in range(features):
        fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 18))
        fig.suptitle(f'Average Feature Value vs. SNR for Feature {feature_idx + 1}')

        for i, label_group in enumerate(groups):
            axs[i].set_xlabel('SNR')
            axs[i].set_ylabel('Average Value')

            for label in label_group:
                mask = labels == label
                average_values = []
                for snr_val in snr_values:
                    feature_values = samples[mask & (snr == snr_val), feature_idx]
                    average = np.mean(feature_values)
                    variance = np.var(feature_values)
                    ratio = (variance)/np.power(average,2)
                    # ratio = np.power(average,1)/variance
                    ratio = np.log10(ratio)
                    average_values.append(ratio)

                axs[i].plot(snr_values, average_values, label=label)

            axs[i].set_title(f'Label Group {i + 1}')
            axs[i].legend()

        plt.tight_layout()
        file_path = os.path.join(name, f"feature_{feature_idx + 1}.png")
        plt.savefig(file_path)
        print(f"Saved plot {file_path}")
        plt.clf()
        plt.close()
        
def plot_confusion_matrix(y_true, y_pred, labels, snr, name):
    # os.makedirs("confusion_matrix", exist_ok=True)
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

    return np.array(features_vec)

def cumulant_fix_real(cum_real):
    # Gets the features vector from the cumulants
    features_vec = []
    for ii in range(len(cum_real)):
        real_abs = abs(cum_real[ii])
        # initiate output
        feats = []
        for jj in range(len(real_abs)):
            feats.append((real_abs[jj]))
        features_vec.append(feats)
    return np.array(features_vec)