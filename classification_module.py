import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def init_classification(samples, labels, snr, name):
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
    groups = [group1, group2, group3]
    x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
        samples,
        labels,
        range(len(samples)),
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    snr_test = snr[test_indices]
    classifier = DecisionTreeClassifier()
    classifier.fit(x_train, y_train)
    # y_pred = classifier.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy for all: {accuracy:.2f}")
    detection_per_snr(x_test, y_test, snr_test, classifier, name)
    detection_per_label(x_test, y_test, snr_test, classifier, groups, name)


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
    plt.plot(unique_snr, accuracy_list)
    plt.xlabel("SNR")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy vs SNR for {name}")
    plt.savefig(f"accuracy_vs_SNR_{name}.png")
    plt.clf()


def detection_per_label(x_test, y_test, snr_test, classifier, groups, name):
    unique_snr = np.unique(snr_test)
    unique_labels = np.unique(y_test)
    num_plots = len(unique_labels) // 8  # Number of plots needed
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
                X_snr = x_label[mask_snr]
                y_snr = y_label[mask_snr]
                y_pred_snr = classifier.predict(X_snr)
                accuracy_snr = accuracy_score(y_snr, y_pred_snr)
                accuracy_list.append(accuracy_snr)
            axs[i].plot(unique_snr, accuracy_list, label=label)
        axs[i].set_xlabel("SNR")
        axs[i].set_ylabel("Accuracy")
        axs[i].legend()
    plt.tight_layout()
    plt.savefig(f"label_group_accuracy_{name}.png")
    print("saved plots")
    plt.clf()
