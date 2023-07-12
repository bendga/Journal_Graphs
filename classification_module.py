import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def init_classification(samples, labels, snr, name, groups):
    x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
        samples,
        labels,
        range(len(samples)),
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    snr_test = snr[test_indices]
    classifier = DecisionTreeClassifier(max_depth=8)
    classifier.fit(x_train, y_train)
    # y_pred = classifier.predict(x_test)
    # accuracy = accuracy_score(y_test, y_pred)
    # print(f"Accuracy for all: {accuracy:.2f}")
    # detection_per_snr(x_test, y_test, snr_test, classifier, name)
    os.makedirs("plots", exist_ok=True)
    print("Classify per SNR")
    accuracy_data = detection_per_snr(x_test, y_test, snr_test, classifier, name)
    print("Classify per Label")
    detection_per_label(x_test, y_test, snr_test, classifier, groups, name)

    return accuracy_data


def combine_accuracy_graphs(accuracy_data1, accuracy_data2, accuracy_data3):
    fig, ax = plt.subplots()

    ax.plot(
        accuracy_data1["snr"], accuracy_data1["accuracy"], label=accuracy_data1["name"]
    )
    ax.plot(
        accuracy_data2["snr"], accuracy_data2["accuracy"], label=accuracy_data2["name"]
    )
    ax.plot(
        accuracy_data3["snr"], accuracy_data3["accuracy"], label=accuracy_data3["name"]
    )

    ax.set_xlabel("SNR")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs SNR")
    ax.legend()

    # Specify the folder name
    # Save the figure in the specified folder
    file_path = os.path.join("plots", "combined_accuracy.png")
    plt.savefig(file_path)
    plt.close(fig)


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
    # plt.plot(unique_snr, accuracy_list)
    # plt.xlabel("SNR")
    # plt.ylabel("Accuracy")
    # plt.title(f"Accuracy vs SNR for {name}")
    # file_path = os.path.join('plots', f"accuracy_vs_SNR_{name}.png")
    return {"snr": unique_snr, "accuracy": accuracy_list, "name": name}
    # plt.savefig(file_path)
    # plt.clf()


def detection_per_label(x_test, y_test, snr_test, classifier, groups, name):
    unique_snr = np.unique(snr_test)
    unique_labels = np.unique(y_test)
    # num_plots = len(unique_labels) // 8  # Number of plots needed
    num_plots = 3  # Number of plots needed

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
    plt.tight_layout()
    file_path = os.path.join("plots", f"label_group_accuracy_{name}.png")
    plt.savefig(file_path)
    print("saved plots")
    plt.clf()
