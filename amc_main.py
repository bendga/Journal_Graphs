from setup_module import init_setup
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split







samples, samples_dx, sampels_real, samples_imag, labels, snr = init_setup()
print('we are here')

group1 = ['OOK','AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK'] # Low Order and Analog
group2 = ['4ASK','8ASK','16QAM','32QAM','64QAM','128QAM','256QAM'] # ASK and QAM
group3 = ['BPSK','QPSK','8PSK','16PSK','32PSK','16APSK','32APSK','64APSK','128APSK'] # PSK and APSK
groups=[group1,group2,group3]


unique_snr, snr_counts = np.unique(snr, return_counts=True)
unique_labels, label_counts = np.unique(labels, return_counts=True)

X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(samples, labels, range(len(samples)), test_size=0.2, random_state=42, stratify=labels)
snr_test = snr[test_indices]
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
accuracy_list = []
for snr_val in unique_snr:
    mask = snr_test == snr_val
    X_snr = X_test[mask]
    y_snr = y_test[mask]
    y_pred_snr = dt_classifier.predict(X_snr)
    accuracy_snr = accuracy_score(y_snr, y_pred_snr)
    accuracy_list.append(accuracy_snr)
plt.plot(unique_snr, accuracy_list)
plt.xlabel('SNR')
plt.ylabel('Accuracy')
plt.title('Accuracy vs SNR')
plt.savefig('accuracy_vs_snr.png')
unique_labels = np.unique(labels)
num_plots = len(unique_labels) // 8  # Number of plots needed
fig, axs = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, 6*num_plots))

# for i, label_group in enumerate(np.array_split(unique_labels, num_plots)):
for i, label_group in enumerate(groups):
    axs[i].set_title(f'Accuracy vs SNR for Label Group {i+1}')
    for label in label_group:
        mask = y_test == label
        X_label = X_test[mask]
        y_label = y_test[mask]
        snr_label = snr_test[mask]
        accuracy_list = []
        for snr_val in unique_snr:
            mask_snr = (snr_label == snr_val)
            X_snr = X_label[mask_snr]
            y_snr = y_label[mask_snr]
            y_pred_snr = dt_classifier.predict(X_snr)
            accuracy_snr = accuracy_score(y_snr, y_pred_snr)
            accuracy_list.append(accuracy_snr)
        axs[i].plot(unique_snr, accuracy_list, label=label)
    axs[i].set_xlabel('SNR')
    axs[i].set_ylabel('Accuracy')
    axs[i].legend()

plt.tight_layout()
plt.savefig('label_group_accuracy.png')


print('saved plots')





if __name__ == "__main__":
    filepath = "/home/bendegani/AMC_v2/GOLD_XYZ_OSC.0001_1024.hdf5"