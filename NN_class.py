import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import LabelEncoder
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn.metrics import confusion_matrix

def create_nn_classifier(input_data, output_data):
    # Create the neural network model
    model = models.Sequential()
    model.add(layers.Dense(16, activation='relu', input_shape=(input_data.shape[1],)))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(np.unique(output_data).shape[0], activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(input_data, output_data, epochs=10, batch_size=32)
    
    return model

def init_classification_NN(samples, labels, snr, name, groups, classifier, train_tresh):
    
    # labeling = 'train_for_'+str(train_tresh)+'_SNR_'+name
    print('change')
    labeling = f"train_for_{train_tresh}_SNR_{name}"
    
    x_train, x_test, y_train, y_test,snr_test = data_spliting(samples,labels,snr,train_tresh)

    # y_train = encode_names(y_train)  ############ NN addition ######################
    # y_test = encode_names(y_test)  ############ NN addition ######################
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y_train)
    encoded_test =  label_encoder.fit_transform(y_test)
    classifier = create_nn_classifier(x_train,encoded_labels)  ############ NN addition ######################

    # classifier = DecisionTreeClassifier(max_depth=8)
    # classifier.fit(x_train, y_train)
    
    os.mkdir(labeling)
    print("Classify per SNR")
    accuracy_data = detection_per_snr_NN(x_test, encoded_test, snr_test, classifier, labeling,label_encoder)
    # accuracy_data =1
    print("Classify per Label")
    detection_per_label_NN(x_test, encoded_test, snr_test, classifier, groups, labeling,label_encoder)
    
    return accuracy_data

def detection_per_snr_NN(x_test, y_test, snr_test, classifier, name, label_encoder):

    unique_snr = np.unique(snr_test)
    accuracy_list = []
    
    for snr_val in unique_snr:
        
        mask = snr_test == snr_val
        x_snr = x_test[mask]
        y_snr = y_test[mask]
        y_pred_snr = classifier.predict(x_snr)
        y_pred_snr = np.argmax(y_pred_snr,1)
        accuracy_snr = accuracy_score(y_snr, y_pred_snr)
        accuracy_list.append(accuracy_snr)
        
        if snr_val in [0,6,10]:
            plot_confusion_matrix(y_snr, y_pred_snr, np.unique(y_snr), snr_val, name)
            
    return {"snr": unique_snr, "accuracy": accuracy_list, "name": name}

def detection_per_label_NN(x_test, y_test, snr_test, classifier, groups, name, label_encoder):
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
                y_pred_snr = label_encoder.inverse_transform(np.argmax(y_pred_snr,1))
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

def data_spliting(samples, labels ,snr , train_tresh):
    
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

name = "cums_dx_NN"
train_tresh = 1
labeling = f"train_for_{train_tresh}_SNR_{name}"
x_train, x_test, y_train, y_test,snr_test = data_spliting(sample_comb1,labels,snr,train_tresh)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(y_train)
encoded_test =  label_encoder.fit_transform(y_test)
classifier = create_nn_classifier(x_train,encoded_labels)  ############ NN addition ######################
print("Classify per SNR")
accuracy_data = detection_per_snr_NN(x_test, encoded_test, snr_test, classifier, labeling,label_encoder)
print("Classify per Label")
detection_per_label_NN(x_test, y_test, snr_test, classifier, groups, labeling,label_encoder)