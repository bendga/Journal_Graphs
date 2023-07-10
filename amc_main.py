from setup_module import init_setup
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split







samples,samples_wth_dx,labels,snr = init_setup()
print('we are here')
unique_snr, snr_counts = np.unique(snr, return_counts=True)
unique_labels, label_counts = np.unique(labels, return_counts=True)








if __name__ == "__main__":
    filepath = "/home/bendegani/AMC_v2/GOLD_XYZ_OSC.0001_1024.hdf5"