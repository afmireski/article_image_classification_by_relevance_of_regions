import numpy as np
import os

def save_features_to_files(features, files):
    def save(feature, file):
        path = f'features/{file}'
        with open(path, 'wb') as f:
            np.save(f, feature, allow_pickle=True)
        
    for i, file in enumerate(files):
        save(features[i], file)

def load_features_from_files(files):
    def load(file):
        path = f'features/{file}'
        with open(path, 'rb') as f:
            data = np.load(f, allow_pickle=True)
            return data
        
    features = [load(file) for file in files]

    return features

def verify_features_files(files):
    verify_array = [ os.path.exists(f'features/{file}') for file in files ]

    if False in verify_array:
        return False

    return True
    