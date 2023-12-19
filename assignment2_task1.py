import os
import librosa
from sklearn.mixture import GaussianMixture
import numpy as np
from tqdm import tqdm


def extract_MFCC_features(file_path):
    # this function is used to extract MFCC features for fit in GMM，
    # transposing to ensure each MFCC features has a consistent format that aligns with the expectations of GMM
    data, sr = librosa.load(file_path)
    features = librosa.feature.mfcc(y=data, sr=sr).T
    return features

def fit_gmm(features, num_components=13, covariance_type='full', tolerance=0.001):
    """
    This function to fit a GMM to the given features.
    Parameters:
    - features: The input features for training the GMM.
    - num_components: The number of Gaussians/components in the mixture model, we are using MFCC so it was set to be 13.
    - covariance_type: The type of covariance matrix for each component ('full', 'tied', 'diag', 'spherical').
    - tolerance: Convergence tolerance to stop training when the improvement(log-likelihood) is smaller than this value.
    Returns:
    - gmm: The trained Gaussian Mixture Model.
    """
    gmm = GaussianMixture(n_components=num_components, covariance_type=covariance_type, tol=tolerance)
    gmm.fit(features)
    return gmm

def speaker_rec_GMM(audio_dir, test_dir, num_components=13, covariance_type='full', tolerance=0.001):
    # Create and fit one GMM per speaker found from audio_dir
    speaker_GMMs = {}
    speaker_files = {}

    # Populate speaker_files
    # here iterate through the directory and subdirectories to found wav file then extract speaker name from directory path 
    # then append such file path to the list of files
    for root, _, files in os.walk(audio_dir):
        for f in files:
            if f.endswith('.wav'):
                speaker = os.path.basename(root)
                speaker_files[speaker] = speaker_files.get(speaker, [])
                speaker_files[speaker].append(os.path.join(root, f))

    # Create and fit GMMs
    # here concatenate MFCC features for all files associated with each speaker, then fit GMM to those features and store the trained GMM for each
    # these features are concatenated along the 0-axis to form a single feature matrix
    # by this I created GMM models for each speaker for futhur usage
    for speaker, files in tqdm(speaker_files.items(), desc="Fitting GMMs"):
        features = np.concatenate([extract_MFCC_features(file) for file in files], axis=0)
        gmm = fit_gmm(features, num_components=num_components, covariance_type=covariance_type, tolerance=tolerance)
        speaker_GMMs[speaker] = gmm

    # Below start speaker recognition on test data
    # I first extracted MFCC from each test file and initiated variable max_score to track max likelihood while comparing, and a label variable
    # then I used score_samples method of GMM to compare test_features with each trained GMM, give each of test file with highest scored label
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".wav")])
    predict_dict = {}

    for file in tqdm(test_files, desc="Predicting speakers"):
        file_path = os.path.join(test_dir, file)
        test_features = extract_MFCC_features(file_path)

        max_score = float('-inf')
        predicted_label = None

        for label, gmm in speaker_GMMs.items():
            score = np.sum(gmm.score_samples(test_features))
            if score > max_score:
                max_score = score
                predicted_label = label
        
        predict_dict[file] = predicted_label

    return predict_dict


# Example usage
predictions = speaker_rec_GMM('/Users/LeiYi/Desktop/assignment2_data/train_data', '/Users/LeiYi/Desktop/assignment2_data/test_data')
print(predictions)
