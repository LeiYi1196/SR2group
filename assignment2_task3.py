from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
import os
import re
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
from librosa.feature import mfcc
from sklearn.cluster import KMeans


class CustomGMM: 
    """
    Custom Gaussian Mixture Model implementation.
    This class implements the basic functionalities of a GMM using the Expectation-Maximization algorithm.
    """
    def __init__(self, n_components, max_iter=100, tol=1e-3):
        """
        Initialize parameters such as the number of components, maximum iterations, and tolerance for convergence.
        """
        self.n_components = n_components # Number of Gaussian components in the mixture
        self.max_iter = max_iter # Maximum number of iterations for the EM algorithm to prevent overfitting or heavy computation workload
        self.tol = tol # Tolerance to determine convergence
        self.weights = None
        self.means = None
        self.covariances = None
        self.converged = False # Boolean flag to indicate if convergence was reached

    def initialize(self, X):
        """
        Initialize the weights, means, and covariances of the Gaussian mixture components.
        K-Means clustering is used to initialize the means.
        """
        n_samples, n_features = X.shape

        kmeans = KMeans(n_clusters=self.n_components, n_init=10)
        kmeans.fit(X)
        self.means = kmeans.cluster_centers_ # Initialize means using K-Means centroids

        self.weights = np.full(self.n_components, 1 / self.n_components) # Initialize weights to be uniform
        self.covariances = np.array([np.cov(X.T) for _ in range(self.n_components)]) # Initialize covariances

    def e_step(self, X):
        """
        Perform the Expectation (E) step of the EM algorithm.
        Compute the responsibilities (posterior probabilities) of each Gaussian component for each data point.
        """
        n_samples = X.shape[0]
        resp = np.zeros((n_samples, self.n_components))  # Responsibility matrix
        for k in range(self.n_components):
            # Compute the responsibility for each component using the current parameters
            resp[:, k] = self.weights[k] * multivariate_normal.pdf(X, mean=self.means[k], cov=self.covariances[k])
        resp_sum = resp.sum(axis=1)[:, np.newaxis]  # Normalization term (sum over all components)
        resp = resp / resp_sum  # Normalize responsibilities
        return resp

    def m_step(self, X, resp):
        """
        Perform the Maximization (M) step of the EM algorithm.
        Update the weights, means, and covariances of each component based on the computed responsibilities.
        """
        n_samples, n_features = X.shape
        weights = resp.sum(axis=0)  # Sum of responsibilities for each component
        self.weights = weights / n_samples  # Update weights
        self.means = np.dot(resp.T, X) / weights[:, np.newaxis]  # Update means

        for k in range(self.n_components):
            diff = X - self.means[k]  # Difference from mean for each data point
            # Update covariance matrix for each component
            self.covariances[k] = np.dot(resp[:, k] * diff.T, diff) / weights[k]

    def fit(self, X):
        """
        Train the GMM model using the Expectation-Maximization algorithm.
        """
        self.initialize(X)  # Initialize the model parameters
        for _ in range(self.max_iter):
            prev_means = self.means.copy()  # Store the means from the previous iteration
            resp = self.e_step(X)  # E-step: compute responsibilities
            self.m_step(X, resp)  # M-step: update model parameters

            # Check for convergence (if means do not change significantly)
            if np.allclose(prev_means, self.means, atol=self.tol):
                self.converged = True
                break

    def predict(self, X):
        """
        Predict the component index for each data point in X.
        """
        resp = self.e_step(X)  # Compute responsibilities
        return resp.argmax(axis=1)  # Return
    
    def score_samples(self, X):
        """ Compute the log likelihood of each sample under each component """
        log_likelihood = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            log_likelihood[:, k] = multivariate_normal.logpdf(X, mean=self.means[k], cov=self.covariances[k])
        return log_likelihood


def amplify_audio(audio, threshold=0.01, amplify_factor=2.0):
    """ Amplify the audio if the average amplitude is below a threshold. """
    avg_amplitude = np.mean(np.abs(audio))
    if avg_amplitude < threshold:
        return audio * amplify_factor
    return audio

def extract_mfcc(file_path):
    """
    Extract Mel Frequency Cepstral Coefficients (MFCC) features from an audio file.
    """
    sample_rate, audio = wavfile.read(file_path)  # Read the audio file
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max  # Convert audio to float32

    audio = amplify_audio(audio)  # Amplify the audio if necessary

    # Extract MFCC features
    mfcc_features = mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    return mfcc_features.T  # Transpose so that each row is a time frame and each column is an MFCC feature

def train_custom_gmm(audio_dir, n_components=4):
    """
    Train GMM models for each speaker, where each sentence category contributes one component.
    """
    speakers_gmm = {}

    # Iterate over each sentence category
    for category in tqdm(os.listdir(audio_dir), desc="Processing Categories"):
        category_path = os.path.join(audio_dir, category)
        if os.path.isdir(category_path):

            # Iterate over each speaker in the sentence category
            for speaker in os.listdir(category_path):
                speaker_path = os.path.join(category_path, speaker)
                if os.path.isdir(speaker_path):

                    # Collect audio data for different sentences for each speaker
                    if speaker not in speakers_gmm:
                        speakers_gmm[speaker] = []

                    # Iterate over each audio file for the speaker
                    for audio_file in os.listdir(speaker_path):
                        if audio_file.endswith('.wav'):
                            mfccs = extract_mfcc(os.path.join(speaker_path, audio_file))
                            speakers_gmm[speaker].append(mfccs)

    # Train a GMM model for each speaker
    for speaker, data in speakers_gmm.items():
        if data:
            combined_data = np.vstack(data)  # Combine data from all sentence categories
            gmm = CustomGMM(n_components=n_components)
            gmm.fit(combined_data)  # Fit the GMM model
            speakers_gmm[speaker] = gmm  # Store the trained GMM model

    return speakers_gmm


def extract_number(f):
    """
    Extract numbers from file names. Useful for sorting files numerically.
    """
    s = re.findall("\d+", f)
    return int(s[0]) if s else -1

def custom_GMM_multi(audio_dir, test_dir, n_components=1):
    """
    Predict the speaker of each audio file in the test directory using trained GMM models.
    """
    speakers_gmm = train_custom_gmm(audio_dir, n_components)

    test_files = sorted(os.listdir(test_dir), key=extract_number)  # Sort test files

    predict_dict = {}
    # Iterate over each test file
    for test_file in test_files:
        if test_file.endswith('.wav'):
            test_file_path = os.path.join(test_dir, test_file)
            mfccs = extract_mfcc(test_file_path)

            best_log_likelihood, best_speaker = float('-inf'), None
            # Compare against each speaker model
            for speaker, gmm in speakers_gmm.items():
                log_likelihood = gmm.score_samples(mfccs).sum(axis=0)
                if log_likelihood > best_log_likelihood:
                    best_log_likelihood, best_speaker = log_likelihood, speaker
            predict_dict[test_file] = best_speaker  # Predicted speaker for the test file

    return predict_dict

# Example usage
# prediction = custom_GMM_multi('./train_data', './test_data')
# print(prediction)
