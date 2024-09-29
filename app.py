from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from collections import Counter
import math
from scipy.stats import skew, kurtosis, chisquare, kstest, entropy as scipy_entropy
from numpy.fft import fft
import itertools
import gzip

flask_app = Flask(__name__)

# Load your machine learning model (ensure 'model.pkl' is in the same directory)
model = pickle.load(open("model.pkl", "rb"))

# Helper Functions

def hex_to_bytes(hex_str):
    """Convert hexadecimal string to bytes."""
    return bytes.fromhex(hex_str)

def bytes_to_integers(data):
    """Convert bytes to a NumPy array of integers."""
    return np.array([int(b) for b in data], dtype=np.int64)

def calculate_entropy(data):
    """Calculate the Shannon entropy of the data."""
    if len(data) == 0:
        return 0
    entropy = 0
    data_len = len(data)
    counter = Counter(data)
    for count in counter.values():
        probability = count / data_len
        entropy -= probability * math.log2(probability)
    return entropy

def extract_iv_and_infer_mode(ciphertext_hex, features, block_size=16):
    """Extract IV and infer mode of operation from ciphertext."""
    ciphertext_bytes = hex_to_bytes(ciphertext_hex)
    iv = ciphertext_bytes[:block_size]
    features['iv'] = iv.hex()  # Store IV as hexadecimal string

    if len(ciphertext_bytes) % block_size != 0:
        features['mode'] = 'Unknown or Stream Cipher'
    else:
        blocks = [ciphertext_bytes[i:i + block_size] for i in range(0, len(ciphertext_bytes), block_size)]
        if len(blocks) != len(set(blocks)):
            features['mode'] = 'ECB'
        else:
            features['mode'] = 'CBC or other block mode'

    return features

def byte_value_range(data):
    """Compute the range of byte values."""
    return np.ptp(data)

def mode_of_byte_values(data):
    """Compute the mode of byte values."""
    return Counter(data).most_common(1)[0][0]

def frequency_of_mode_byte_value(data):
    """Compute the frequency of the mode byte value."""
    return Counter(data).most_common(1)[0][1] / len(data)

def byte_value_histogram(data, bins=256):
    """Compute histogram of byte values."""
    hist, _ = np.histogram(data, bins=bins, range=(0, 255))
    return hist.tolist()

def byte_value_percentiles(data):
    """Compute percentiles of byte values."""
    return np.percentile(data, [25, 50, 75]).tolist()

def entropy_of_byte_value_differences(data):
    """Compute entropy of differences between consecutive byte values."""
    differences = np.diff(data)
    return calculate_entropy(differences)

def frequency_of_byte_value_differences(data):
    """Compute frequency of differences between consecutive byte values."""
    differences = np.diff(data)
    return dict(Counter(differences))

def longest_increasing_subsequence(data):
    """Compute the length of the longest increasing subsequence."""
    n = len(data)
    if n == 0:
        return 0
    lengths = [1] * n
    for i in range(1, n):
        for j in range(i):
            if data[i] > data[j] and lengths[i] < lengths[j] + 1:
                lengths[i] = lengths[j] + 1
    return max(lengths)

def longest_decreasing_subsequence(data):
    """Compute the length of the longest decreasing subsequence."""
    return longest_increasing_subsequence([-x for x in data])

def run_length_encoding(data):
    """Perform run-length encoding on the data."""
    return [(len(list(group)), name) for name, group in itertools.groupby(data)]

def byte_value_transition_matrix(data):
    """Compute the byte value transition matrix."""
    matrix = np.zeros((256, 256), dtype=int)
    for i in range(len(data) - 1):
        matrix[data[i]][data[i+1]] += 1
    return matrix.tolist()

def frequency_of_byte_value_n_grams(data, n):
    """Compute frequency of n-grams in byte values."""
    n_grams = zip(*[data[i:] for i in range(n)])
    return dict(Counter(n_grams))

def entropy_of_byte_value_n_grams(data, n):
    """Compute entropy of n-grams in byte values."""
    n_gram_freq = frequency_of_byte_value_n_grams(data, n)
    return scipy_entropy(list(n_gram_freq.values()))

def byte_value_autocorrelation_function(data, nlags=50):
    """Compute autocorrelation function of byte values."""
    data = data - np.mean(data)
    result = np.correlate(data, data, mode='full')
    result = result[result.size//2:]
    return result[:nlags].tolist()

def byte_value_power_spectrum(data):
    """Compute power spectrum of byte values."""
    return np.abs(np.fft.fft(data))**2

# Additional Feature Extraction Functions

def byte_statistics(data):
    """Compute basic byte statistics."""
    byte_values = bytes_to_integers(data)
    stats = {
        'mean_byte_value': np.mean(byte_values),
        'median_byte_value': np.median(byte_values),
        'variance_byte_value': np.var(byte_values),
        'std_dev_byte_value': np.std(byte_values),
        'skewness_byte_value': skew(byte_values),
        'kurtosis_byte_value': kurtosis(byte_values),
    }
    return stats

def frequency_statistics(data):
    """Compute frequency distribution statistics."""
    freq_dist = Counter(data)
    freqs = np.array(list(freq_dist.values()))
    stats = {
        'max_byte_freq': np.max(freqs),
        'min_byte_freq': np.min(freqs),
        'range_byte_freq': np.max(freqs) - np.min(freqs),
        'std_dev_byte_freq': np.std(freqs),
        'entropy_byte_freq': scipy_entropy(list(freqs)) if len(freqs) > 0 else 0
    }
    return stats

def ngram_statistics(data, n=2):
    """Compute n-gram statistics."""
    ngrams = Counter([tuple(data[i:i+n]) for i in range(len(data)-n+1)])
    freqs = np.array(list(ngrams.values()))
    stats = {
        f'{n}gram_max_freq': np.max(freqs) if len(freqs) > 0 else 0,
        f'{n}gram_min_freq': np.min(freqs) if len(freqs) > 0 else 0,
        f'{n}gram_range_freq': (np.max(freqs) - np.min(freqs)) if len(freqs) > 0 else 0,
        f'{n}gram_std_dev_freq': np.std(freqs) if len(freqs) > 0 else 0,
        f'{n}gram_entropy_freq': scipy_entropy(list(freqs)) if len(freqs) > 0 else 0
    }
    return stats

def fft_statistics(data):
    """Compute FFT statistics."""
    byte_values = bytes_to_integers(data).astype(np.float64)
    if byte_values.size == 0:
        return {
            'fft_mean_magnitude': 0,
            'fft_std_dev_magnitude': 0,
            'fft_max_magnitude': 0,
            'fft_min_magnitude': 0,
            'fft_median_magnitude': 0,
        }
    fft_vals = np.abs(fft(byte_values))
    return {
        'fft_mean_magnitude': np.mean(fft_vals),
        'fft_std_dev_magnitude': np.std(fft_vals),
        'fft_max_magnitude': np.max(fft_vals),
        'fft_min_magnitude': np.min(fft_vals),
        'fft_median_magnitude': np.median(fft_vals),
    }

def compression_ratio(data):
    """Compute compression ratio using gzip."""
    try:
        compressed_data = gzip.compress(data)
        ratio = len(compressed_data) / len(data) if len(data) > 0 else 0
    except Exception:
        ratio = 0
    return {'compression_ratio': ratio}

def serial_correlation(data):
    """Compute serial correlation of byte values."""
    byte_values = bytes_to_integers(data)
    if len(byte_values) < 2:
        return 0
    return np.corrcoef(byte_values[:-1], byte_values[1:])[0, 1]

def printable_ascii_percentage(data):
    """Compute the percentage of printable ASCII characters."""
    printable = sum(32 <= byte <= 126 for byte in data)
    return printable / len(data) if len(data) > 0 else 0

def average_hamming_weight(data):
    """Compute average Hamming weight of byte values."""
    if len(data) == 0:
        return 0
    hamming_weight = sum(bin(byte).count('1') for byte in data)
    return hamming_weight / len(data)
def autocorrelation(data, lag=1):
    n = len(data)
    if n < lag + 1:
        return 0
    data_mean = np.mean(data)
    data_variance = np.var(data)
    acf = np.sum((data[:n - lag] - data_mean) * (data[lag:] - data_mean)) / ((n - lag) * data_variance)
    return acf
def runs_test(data):
    """Compute runs test statistic for randomness."""
    if len(data) == 0:
        return 0
    runs = 1
    for i in range(1, len(data)):
        if data[i] != data[i-1]:
            runs += 1
    return runs

def chi_square_test(data):
    """Compute Chi-square test statistic."""
    byte_values = bytes_to_integers(data)
    if len(byte_values) == 0:
        return {'chi_square_stat': 0, 'chi_square_p_value': 1}
    observed_freq = np.array(list(Counter(byte_values).values()))
    expected_freq = np.full_like(observed_freq, np.mean(observed_freq))
    if np.any(expected_freq == 0):
        expected_freq = observed_freq.copy()
    chi_stat, p_value = chisquare(observed_freq, expected_freq)
    return {'chi_square_stat': chi_stat, 'chi_square_p_value': p_value}

def ks_test(data):
    """Compute Kolmogorov-Smirnov test statistic."""
    byte_values = bytes_to_integers(data)
    if len(byte_values) == 0:
        return {'ks_statistic': 0, 'ks_p_value': 1}
    statistic, p_value = kstest(byte_values, 'norm')
    return {'ks_statistic': statistic, 'ks_p_value': p_value}

# Comprehensive Feature Extraction Function

def extract_features(ciphertext_hex, features):
    """Extract all features from the given ciphertext in hexadecimal form."""
    ciphertext_bytes = hex_to_bytes(ciphertext_hex)
    byte_values = bytes_to_integers(ciphertext_bytes)

    # Basic Features
    features['length'] = len(ciphertext_bytes)
    features['entropy'] = calculate_entropy(ciphertext_bytes)
    byte_stats = byte_statistics(ciphertext_bytes)
    features.update(byte_stats)
    freq_stats = frequency_statistics(ciphertext_bytes)
    features.update(freq_stats)
    fft_stats = fft_statistics(ciphertext_bytes)
    features.update(fft_stats)
    autocorr_1 = autocorrelation(ciphertext_bytes, lag=1)
    autocorr_5 = autocorrelation(ciphertext_bytes, lag=5)
    features['autocorrelation_lag_1'] = autocorr_1
    features['autocorrelation_lag_5'] = autocorr_5
    chi_square_stats = chi_square_test(ciphertext_bytes)
    features.update(chi_square_stats)
    ks_stats = ks_test(ciphertext_bytes)
    features.update(ks_stats)
    comp_ratio = compression_ratio(ciphertext_bytes)
    features.update(comp_ratio)
    features['serial_correlation'] = serial_correlation(ciphertext_bytes)
    features['printable_ascii_percentage'] = printable_ascii_percentage(ciphertext_bytes)
    features['avg_hamming_weight'] = average_hamming_weight(ciphertext_bytes)
    features['runs_test'] = runs_test(ciphertext_bytes)

    # Additional Features
    features['byte_value_range'] = byte_value_range(byte_values)
    features['mode_byte_value'] = mode_of_byte_values(byte_values)
    features['freq_mode_byte_value'] = frequency_of_mode_byte_value(byte_values)
    features['byte_value_histogram'] = byte_value_histogram(byte_values)
    features['byte_value_percentiles'] = byte_value_percentiles(byte_values)
    features['entropy_byte_value_diff'] = entropy_of_byte_value_differences(byte_values)
    features['freq_byte_value_diff'] = frequency_of_byte_value_differences(byte_values)
    features['longest_increasing_subseq'] = longest_increasing_subsequence(byte_values)
    features['longest_decreasing_subseq'] = longest_decreasing_subsequence(byte_values)
    features['run_length_encoding'] = run_length_encoding(byte_values)
    features['byte_value_transition_matrix'] = byte_value_transition_matrix(byte_values)

    for n in [2, 3, 4]:
        freq_ngrams = frequency_of_byte_value_n_grams(byte_values, n)
        entropy_ngrams = entropy_of_byte_value_n_grams(byte_values, n)
        features[f'freq_byte_value_{n}grams'] = freq_ngrams
        features[f'entropy_byte_value_{n}grams'] = entropy_ngrams

    features['byte_value_acf'] = byte_value_autocorrelation_function(byte_values)
    features['byte_value_power_spectrum'] = byte_value_power_spectrum(byte_values).tolist()

    return features

# Flask Routes

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route('/predict', methods=['POST'])
def predict():
    # Get the hexadecimal data from the form
    hex_data = request.form.get('hexa')
    
    if hex_data is None or hex_data.strip() == "":
        return jsonify({'error': 'No hex_data provided'}), 400
    
    try:
        # Clean the input by removing any whitespace
        hex_data = ''.join(hex_data.split())
        print(hex_data)
        # Step 1: Extract features from the hexadecimal data
        features = {}
        features = extract_features(hex_data, features)
        features = extract_iv_and_infer_mode(hex_data, features)
        print(features)
        # Convert features to DataFrame for prediction
        features_df = pd.DataFrame([features])
        print(len(features_df.iloc[0]))
        # Handle features that are dictionaries or lists
        # For example, 'byte_value_transition_matrix' and 'byte_value_power_spectrum' are lists
        # Depending on your model, you might need to flatten these or handle them appropriately
        # Here, we'll convert lists to strings, but you should adjust based on your model's expectations
        for column in features_df.columns:
            if isinstance(features_df[column].iloc[0], list) or isinstance(features_df[column].iloc[0], dict):
                features_df[column] = features_df[column].apply(lambda x: str(x))
        
        # Optional: If your model expects specific preprocessing (like label encoding for 'mode'), apply it here
        # Example:
        # if 'mode' in features_df.columns:
        #     label_encoder = LabelEncoder()
        #     features_df['mode'] = label_encoder.fit_transform(features_df['mode'])
        
        # Step 2: Make predictions using your ML model
        prediction = model.predict(features_df)
        
        # Step 3: Return the prediction result to the user
        return render_template("index.html", prediction_text=f"The predicted algorithm is: {prediction[0]}")
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    flask_app.run(debug=True, port=5001)
