"""
Utility functions for EMG analysis.
"""
# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch

# Local imports
from data_loader import load_emg_data, save_dataset, load_saved_dataset
from feature_extraction import compute_rms, compute_mnf
from data_validation import get_valid_channels_for_person, zero_invalid_channels, get_channel_rejection_reason
from visualization import create_output_folder, plot_all_rms_values, plot_all_mnf_values
from dataset_creation import create_training_dataset


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the EMG data.
    
    Args:
        data (numpy.ndarray): Input signal
        lowcut (float): Lower cutoff frequency in Hz
        highcut (float): Upper cutoff frequency in Hz
        fs (float): Sampling frequency in Hz
        order (int): Filter order
    
    Returns:
        numpy.ndarray: Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def notch_filter(signal, notch_freq, fs, quality_factor=30):
    """
    Apply a notch filter to remove power line interference.
    
    Args:
        signal (numpy.ndarray): Input signal
        notch_freq (float): Notch frequency to remove (e.g., 50/60 Hz)
        fs (float): Sampling frequency in Hz
        quality_factor (float): Quality factor of the notch filter
    
    Returns:
        numpy.ndarray: Filtered signal
    """
    nyquist = 0.5 * fs
    w0 = notch_freq / nyquist
    b, a = iirnotch(w0, quality_factor)
    return filtfilt(b, a, signal)


def preprocess_emg_data(emg_data, fs=512):
    """
    Preprocess EMG data by applying bandpass and notch filters.
    
    Args:
        emg_data (numpy.ndarray): Raw EMG data (channels x samples)
        fs (float): Sampling frequency in Hz
    
    Returns:
        numpy.ndarray: Preprocessed EMG data
    """
    lowcut = 20  # Lower cutoff frequency in Hz
    highcut = 200  # Upper cutoff frequency in Hz
    filtered_data = []

    for ch in range(emg_data.shape[0]):  # Iterate over channels
        filtered_signal = bandpass_filter(emg_data[ch], lowcut, highcut, fs)
        # Optionally add notch filter for power line interference (50/60 Hz)
        # filtered_signal = notch_filter(filtered_signal, 50, fs)  # For 50 Hz
        filtered_data.append(filtered_signal)

    return np.array(filtered_data)


def analyze_and_plot_emg_with_zeros(data_type=None, exercise_num=None, gender=None):
    """
    Analyze EMG data and create plots, replacing invalid channel data with zeros.
    
    Args:
        data_type (str, optional): Type of data to analyze
        exercise_num (int, optional): Exercise number to analyze
        gender (str, optional): Gender filter for data
    
    Returns:
        tuple: Contains pre-validation dict, post-validation dict, dataset path, 
               and metadata files
    """
    # Parameters
    SAMPLING_RATE = 512
    WINDOW_SIZE = 1  # seconds
    HOP_SIZE = 0.5  # seconds
    
    # Load data
    emg_data_dict = load_emg_data(
        data_type=data_type,
        exercise_num=exercise_num,
        gender=gender
    )
    
    if not emg_data_dict:
        print("No data found matching the specified criteria.")
        return None
    
    # Create output folders
    pre_rms_folder = create_output_folder(data_type, exercise_num, gender, "RMS_PreValidation")
    pre_mnf_folder = create_output_folder(data_type, exercise_num, gender, "MNF_PreValidation")
    post_rms_folder = create_output_folder(data_type, exercise_num, gender, "RMS_ZeroedChannels")
    post_mnf_folder = create_output_folder(data_type, exercise_num, gender, "MNF_ZeroedChannels")
    
    # Process and validate data
    pre_validation_dict = process_raw_data(emg_data_dict, WINDOW_SIZE, HOP_SIZE, SAMPLING_RATE)
    post_validation_dict = validate_and_zero_channels(pre_validation_dict)
    
    # Create visualizations
    create_visualizations(pre_validation_dict, post_validation_dict, SAMPLING_RATE,
                         pre_rms_folder, pre_mnf_folder, post_rms_folder, post_mnf_folder,
                         data_type, exercise_num, gender)
    
    # Create and save dataset
    dataset_df = create_training_dataset(post_validation_dict, WINDOW_SIZE, HOP_SIZE, SAMPLING_RATE)
    
    # Create prefix for files
    prefix = create_file_prefix(data_type, exercise_num, gender)
    
    # Save dataset
    dataset_path, metadata_json, metadata_txt = save_dataset(dataset_df, prefix=prefix)
    
    return pre_validation_dict, post_validation_dict, dataset_path, metadata_json, metadata_txt


def process_raw_data(emg_data_dict, window_size, hop_size, sampling_rate):
    """
    Helper function to compute initial features from raw data.
    
    Args:
        emg_data_dict (dict): Dictionary containing raw EMG data
        window_size (float): Window size in seconds
        hop_size (float): Hop size in seconds
        sampling_rate (float): Sampling rate in Hz
    
    Returns:
        dict: Dictionary containing processed EMG data and features
    """
    processed_dict = {}
    for key, data_info in emg_data_dict.items():
        raw_data = data_info['data']
        
        # Apply preprocessing steps (filtering)
        filtered_data = preprocess_emg_data(raw_data, fs=sampling_rate)
        
        # Compute features from filtered data
        rms_values = compute_rms(filtered_data, window_size, hop_size, sampling_rate)
        mnf_values = compute_mnf(filtered_data, window_size, hop_size, sampling_rate)
        
        processed_dict[key] = {
            **data_info,
            'raw_data': raw_data,  # Keep original data
            'filtered_data': filtered_data,  # Store filtered data
            'rms': rms_values,
            'mnf': mnf_values
        }
    return processed_dict


def validate_and_zero_channels(pre_validation_dict):
    """
    Helper function to validate and zero invalid channels.
    
    Args:
        pre_validation_dict (dict): Dictionary containing pre-validation data
        
    Returns:
        dict: Dictionary containing post-validation data with zeroed invalid channels
    """
    post_validation_dict = {}
    for key, data_info in pre_validation_dict.items():
        valid_channels = get_valid_channels_for_person(data_info['rms'], data_info['mnf'])
        
        zeroed_rms = zero_invalid_channels(data_info['rms'], valid_channels)
        zeroed_mnf = zero_invalid_channels(data_info['mnf'], valid_channels)
        
        post_validation_dict[key] = {
            **data_info,
            'rms': zeroed_rms,
            'mnf': zeroed_mnf,
            'valid_channels': valid_channels
        }
    return post_validation_dict


def create_visualizations(pre_dict, post_dict, sampling_rate, 
                         pre_rms_folder, pre_mnf_folder, 
                         post_rms_folder, post_mnf_folder,
                         data_type, exercise_num, gender):
    """
    Helper function to create all visualizations.
    
    Args:
        pre_dict (dict): Pre-validation data dictionary
        post_dict (dict): Post-validation data dictionary
        sampling_rate (float): Sampling rate in Hz
        pre_rms_folder (Path): Folder for pre-validation RMS plots
        pre_mnf_folder (Path): Folder for pre-validation MNF plots
        post_rms_folder (Path): Folder for post-validation RMS plots
        post_mnf_folder (Path): Folder for post-validation MNF plots
        data_type (str): Type of data being analyzed
        exercise_num (int): Exercise number being analyzed
        gender (str): Gender filter being applied
    """
    # Plot pre-validation features
    plot_all_rms_values(pre_dict, sampling_rate, pre_rms_folder, 
                       data_type, exercise_num, gender)
    plot_all_mnf_values(pre_dict, sampling_rate, pre_mnf_folder, 
                       data_type, exercise_num, gender)
    
    # Plot post-validation features
    plot_all_rms_values(post_dict, sampling_rate, post_rms_folder, 
                       data_type, exercise_num, gender)
    plot_all_mnf_values(post_dict, sampling_rate, post_mnf_folder, 
                       data_type, exercise_num, gender)


def create_file_prefix(data_type, exercise_num, gender):
    """
    Helper function to create file prefix.
    
    Args:
        data_type (str): Type of data
        exercise_num (int): Exercise number
        gender (str): Gender filter
        
    Returns:
        str: Formatted file prefix
    """
    prefix_parts = []
    if data_type:
        prefix_parts.append(data_type.lower())
    if exercise_num:
        prefix_parts.append(f'ex{exercise_num}')
    if gender:
        prefix_parts.append(gender.lower())
    return '_'.join(prefix_parts) + '_' if prefix_parts else ''