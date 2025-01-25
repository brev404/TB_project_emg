"""
Utility functions for EMG analysis.
"""
# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np

# Local imports
from data_loader import load_emg_data, save_dataset, load_saved_dataset
from feature_extraction import compute_rms, compute_mnf
from data_validation import get_valid_channels_for_person, zero_invalid_channels, get_channel_rejection_reason
from visualization import create_output_folder, plot_all_rms_values, plot_all_mnf_values
from dataset_creation import create_training_dataset


def analyze_and_plot_emg_with_zeros(data_type=None, exercise_num=None, gender=None):
    """
    Analyze EMG data and create plots, replacing invalid channel data with zeros.
    """
    # Parameters
    SAMPLING_RATE = 512
    WINDOW_SIZE = 1  # seconds
    HOP_SIZE = 0.5    # seconds
    
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
    """Helper function to compute initial features from raw data."""
    processed_dict = {}
    for key, data_info in emg_data_dict.items():
        raw_data = data_info['data']
        rms_values = compute_rms(raw_data, window_size, hop_size, sampling_rate)
        mnf_values = compute_mnf(raw_data, window_size, hop_size, sampling_rate)
        
        processed_dict[key] = {
            **data_info,
            'rms': rms_values,
            'mnf': mnf_values
        }
    return processed_dict

def validate_and_zero_channels(pre_validation_dict):
    """Helper function to validate and zero invalid channels."""
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
    """Helper function to create all visualizations."""
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
    """Helper function to create file prefix."""
    prefix_parts = []
    if data_type:
        prefix_parts.append(data_type.lower())
    if exercise_num:
        prefix_parts.append(f'ex{exercise_num}')
    if gender:
        prefix_parts.append(gender.lower())
    return '_'.join(prefix_parts) + '_' if prefix_parts else ''
