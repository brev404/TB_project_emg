"""
Main entry point for EMG analysis program.
"""

# Standard library imports
from pathlib import Path

# Third-party imports
import numpy as np

# Local imports
from data_loader import load_emg_data, save_dataset, load_saved_dataset
from feature_extraction import compute_rms, compute_mnf
from data_validation import get_valid_channels_for_person, zero_invalid_channels
from visualization import create_output_folder, plot_all_rms_values, plot_all_mnf_values
from dataset_creation import create_training_dataset
from utils_data import (
    analyze_and_plot_emg_with_zeros,
    process_raw_data,
    validate_and_zero_channels,
    create_visualizations,
    create_file_prefix
)

def main():
    """Main entry point for the EMG analysis program."""
    # Example usage: Analyze all hand exercises for females
    pre_dict, post_dict, dataset_path, metadata_json, metadata_txt = analyze_and_plot_emg_with_zeros(
        data_type='Hand', 
        exercise_num=1,
        gender='m'
    )
    
    # Load the saved dataset if needed
    if dataset_path:
        loaded_df = load_saved_dataset(dataset_path, metadata_json)
        print("Dataset loaded successfully!")

if __name__ == "__main__":
    main()