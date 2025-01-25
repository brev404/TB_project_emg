import pandas as pd
import numpy as np
from pathlib import Path

def replace_invalid_channels_with_mean(df, exercise_type):
    """
    Replace zeroed invalid channels with feature means and remove subjects with all invalid channels.
    Excludes specific channels (6 and 7) for exercise type 1.

    Parameters:
    df (pd.DataFrame): Input DataFrame with EMG data
    exercise_type (int): Exercise type to determine exclusions

    Returns:
    pd.DataFrame: Processed DataFrame with replaced values and filtered subjects
    """
    processed_df = df.copy()
    
    # Define columns and feature pairs
    context_cols = ['subject', 'category', 'exercise', 'side', 'gender', 'window']
    n_channels = 8
    excluded_channels = [6, 7] if exercise_type == 1 else []
    
    # Create list of channels to process (excluding specified channels for exercise 1)
    feature_pairs = [(f'rms_ch{i+1}', f'mnf_ch{i+1}', f'ch{i+1}_valid') 
                     for i in range(n_channels) if (i+1) not in excluded_channels]
    
    # Track subjects with all invalid channels
    subjects_to_remove = set()
    
    # Check each subject for completely invalid channels
    for subject in df['subject'].unique():
        subject_mask = df['subject'] == subject
        subject_data = df[subject_mask]
        
        # Count valid channels for this subject
        valid_channel_count = 0
        total_channels = len(feature_pairs)
        
        for _, _, valid_col in feature_pairs:
            if subject_data[valid_col].iloc[0] == 1:  # Check first row since validity is constant per subject
                valid_channel_count += 1
        
        # If all channels are invalid, mark subject for removal
        if valid_channel_count == 0:
            subjects_to_remove.add(subject)
    
    # Remove subjects with all invalid channels
    if subjects_to_remove:
        print(f"Removing {len(subjects_to_remove)} subjects with all invalid channels: {subjects_to_remove}")
        processed_df = processed_df[~processed_df['subject'].isin(subjects_to_remove)]
    
    # Calculate feature means for replacement
    feature_means = {}
    for rms_col, mnf_col, valid_col in feature_pairs:
        # Calculate means only from valid data
        valid_mask = processed_df[valid_col] == 1
        feature_means[rms_col] = processed_df[valid_mask][rms_col].mean()
        feature_means[mnf_col] = processed_df[valid_mask][mnf_col].mean()
    
    # Replace invalid channel values with means
    for rms_col, mnf_col, valid_col in feature_pairs:
        invalid_mask = processed_df[valid_col] == 0
        if invalid_mask.any():
            processed_df.loc[invalid_mask, rms_col] = feature_means[rms_col]
            processed_df.loc[invalid_mask, mnf_col] = feature_means[mnf_col]
    
    return processed_df

def process_emg_datasets_with_exclusions(input_folder='EMG_Dataset', output_folder='EMG_Dataset_Processed', exercise_type=0):
    """
    Process EMG datasets with new preprocessing requirements.
    
    Parameters:
    input_folder (str): Path to the input folder
    output_folder (str): Path to the output folder
    exercise_type (int): Exercise type to determine exclusions
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Find all CSV files
    csv_files = list(input_path.glob('*_emg_dataset.csv'))
    
    if not csv_files:
        print(f"No EMG dataset files found in {input_folder}")
        return
    
    print(f"Found {len(csv_files)} dataset files to process")
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        
        # Get the output filename
        output_filename = csv_file.name.replace('_emg_dataset.csv', '_processed_emg_dataset.csv')
        output_file = output_path / output_filename
        
        # Load dataset
        df = pd.read_csv(csv_file)
        
        # Process dataset with new requirements
        processed_df = replace_invalid_channels_with_mean(df, exercise_type)
        
        # Save processed dataset
        processed_df.to_csv(output_file, index=False)
        print(f"Processed dataset saved to: {output_file}")

if __name__ == "__main__":
    process_emg_datasets_with_exclusions()