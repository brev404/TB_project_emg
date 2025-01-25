import pandas as pd
import numpy as np
from pathlib import Path

def create_split_datasets(input_folder='EMG_Dataset_Processed', output_folder='EMG_Dataset_Split'):
    """
    Create two datasets:
    1. Labeled dataset with only first 20 (non-fatigued) and last 20 (fatigued) windows
    2. Unlabeled dataset with all middle windows
    
    Parameters:
    input_folder (str): Path to the folder containing processed EMG datasets
    output_folder (str): Path to save the split datasets
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True)
    
    # Find all processed CSV files
    csv_files = list(input_path.glob('*_processed_emg_dataset.csv'))
    
    if not csv_files:
        print(f"No processed EMG dataset files found in {input_folder}")
        return
    
    print(f"Found {len(csv_files)} dataset files to process")
    
    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}...")
        
        # Load dataset
        df = pd.read_csv(csv_file)
        
        # Initialize lists for labeled and unlabeled data
        labeled_data = []
        unlabeled_data = []
        
        # Process each subject separately
        for subject in df['subject'].unique():
            subject_data = df[df['subject'] == subject].copy()
            subject_data = subject_data.sort_values('window')
            
            # Get total number of windows for this subject
            total_windows = len(subject_data)
            
            if total_windows < 40:
                print(f"Warning: Subject {subject} has less than 40 windows. Skipping...")
                continue
            
            # Process each window
            for idx, row in subject_data.iterrows():
                window_num = row['window']
                
                # Create feature row (without label initially)
                feature_row = {
                    'subject': subject,
                    'category': row['category'],
                    'exercise': row['exercise'],
                    'side': row['side'],
                    'gender': row['gender'],
                    'window': window_num
                }
                
                # Add EMG features
                for col in df.columns:
                    if col.startswith(('rms_', 'mnf_')):
                        feature_row[col] = row[col]
                
                # Determine which dataset this window belongs to
                if window_num < 20:
                    feature_row['label'] = 0  # Non-fatigued
                    labeled_data.append(feature_row)
                elif window_num >= total_windows - 20:
                    feature_row['label'] = 1  # Fatigued
                    labeled_data.append(feature_row)
                else:
                    unlabeled_data.append(feature_row)
        
        # Create DataFrames
        labeled_df = pd.DataFrame(labeled_data)
        unlabeled_df = pd.DataFrame(unlabeled_data)
        
        # Create output filenames
        base_filename = csv_file.name.replace('_processed_emg_dataset.csv', '')
        labeled_file = output_path / f'{base_filename}_labeled.csv'
        unlabeled_file = output_path / f'{base_filename}_unlabeled.csv'
        
        # Save datasets
        labeled_df.to_csv(labeled_file, index=False)
        unlabeled_df.to_csv(unlabeled_file, index=False)
        
        # Print summary
        print("\nLabeled Dataset Summary:")
        print(f"Total samples: {len(labeled_df)}")
        print(f"Non-fatigued samples: {len(labeled_df[labeled_df['label'] == 0])}")
        print(f"Fatigued samples: {len(labeled_df[labeled_df['label'] == 1])}")
        print(f"Number of subjects: {len(labeled_df['subject'].unique())}")
        print(f"Saved to: {labeled_file}")
        
        print("\nUnlabeled Dataset Summary:")
        print(f"Total samples: {len(unlabeled_df)}")
        print(f"Number of subjects: {len(unlabeled_df['subject'].unique())}")
        print(f"Saved to: {unlabeled_file}")

def main():
    """
    Main function to run the dataset splitting process
    """
    create_split_datasets()
    
if __name__ == "__main__":
    main()