# Standard library
import json
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd

def load_emg_data(root_folder='dataset_tb', data_type=None, exercise_num=None, gender=None):
    """
    Load EMG data from all .npy files in the specified folder structure.
    
    Parameters:
    root_folder (str): Path to the dataset folder
    data_type (str, optional): Type of data to load ('Hand' or 'Leg')
    exercise_num (str or int, optional): Specific exercise number to load
    gender (str, optional): Gender to load ('m' or 'f')
    
    Returns:
    dict: Dictionary with metadata and data for each recording
    """
    root_path = Path(root_folder)
    data_dict = {}
    
    # Validate and process data_type
    valid_types = ['Hand', 'Leg']
    if data_type is not None:
        data_type = data_type.capitalize()
        if data_type not in valid_types:
            raise ValueError(f"data_type must be one of {valid_types}")
        categories = [data_type]
    else:
        categories = valid_types
    
    # Convert exercise_num to string if it's provided
    if exercise_num is not None:
        exercise_num = str(exercise_num)
    
    # Convert gender to lowercase if provided
    if gender is not None:
        gender = gender.lower()
        if gender not in ['m', 'f']:
            raise ValueError("Gender must be 'm' or 'f'")
    
    # Iterate through selected categories (Hand and/or Leg)
    for category in categories:
        category_path = root_path / category
        if not category_path.exists():
            continue
            
        # Find all .npy files in the category folder
        for file_path in category_path.glob('*.npy'):
            # Parse filename
            filename = file_path.stem
            parts = filename.split('_')
            if len(parts) != 5:
                continue
                
            name, surname, curr_exercise, side, curr_gender = parts
            
            # Apply filters
            if exercise_num is not None and curr_exercise != exercise_num:
                continue
            if gender is not None and curr_gender != gender:
                continue
            
            # Load the data
            emg_data = np.load(file_path)
            
            # Store in dictionary with metadata
            key = f"{category}_{filename}"
            data_dict[key] = {
                'data': emg_data,
                'category': category,
                'name': name,
                'surname': surname,
                'exercise': curr_exercise,
                'side': side,
                'gender': curr_gender,
                'label': f"{name} {surname} ({curr_gender})"
            }
    
    # Print summary
    if len(data_dict) == 0:
        print(f"No files found matching the criteria:")
        print(f"  Data type: {data_type}")
        print(f"  Exercise: {exercise_num}")
        print(f"  Gender: {gender}")
    else:
        print(f"Loaded {len(data_dict)} files:")
        print(f"  Data type: {data_type if data_type else 'All'}")
        print(f"  Exercise: {exercise_num if exercise_num else 'All'}")
        print(f"  Gender: {gender if gender else 'All'}")
    
    return data_dict

def save_dataset(df, output_folder='EMG_Dataset', prefix=''):
    """
    Save the dataset and detailed channel status information.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the EMG data
    output_folder (str): Folder to save the output files
    prefix (str): Prefix to add to output filenames
    
    Returns:
    tuple: Paths to the saved dataset and metadata files
    """
    folder_path = Path(output_folder)
    folder_path.mkdir(exist_ok=True)
    
    # Save main dataset
    parts = ['emg_dataset']
    if prefix:
        parts.insert(0, prefix)
    filename = '_'.join(parts) + '.csv'
    filepath = folder_path / filename
    
    # Create a copy of the DataFrame without the attrs
    df_to_save = df.copy()
    
    # Save the main dataset
    df_to_save.to_csv(filepath, index=False)
    
    # Save channel status separately as JSON
    if hasattr(df, 'attrs') and 'channel_status' in df.attrs:
        metadata_json = folder_path / f'{prefix}channel_status.json'
        with open(metadata_json, 'w') as f:
            json.dump(df.attrs['channel_status'], f, indent=4)
        
        # Also save a human-readable text version
        metadata_txt = folder_path / f'{prefix}channel_status.txt'
        with open(metadata_txt, 'w') as f:
            f.write("Channel Status by Person:\n")
            f.write("=" * 50 + "\n\n")
            
            for person_id, status in df.attrs['channel_status'].items():
                f.write(f"Subject: {person_id}\n")
                f.write(f"Valid channels: {status['valid_channels']}\n")
                f.write(f"Number of valid channels: {status['n_valid_channels']}\n")
                
                removed_channels = status.get('channels_zeroed_reason', {})
                
                if removed_channels:
                    f.write("Zeroed channels and reasons:\n")
                    for ch, reasons in removed_channels.items():
                        f.write(f"  Channel {int(ch)+1}: {', '.join(reasons)}\n")
                f.write("-" * 30 + "\n\n")
        
        print(f"Dataset saved to: {filepath}")
        print(f"Channel status saved to: {metadata_json} and {metadata_txt}")
        return filepath, metadata_json, metadata_txt
    
    else:
        print(f"Dataset saved to: {filepath}")
        print("No channel status information to save.")
        return filepath, None, None
    
def load_saved_dataset(filepath, metadata_json=None):
    """
    Load a saved dataset and its metadata if available.
    
    Parameters:
    filepath (str or Path): Path to the CSV dataset
    metadata_json (str or Path, optional): Path to the JSON metadata file
    
    Returns:
    pd.DataFrame: Loaded dataset with metadata in attrs if available
    """
    # Load the main dataset
    df = pd.read_csv(filepath)
    
    # Load metadata if available
    if metadata_json is not None:
        metadata_path = Path(metadata_json)
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                channel_status = json.load(f)
            df.attrs['channel_status'] = channel_status
    
    return df