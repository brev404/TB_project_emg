# Third-party
import numpy as np
import pandas as pd

# Local
from feature_extraction import compute_rms, compute_mnf
from data_validation import get_valid_channels_for_person, zero_invalid_channels, get_channel_rejection_reason

def create_training_dataset_with_zeros(data_dict, window_size, hop_size, fs):
    """
    Create training dataset with RMS and MNF features, zeroing invalid channels.
    """
    dataset_rows = []
    channel_status = {}
    
    for key, data_info in data_dict.items():
        print(f"Processing {key} for dataset creation...")
        
        # Get raw data
        raw_data = data_info['data']
        
        # Compute features for all channels first
        rms_values = compute_rms(raw_data, window_size, hop_size, fs)
        mnf_values = compute_mnf(raw_data, window_size, hop_size, fs)
        
        # Check which channels are valid based on computed features
        valid_channels = get_valid_channels_for_person(rms_values, mnf_values)
        
        # Zero out invalid channels
        rms_values = zero_invalid_channels(rms_values, valid_channels)
        mnf_values = zero_invalid_channels(mnf_values, valid_channels)
        
        # Store channel status for this person
        person_id = f"{data_info['name']}_{data_info['surname']}"
        channel_status[person_id] = {
            'valid_channels': valid_channels.tolist(),
            'n_valid_channels': np.sum(valid_channels),
            'channels_zeroed_reason': {
                ch: get_channel_rejection_reason(
                    data_info['data'][ch, :],  # Use raw data for reason checking
                    mnf_values[ch, :]
                )
                for ch in range(len(valid_channels)) if not valid_channels[ch]
            }
        }
        
        # Create datapoints for each window
        n_windows = rms_values.shape[1]
        for window_idx in range(n_windows):
            row_data = {
                'subject': person_id,
                'category': data_info['category'],
                'exercise': data_info['exercise'],
                'side': data_info['side'],
                'gender': data_info['gender'],
                'window': window_idx
            }
            
            # Add features for all channels (including zeroed ones)
            for ch in range(len(valid_channels)):
                row_data[f'rms_ch{ch+1}'] = rms_values[ch, window_idx]
                row_data[f'mnf_ch{ch+1}'] = mnf_values[ch, window_idx]
            
            dataset_rows.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(dataset_rows)
    df.attrs['channel_status'] = channel_status
    
    return df

def create_training_dataset(data_dict, window_size, hop_size, fs):
    """
    Create training dataset with RMS and MNF features.
    Invalid channels are zeroed out rather than removed.
    
    Parameters:
    data_dict: Dictionary containing EMG data and metadata
    window_size: Window size in seconds
    hop_size: Hop size in seconds
    fs: Sampling frequency in Hz
    
    Returns:
    pd.DataFrame: Dataset with features and metadata
    """
    dataset_rows = []
    channel_status = {}
    
    for key, data_info in data_dict.items():
        print(f"Processing {key} for dataset creation...")
        
        # Get raw data
        raw_data = data_info['data']
        
        # Compute features for all channels first
        rms_values = compute_rms(raw_data, window_size, hop_size, fs)
        mnf_values = compute_mnf(raw_data, window_size, hop_size, fs)
        
        # Check which channels are valid based on computed features
        valid_channels = get_valid_channels_for_person(rms_values, mnf_values)
        
        # Zero out invalid channels
        zeroed_rms = zero_invalid_channels(rms_values, valid_channels)
        zeroed_mnf = zero_invalid_channels(mnf_values, valid_channels)
        
        # Store channel status for this person
        person_id = f"{data_info['name']}_{data_info['surname']}"
        channel_status[person_id] = {
            'valid_channels': valid_channels.tolist(),
            'n_valid_channels': int(np.sum(valid_channels)),
            'channels_zeroed_reason': {
                str(ch): get_channel_rejection_reason(
                    raw_data[ch, :],
                    mnf_values[ch, :]
                )
                for ch in range(len(valid_channels)) if not valid_channels[ch]
            }
        }
        
        # Create datapoints for each window
        n_windows = zeroed_rms.shape[1]
        for window_idx in range(n_windows):
            row_data = {
                'subject': person_id,
                'category': data_info['category'],
                'exercise': data_info['exercise'],
                'side': data_info['side'],
                'gender': data_info['gender'],
                'window': window_idx
            }
            
            # Add features for all channels (including zeroed ones)
            for ch in range(len(valid_channels)):
                row_data[f'rms_ch{ch+1}'] = zeroed_rms[ch, window_idx]
                row_data[f'mnf_ch{ch+1}'] = zeroed_mnf[ch, window_idx]
                # Add channel validity flag
                row_data[f'ch{ch+1}_valid'] = int(valid_channels[ch])
            
            dataset_rows.append(row_data)
    
    # Create DataFrame
    df = pd.DataFrame(dataset_rows)
    df.attrs['channel_status'] = channel_status
    
    # Add summary of zeroed channels to DataFrame attributes
    df.attrs['zeroed_channels_summary'] = {
        person_id: {
            'n_valid_channels': status['n_valid_channels'],
            'n_zeroed_channels': len(status['channels_zeroed_reason']),
            'zeroed_channel_numbers': [int(ch) + 1 for ch in status['channels_zeroed_reason'].keys()]
        }
        for person_id, status in channel_status.items()
    }
    
    return df
