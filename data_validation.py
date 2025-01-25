# Third-party
import numpy as np

def check_feature_variation(feature_values, window_size=10, abs_threshold=0.4, feature_type='rms'):
    """
    Check if feature values have enough variation.
    
    Parameters:
    feature_values: 1D array of RMS or MNF values for one channel
    window_size: number of consecutive values to check for RMS
    abs_threshold: minimum frequency for MNF (Hz), unused for RMS
    feature_type: 'rms' or 'mnf' to determine validation strategy
    
    Returns:
    bool: True if feature shows valid variation, False otherwise
    """
    if feature_type.lower() == 'rms':
        # For RMS check for consecutive static values
        for i in range(len(feature_values) - window_size + 1):
            window = feature_values[i:i+window_size]
            # Check if all values in window are equal
            if np.all(window == window[0]):
                return False
        return True
    
    elif feature_type.lower() == 'mnf':
        for i in range(len(feature_values) - window_size + 1):
            window = feature_values[i:i+window_size]
            # Check if all values in window are equal
            if np.all(window == window[0]):
                return False
        
        # For MNF check if most values are below threshold
        values_below_threshold = np.sum(feature_values < abs_threshold)
        percentage_below = values_below_threshold / len(feature_values)
        return percentage_below < 0.5
    
    else:
        raise ValueError("feature_type must be 'rms' or 'mnf'")

def get_valid_channels_for_person(rms_values, mnf_values):
    """
    Check each channel's validity based on feature variation.
    Returns a boolean mask indicating which channels are valid for this person.
    """
    n_channels = rms_values.shape[0]
    valid_channels = np.ones(n_channels, dtype=bool)
    
    for ch in range(n_channels):
        # Check variation in RMS values
        if not check_feature_variation(rms_values[ch, :], feature_type='rms'):
            valid_channels[ch] = False
            continue
            
        # Check variation in MNF values
        if not check_feature_variation(mnf_values[ch, :], feature_type='mnf'):
            valid_channels[ch] = False
            
    return valid_channels

def get_channel_rejection_reason(rms_values, mnf_values):
    """
    Determine why a channel was rejected.
    """
    reasons = []
    
    # Check RMS variation
    if not check_feature_variation(rms_values, feature_type='rms'):
        reasons.append("static_rms")
    
    # Check MNF variation
    if not check_feature_variation(mnf_values, feature_type='mnf'):
        reasons.append("static_mnf")
    
    return reasons

def zero_invalid_channels(feature_values, valid_channels):
    """
    Replace invalid channel values with zeros instead of removing them.
    
    Parameters:
    feature_values: np.array of shape (channels, windows)
    valid_channels: boolean array indicating valid channels
    
    Returns:
    modified_values: np.array of same shape as feature_values
    """
    modified_values = feature_values.copy()
    modified_values[~valid_channels] = 0
    return modified_values
