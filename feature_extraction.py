# Third-party
import numpy as np
from scipy import signal

def compute_rms(data, window_size, hop_size, fs):
    """
    Compute RMS values for each channel with specified window and hop size.
    
    Parameters:
    data: np.array of shape (channels, samples)
    window_size: window size in seconds
    hop_size: hop size in seconds
    fs: sampling frequency in Hz
    
    Returns:
    rms_values: np.array of shape (channels, n_windows)
    """
    n_channels, n_samples = data.shape
    samples_per_window = int(window_size * fs)
    samples_per_hop = int(hop_size * fs)
    
    # Calculate number of windows
    n_windows = (n_samples - samples_per_window) // samples_per_hop + 1
    
    # Initialize RMS array
    rms_values = np.zeros((n_channels, n_windows))
    
    # Compute RMS for each channel and window
    for ch in range(n_channels):
        for w in range(n_windows):
            start_idx = w * samples_per_hop
            end_idx = start_idx + samples_per_window
            window_data = data[ch, start_idx:end_idx]
            rms_values[ch, w] = np.sqrt(np.mean(window_data**2))
    
    return rms_values

def compute_mnf(data, window_size, hop_size, fs):
    """
    Compute Mean Frequency (MNF) values for each channel with specified window and hop size.
    
    Parameters:
    data: np.array of shape (channels, samples)
    window_size: window size in seconds
    hop_size: hop size in seconds
    fs: sampling frequency in Hz
    
    Returns:
    mnf_values: np.array of shape (channels, n_windows)
    """
    n_channels, n_samples = data.shape
    samples_per_window = int(window_size * fs)
    samples_per_hop = int(hop_size * fs)
    
    # Calculate number of windows
    n_windows = (n_samples - samples_per_window) // samples_per_hop + 1
    
    # Initialize MNF array
    mnf_values = np.zeros((n_channels, n_windows))
    
    # Prepare frequency axis for FFT
    freqs = np.fft.fftfreq(samples_per_window, d=1/fs)[:samples_per_window//2]
    
    # Compute MNF for each channel and window
    for ch in range(n_channels):
        for w in range(n_windows):
            start_idx = w * samples_per_hop
            end_idx = start_idx + samples_per_window
            
            # Get window data
            window_data = data[ch, start_idx:end_idx]
            
            # Apply Hanning window
            window_data = window_data * np.hanning(samples_per_window)
            
            # Compute FFT
            spectrum = np.abs(np.fft.fft(window_data))[:samples_per_window//2]
            
            # Compute power spectrum
            power_spectrum = spectrum ** 2
            
            # Compute MNF
            mnf_values[ch, w] = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
    
    return mnf_values
