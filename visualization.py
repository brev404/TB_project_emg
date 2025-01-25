# Standard library
from pathlib import Path

# Third-party
import numpy as np
import matplotlib.pyplot as plt

def plot_all_rms_values(all_data_dict, fs, output_folder, data_type=None, exercise_num=None, gender=None):
    """
    Plot RMS values for all persons simultaneously and save to files.
    """
    if not all_data_dict:
        print("No data to plot!")
        return
        
    # Get number of channels from first data entry
    first_key = list(all_data_dict.keys())[0]
    n_channels = all_data_dict[first_key]['rms'].shape[0]
    
    # Create title suffix based on parameters
    title_parts = []
    if data_type:
        title_parts.append(f"Type: {data_type}")
    if exercise_num:
        title_parts.append(f"Exercise: {exercise_num}")
    if gender:
        title_parts.append(f"Gender: {gender}")
    title_suffix = " - " + ", ".join(title_parts) if title_parts else ""
    
    # Plot each channel in a separate figure
    for ch in range(n_channels):
        # Calculate figure size based on number of participants
        n_participants = len(all_data_dict)
        fig_width = 12 + (n_participants * 0.3)
        plt.figure(figsize=(fig_width, 8))
        
        # Plot each person's data for this channel
        for key, data_info in all_data_dict.items():
            rms_values = data_info['rms']
            time = np.arange(rms_values.shape[1])
            #plt.scatter(time, rms_values[ch, :], label=data_info['label'], alpha=0.7, marker='o', s=50)
            plt.plot(time, rms_values[ch, :], label=data_info['label'], alpha=0.7, marker='o')
        
        plt.title(f'Channel {ch+1} - RMS Values{title_suffix}', fontsize=14)
        plt.xlabel('Window Number', fontsize=12)
        plt.ylabel('RMS Amplitude', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Adjust legend position
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        
        # Adjust layout to ensure legend fits
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        # Save the plot
        filename = f'channel_{ch+1}.png'
        filepath = output_folder / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved plot: {filepath}")
        
def plot_all_mnf_values(all_data_dict, fs, output_folder, data_type=None, exercise_num=None, gender=None):
    """
    Plot MNF values for all persons simultaneously and save to files.
    """
    if not all_data_dict:
        print("No data to plot!")
        return
        
    # Get number of channels from first data entry
    first_key = list(all_data_dict.keys())[0]
    n_channels = all_data_dict[first_key]['mnf'].shape[0]
    
    # Create title suffix based on parameters
    title_parts = []
    if data_type:
        title_parts.append(f"Type: {data_type}")
    if exercise_num:
        title_parts.append(f"Exercise: {exercise_num}")
    if gender:
        title_parts.append(f"Gender: {gender}")
    title_suffix = " - " + ", ".join(title_parts) if title_parts else ""
    
    # Plot each channel in a separate figure
    for ch in range(n_channels):
        # Calculate figure size based on number of participants
        n_participants = len(all_data_dict)
        fig_width = 12 + (n_participants * 0.3)
        plt.figure(figsize=(fig_width, 8))
        
        # Plot each person's data for this channel
        for key, data_info in all_data_dict.items():
            mnf_values = data_info['mnf']
            time = np.arange(mnf_values.shape[1])
            plt.plot(time, mnf_values[ch, :], label=data_info['label'], alpha=0.7, marker='o')
        
        plt.title(f'Channel {ch+1} - Mean Frequency Values{title_suffix}', fontsize=14)
        plt.xlabel('Window Number', fontsize=12)
        plt.ylabel('Mean Frequency (Hz)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Adjust legend position
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
        
        # Adjust layout to ensure legend fits
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)
        
        # Save the plot
        filename = f'channel_{ch+1}_mnf.png'
        filepath = output_folder / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved MNF plot: {filepath}")
        
def create_output_folder(data_type=None, exercise_num=None, gender=None, feature_type="RMS"):
    """
    Create an output folder based on the parameters.
    """
    # Create base folder name
    parts = [f'EMG_{feature_type}_Plots']
    
    if data_type:
        parts.append(f"type_{data_type}")
    if exercise_num:
        parts.append(f"ex_{exercise_num}")
    if gender:
        parts.append(f"gender_{gender}")
        
    folder_name = '_'.join(parts)
    
    # Create folder
    folder_path = Path(folder_name)
    folder_path.mkdir(exist_ok=True)
    
    print(f"Created output folder: {folder_path}")
    return folder_path
