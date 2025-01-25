# Standard library
from pathlib import Path

# Third-party
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_features(df, exercise):
    """Get features from all channels, excluding channels 6 and 7 for exercise 1."""
    feature_cols = []
    for i in range(1, 9):  # 8 channels
        if exercise == 'ex1' and i in [6, 7]:
            continue  # Skip channels 6 and 7 for exercise 1
        feature_cols.extend([f'rms_ch{i}', f'mnf_ch{i}'])
    
    X = df[feature_cols].values
    return X, feature_cols

def normalize_features(X):
    """Normalize features using StandardScaler."""
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    return X_norm, scaler

def train_gmm(X, n_components=2):
    """Train GMM model."""
    model = GaussianMixture(n_components=2, random_state=42, n_init=10, covariance_type='full', max_iter=100)

    model.fit(X)
    return model

def plot_clusters(X_original, X_norm, labels, model, scaler, config, save_path):
    """
    Plot clusters using PCA with GMM ellipses.
    Uses normalized data for PCA but can transform back to original scale if needed.
    """
    # Apply PCA to normalized data
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_norm)
    explained_var = pca.explained_variance_ratio_ * 100
    
    # Transform model parameters to PCA space
    means_pca = pca.transform(model.means_)
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot points
    colors = ['green', 'red']
    for i, label in enumerate(['Non-fatigued', 'Fatigued']):
        mask = labels == i
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i], label=label, alpha=0.6)
    
    # Plot GMM ellipses for each cluster
    for i in range(model.n_components):
        # Get covariance matrix in PCA space
        cov_pca = pca.transform(model.covariances_[i].reshape(X_norm.shape[1], -1))
        cov_pca = cov_pca.reshape(2, -1)[:, :2]
        
        # Compute eigenvalues and eigenvectors of the covariance matrix
        evals, evecs = np.linalg.eigh(cov_pca)
        evecs = evecs[0] / np.linalg.norm(evecs[0])
        
        # Compute ellipse parameters
        ell_size = 2 * np.sqrt(2) * np.sqrt(evals)
        ell_angle = np.arctan2(evecs[1], evecs[0]) * 180/np.pi
        
        # Create and add ellipse
        ell = plt.matplotlib.patches.Ellipse(means_pca[i], 
                                           ell_size[0], ell_size[1],
                                           angle=ell_angle+180,
                                           color=colors[i],
                                           fill=False,
                                           linewidth=2)
        plt.gca().add_artist(ell)
        ell.set_alpha(0.5)
    
    plt.xlabel(f'First Principal Component ({explained_var[0]:.1f}% variance)')
    plt.ylabel(f'Second Principal Component ({explained_var[1]:.1f}% variance)')
    
    title = 'EMG Fatigue States Clustering with GMM Ellipses\n'
    title += f"({config['category']}, {config['exercise']}, {config['gender']})"
    plt.title(title)
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(save_path / 'clusters_with_ellipses.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Created cluster visualization with GMM ellipses")

def plot_fatigue_timeline(df, labels, X_original, config, save_path):
    """Plot fatigue state over time windows for each person."""
    print("\nCreating fatigue timelines for each person...")
    
    # Create figure for each person
    for subject in df['subject'].unique():
        # Get person's data
        subject_df = df[df['subject'] == subject].copy()
        subject_df = subject_df.sort_values('window')  # Sort by window number
        subject_labels = labels[subject_df.index]
        
        plt.figure(figsize=(15, 3))
        
        # Plot points
        y_values = [0 if l == 0 else 1 for l in subject_labels]
        
        plt.scatter(subject_df['window'].values, 
                   y_values,
                   c=['green' if l == 0 else 'red' for l in subject_labels],
                   s=50)
        
        plt.ylim(-0.5, 1.5)
        plt.yticks([0, 1], ['Non-fatigued', 'Fatigued'])
        plt.xlabel('Window Number')
        plt.title(f'Fatigue State Timeline - Subject: {subject}\n'
                 f'({config["category"]}, {config["exercise"]}, {config["gender"]})')
        
        plt.savefig(save_path / f'{subject}_fatigue_timeline.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created fatigue timeline for subject: {subject}")

def parse_filename(filename):
    """Parse configuration from filename."""
    base = filename.replace('_processed_emg_dataset.csv', '')
    parts = base.split('_')
    
    config = {
        'filename': filename,
        'category': parts[0],  # 'hand'
        'exercise': parts[1],  # 'ex1'
    }
    
    if len(parts) > 2:
        config['gender'] = parts[2]  # 'f' or 'm'
    else:
        config['gender'] = 'all'
        
    return config

def process_dataset(file_path):
    """Process a single dataset."""
    # Parse configuration
    config = parse_filename(file_path.name)
    print(f"\nProcessing {file_path.name}")
    
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Get features
    X, feature_cols = get_features(df, config['exercise'])
    
    # Normalize features - store scaler for later use
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X)
    print(f"Using {len(feature_cols)} normalized features from all channels")
    
    # Train model
    model = train_gmm(X_norm)
    
    # Get cluster labels
    labels = model.predict(X_norm)
    
    return X, X_norm, labels, config, df, model, scaler

def create_output_folders(base_dir, config):
    """Create output folders for current dataset."""
    # Create base output folder using dataset name
    output_name = f"{config['category']}_{config['exercise']}"
    if config['gender'] != 'all':
        output_name += f"_{config['gender']}"
    
    output_path = Path(base_dir) / output_name
    
    # Create subfolders
    clusters_path = output_path / 'clusters'
    timeline_path = output_path / 'timelines'
    
    # Create all directories
    clusters_path.mkdir(parents=True, exist_ok=True)
    timeline_path.mkdir(parents=True, exist_ok=True)
    
    return output_path, clusters_path, timeline_path

def main():
    # Process all dataset files
    dataset_dir = Path('EMG_Dataset_Processed')
    base_output_dir = Path('EMG_Analysis_Results_GMM')
    
    for file_path in dataset_dir.glob('*_processed_emg_dataset.csv'):
        # Process dataset and get both original and normalized features
        X_original, X_norm, labels, config, df, model, scaler = process_dataset(file_path)
        
        # Create output folders for this dataset
        output_path, clusters_path, timeline_path = create_output_folders(base_output_dir, config)
        
        # Create visualizations using normalized data for clustering
        plot_clusters(X_original, X_norm, labels, model, scaler, config, clusters_path)
        plot_fatigue_timeline(df, labels, X_original, config, timeline_path)
        
        print(f"\nResults saved in: {output_path}")

if __name__ == "__main__":
    main()