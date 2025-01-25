import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(labeled_file, unlabeled_file=None):
    """
    Load and prepare the data for analysis.
    Returns feature columns and separated features/labels.
    """
    # Load labeled data
    df_labeled = pd.read_csv(labeled_file)
    
    # Get feature columns (RMS and MNF columns)
    feature_cols = [col for col in df_labeled.columns 
                   if col.startswith(('rms_', 'mnf_'))]
    
    # Prepare labeled data
    X_labeled = df_labeled[feature_cols].values
    y_labeled = df_labeled['label'].values
    
    # Prepare unlabeled data if provided
    if unlabeled_file:
        df_unlabeled = pd.read_csv(unlabeled_file)
        X_unlabeled = df_unlabeled[feature_cols].values
        return (X_labeled, y_labeled, X_unlabeled, df_labeled, 
                df_unlabeled, feature_cols)
    
    return X_labeled, y_labeled, df_labeled, feature_cols

def train_and_evaluate_svm(X, y, n_folds=5):
    """
    Train SVM with cross-validation and return results.
    """
    # Initialize SVM
    svm = SVC(kernel='rbf', random_state=42)
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Define scoring metrics
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision_weighted',
        'recall': 'recall_weighted',
        'f1': 'f1_weighted'
    }
    
    # Perform cross-validation
    cv_results = cross_validate(svm, X, y, cv=cv, scoring=scoring)
    
    return cv_results

def plot_cv_results(cv_results, output_path):
    """
    Plot cross-validation results.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    means = [cv_results[f'test_{m}'].mean() for m in metrics]
    stds = [cv_results[f'test_{m}'].std() for m in metrics]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, means)
    plt.errorbar(metrics, means, yerr=stds, fmt='none', color='black')
    
    # Add value labels
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.3f}', ha='center', va='bottom')
    
    plt.title('Cross-validation Results')
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path / 'cv_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_path):
    """
    Create and save confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-fatigued', 'Fatigued'],
                yticklabels=['Non-fatigued', 'Fatigued'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_timeline(subject_data, predictions, output_path):
    """
    Create binary timeline visualization for a subject.
    """
    plt.figure(figsize=(15, 5))
    
    # Plot binary predictions
    windows = subject_data['window'].values
    plt.scatter(windows, predictions, 
               c=['green' if p == 0 else 'red' for p in predictions],
               s=100, alpha=0.6)
    
    plt.title(f'Fatigue State Timeline - Subject: {subject_data["subject"].iloc[0]}')
    plt.xlabel('Window Number')
    plt.ylabel('Fatigue State')
    plt.yticks([0, 1], ['Non-fatigued', 'Fatigued'])
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(output_path / f'timeline_{subject_data["subject"].iloc[0]}.png',
                dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    Main function to run the SVM analysis and visualization.
    """
    # Setup paths
    input_path = Path('EMG_Dataset_Split')
    output_path = Path('EMG_Analysis_SVM')
    output_path.mkdir(exist_ok=True)
    
    # Find paired labeled and unlabeled files
    labeled_files = list(input_path.glob('*_labeled.csv'))
    
    for labeled_file in labeled_files:
        print(f"\nProcessing {labeled_file.name}...")
        
        # Get corresponding unlabeled file
        unlabeled_file = labeled_file.parent / labeled_file.name.replace('_labeled.csv', '_unlabeled.csv')
        
        if not unlabeled_file.exists():
            print(f"Warning: No matching unlabeled file found for {labeled_file.name}")
            continue
        
        # Create output subfolder
        dataset_output = output_path / labeled_file.stem
        dataset_output.mkdir(exist_ok=True)
        
        # Load and prepare data
        X_labeled, y_labeled, X_unlabeled, df_labeled, df_unlabeled, feature_cols = \
            load_and_prepare_data(labeled_file, unlabeled_file)
        
        # Scale the data
        scaler = StandardScaler()
        X_labeled_scaled = scaler.fit_transform(X_labeled)
        X_unlabeled_scaled = scaler.transform(X_unlabeled)
        
        # Save scaler
        joblib.dump(scaler, dataset_output / 'scaler.joblib')
        
        # Perform cross-validation
        cv_results = train_and_evaluate_svm(X_labeled_scaled, y_labeled)
        
        # Plot cross-validation results
        plot_cv_results(cv_results, dataset_output)
        
        # Train final model on all labeled data
        final_model = SVC(kernel='rbf', random_state=42)
        final_model.fit(X_labeled_scaled, y_labeled)
        
        # Save model
        joblib.dump(final_model, dataset_output / 'svm_model.joblib')
        
        # Create confusion matrix for labeled data
        y_pred = final_model.predict(X_labeled_scaled)
        plot_confusion_matrix(y_labeled, y_pred, dataset_output)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_labeled, y_pred))
        
        # Create timeline visualizations for unlabeled data
        timeline_path = dataset_output / 'timelines'
        timeline_path.mkdir(exist_ok=True)
        
        # Get binary predictions for unlabeled data
        unlabeled_predictions = final_model.predict(X_unlabeled_scaled)
        
        # Create timeline for each subject
        for subject in df_unlabeled['subject'].unique():
            subject_mask = df_unlabeled['subject'] == subject
            subject_data = df_unlabeled[subject_mask]
            subject_predictions = unlabeled_predictions[subject_mask]
            
            plot_timeline(subject_data, subject_predictions, timeline_path)
        
        print(f"Analysis completed. Results saved in: {dataset_output}")

if __name__ == "__main__":
    main()