"""
Visualization utilities for latent space and calibration.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
import torch


def plot_2d_latent_space(features, labels, prototypes=None, save_path=None, title="2D Latent Space"):
    """Plot 2D latent space directly (no dimensionality reduction needed).
    
    Args:
        features: Feature vectors, shape (n_samples, 2)
        labels: Class labels, shape (n_samples,)
        prototypes: Class prototypes, shape (n_classes, 2) [optional]
        save_path: Path to save the figure
        title: Plot title
    """
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    if prototypes is not None and isinstance(prototypes, torch.Tensor):
        prototypes = prototypes.numpy()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot feature points
    scatter = plt.scatter(features[:, 0], features[:, 1], 
                         c=labels, cmap='tab10', 
                         alpha=0.6, s=20, edgecolors='none')
    plt.colorbar(scatter, label='Class')
    
    # Plot prototypes if available
    if prototypes is not None:
        plt.scatter(prototypes[:, 0], prototypes[:, 1],
                   marker='*', s=500, c='red', 
                   edgecolors='black', linewidths=2,
                   label='Class Prototypes', zorder=10)
        
        # Annotate prototypes
        for i, (x, y) in enumerate(prototypes):
            plt.annotate(f'{i}', (x, y), fontsize=12, 
                        ha='center', va='center', color='white', weight='bold')
    
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved 2D latent space plot to {save_path}")
    
    plt.close()


def plot_tsne(features, labels, save_path=None, title="t-SNE Visualization"):
    """Plot t-SNE visualization of high-dimensional features.
    
    Args:
        features: Feature vectors, shape (n_samples, n_features)
        labels: Class labels, shape (n_samples,)
        save_path: Path to save the figure
        title: Plot title
    """
    if isinstance(features, torch.Tensor):
        features = features.numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    # Apply t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1],
                         c=labels, cmap='tab10',
                         alpha=0.6, s=20, edgecolors='none')
    plt.colorbar(scatter, label='Class')
    
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved t-SNE plot to {save_path}")
    
    plt.close()


def plot_calibration_curve(calibration_bins, save_path=None, title="Calibration Curve"):
    """Plot reliability diagram for calibration.
    
    Args:
        calibration_bins: Dictionary from compute_calibration_bins()
        save_path: Path to save the figure
        title: Plot title
    """
    bin_centers = calibration_bins['bin_centers']
    bin_accuracies = calibration_bins['bin_accuracies']
    bin_confidences = calibration_bins['bin_confidences']
    bin_counts = calibration_bins['bin_counts']
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax1.bar(bin_centers, bin_accuracies, width=0.06, alpha=0.7, 
            edgecolor='black', label='Actual Accuracy')
    ax1.plot(bin_centers, bin_confidences, 'ro-', linewidth=2, 
             markersize=8, label='Average Confidence')
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Reliability Diagram', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Sample distribution
    ax2.bar(bin_centers, bin_counts, width=0.06, alpha=0.7, 
            edgecolor='black', color='steelblue')
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Confidence Distribution', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved calibration curve to {save_path}")
    
    plt.close()


def plot_training_curves(train_losses, train_accs, val_accs, save_path=None):
    """Plot training curves.
    
    Args:
        train_losses: List of training losses per epoch
        train_accs: List of training accuracies per epoch
        val_accs: List of validation accuracies per epoch
        save_path: Path to save the figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss curve
    ax1.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, train_accs, 'b-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, val_accs, 'r-', linewidth=2, label='Validation Accuracy')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Accuracy', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    plt.close()


def visualize_results(features, labels, prototypes, calibration_bins, 
                     output_dir, classifier_name, use_tsne=False):
    """Generate all visualizations for evaluation results.
    
    Args:
        features: Feature vectors
        labels: Class labels
        prototypes: Class prototypes (if available)
        calibration_bins: Calibration statistics
        output_dir: Directory to save plots
        classifier_name: Name of classifier for titles
        use_tsne: Whether to use t-SNE for high-dim features
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Latent space visualization
    if features.shape[1] == 2:
        # Direct 2D visualization
        plot_2d_latent_space(
            features, labels, prototypes,
            save_path=f"{output_dir}/{classifier_name}_2d_latent.png",
            title=f"2D Latent Space - {classifier_name}"
        )
    elif use_tsne:
        # t-SNE for high-dimensional features
        plot_tsne(
            features, labels,
            save_path=f"{output_dir}/{classifier_name}_tsne.png",
            title=f"t-SNE Visualization - {classifier_name}"
        )
    
    # Calibration curve
    plot_calibration_curve(
        calibration_bins,
        save_path=f"{output_dir}/{classifier_name}_calibration.png",
        title=f"Calibration Curve - {classifier_name}"
    )
