"""
Utilities package for data loading, metrics, visualization, and training.
"""
from .data import get_data_loaders, get_dataset_info
from .metrics import compute_accuracy, compute_ece, compute_calibration_bins, evaluate_model
from .visualization import (
    plot_2d_latent_space,
    plot_tsne,
    plot_calibration_curve,
    plot_training_curves,
    visualize_results
)
from .training import Trainer

__all__ = [
    'get_data_loaders',
    'get_dataset_info',
    'compute_accuracy',
    'compute_ece',
    'compute_calibration_bins',
    'evaluate_model',
    'plot_2d_latent_space',
    'plot_tsne',
    'plot_calibration_curve',
    'plot_training_curves',
    'visualize_results',
    'Trainer',
]
