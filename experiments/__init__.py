"""
Experiment scripts for training and evaluation.
"""
from .train import train_model
from .evaluate import evaluate_trained_model

__all__ = ['train_model', 'evaluate_trained_model']
