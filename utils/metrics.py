"""
Evaluation metrics including accuracy and Expected Calibration Error (ECE).
"""
import torch
import torch.nn.functional as F
import numpy as np


def compute_accuracy(outputs, labels):
    """Compute classification accuracy.
    
    Args:
        outputs: Model outputs (logits), shape (batch, num_classes)
        labels: Ground truth labels, shape (batch,)
    
    Returns:
        accuracy: Float between 0 and 1
    """
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def compute_ece(outputs, labels, n_bins=15):
    """Compute Expected Calibration Error (ECE).
    
    Args:
        outputs: Model outputs (logits), shape (batch, num_classes)
        labels: Ground truth labels, shape (batch,)
        n_bins: Number of bins for calibration
    
    Returns:
        ece: Expected calibration error (float)
    """
    # Get predicted probabilities
    softmax_outputs = F.softmax(outputs, dim=1)
    confidences, predictions = torch.max(softmax_outputs, dim=1)
    accuracies = (predictions == labels).float()
    
    # Move to CPU and convert to numpy
    confidences = confidences.cpu().numpy()
    accuracies = accuracies.cpu().numpy()
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def compute_calibration_bins(outputs, labels, n_bins=15):
    """Compute calibration statistics for each bin.
    
    Returns:
        bin_stats: Dictionary with bin statistics for plotting
    """
    # Get predicted probabilities
    softmax_outputs = F.softmax(outputs, dim=1)
    confidences, predictions = torch.max(softmax_outputs, dim=1)
    accuracies = (predictions == labels).float()
    
    # Move to CPU and convert to numpy
    confidences = confidences.cpu().numpy()
    accuracies = accuracies.cpu().numpy()
    
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    bin_stats = {
        'bin_centers': [],
        'bin_accuracies': [],
        'bin_confidences': [],
        'bin_counts': []
    }
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_count = np.sum(in_bin)
        
        if bin_count > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            bin_center = (bin_lower + bin_upper) / 2
        else:
            accuracy_in_bin = 0
            avg_confidence_in_bin = 0
            bin_center = (bin_lower + bin_upper) / 2
        
        bin_stats['bin_centers'].append(bin_center)
        bin_stats['bin_accuracies'].append(accuracy_in_bin)
        bin_stats['bin_confidences'].append(avg_confidence_in_bin)
        bin_stats['bin_counts'].append(bin_count)
    
    return bin_stats


@torch.no_grad()
def evaluate_model(model, data_loader, device):
    """Evaluate model on a dataset.
    
    Returns:
        results: Dictionary with accuracy, ECE, and raw outputs
    """
    model.eval()
    
    all_outputs = []
    all_labels = []
    all_features = []
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs, features = model(images)
        
        all_outputs.append(outputs)
        all_labels.append(labels)
        all_features.append(features)
    
    # Concatenate all batches
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_features = torch.cat(all_features, dim=0)
    
    # Compute metrics
    accuracy = compute_accuracy(all_outputs, all_labels)
    ece = compute_ece(all_outputs, all_labels)
    calibration_bins = compute_calibration_bins(all_outputs, all_labels)
    
    results = {
        'accuracy': accuracy,
        'ece': ece,
        'calibration_bins': calibration_bins,
        'outputs': all_outputs.cpu(),
        'labels': all_labels.cpu(),
        'features': all_features.cpu(),
    }
    
    return results
