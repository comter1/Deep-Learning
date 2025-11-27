"""
Different classifier heads based on various similarity functions.
Updated version with paper-aligned implementations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DotProductClassifier(nn.Module):
    """Standard softmax classifier using dot product similarity."""
    
    def __init__(self, in_features, num_classes):
        super(DotProductClassifier, self).__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x):
        # Standard linear layer: logits = x @ W^T
        logits = F.linear(x, self.weight)
        return logits


class RBFClassifier(nn.Module):
    """RBF-based classifier using radial basis function kernel.
    
    Based on "RBF-Softmax: Learning Deep Representative Prototypes with 
    Radial Basis Function Softmax" (ECCV 2020)
    
    Args:
        in_features: Input feature dimension
        num_classes: Number of classes
        gamma: RBF bandwidth parameter (default: 1.0)
        s: Scale parameter for logits (default: 30.0)
        normalize: Whether to L2-normalize features and prototypes (default: False)
        mode: 'distance' (default, stable) or 'rbf_kernel' (paper version)
    """
    
    def __init__(self, in_features, num_classes, gamma=1.0, s=30.0, 
                 normalize=False, mode='distance'):
        super(RBFClassifier, self).__init__()
        # Learnable class prototypes
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.prototypes)
        
        self.gamma = gamma
        self.s = s
        self.normalize = normalize
        self.mode = mode
        
    def forward(self, x):
        # Optional L2 normalization
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
            prototypes = F.normalize(self.prototypes, p=2, dim=1)
        else:
            prototypes = self.prototypes
        
        # Compute squared Euclidean distances
        # ||x - c||^2 = ||x||^2 + ||c||^2 - 2<x, c>
        x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)
        p_norm_sq = (prototypes ** 2).sum(dim=1)
        distances_sq = x_norm_sq + p_norm_sq - 2 * (x @ prototypes.T)
        
        if self.mode == 'rbf_kernel':
            # Paper version: s * exp(-gamma * distance^2)
            rbf_values = torch.exp(-self.gamma * distances_sq)
            logits = self.s * rbf_values
        else:
            # Default: -gamma * distance^2 (numerically stable)
            logits = -self.gamma * distances_sq
        
        return logits


class AdaptiveRBFClassifier(nn.Module):
    """RBF classifier with learnable bandwidth per class.
    
    Similar to RBFClassifier but each class has its own learnable gamma.
    
    Args:
        in_features: Input feature dimension
        num_classes: Number of classes
        s: Scale parameter for logits (default: 30.0)
        normalize: Whether to L2-normalize features and prototypes (default: False)
        mode: 'distance' (default) or 'rbf_kernel' (paper version)
    """
    
    def __init__(self, in_features, num_classes, s=30.0, 
                 normalize=False, mode='distance'):
        super(AdaptiveRBFClassifier, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.prototypes)
        
        # Learnable log-bandwidth for each class (ensures positive values)
        self.log_gamma = nn.Parameter(torch.zeros(num_classes))
        
        self.s = s
        self.normalize = normalize
        self.mode = mode
        
    def forward(self, x):
        # Optional L2 normalization
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
            prototypes = F.normalize(self.prototypes, p=2, dim=1)
        else:
            prototypes = self.prototypes
        
        # Compute squared distances
        x_norm_sq = (x ** 2).sum(dim=1, keepdim=True)
        p_norm_sq = (prototypes ** 2).sum(dim=1)
        distances_sq = x_norm_sq + p_norm_sq - 2 * (x @ prototypes.T)
        
        # Apply learnable bandwidth per class
        gamma = torch.exp(self.log_gamma)  # Ensure positive
        
        if self.mode == 'rbf_kernel':
            # Paper version
            rbf_values = torch.exp(-(gamma.unsqueeze(0) * distances_sq))
            logits = self.s * rbf_values
        else:
            # Default version
            logits = -(gamma.unsqueeze(0) * distances_sq)
        
        return logits


class MahalanobisClassifier(nn.Module):
    """Mahalanobis distance-based classifier.
    
    Args:
        in_features: Input feature dimension
        num_classes: Number of classes
        s: Scale parameter for logits (default: 30.0)
        normalize: Whether to L2-normalize features (default: False)
        mode: 'distance' (default) or 'exponential' (similar to RBF)
    """
    
    def __init__(self, in_features, num_classes, s=30.0, 
                 normalize=False, mode='distance'):
        super(MahalanobisClassifier, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.prototypes)
        
        # Learn precision matrix (inverse covariance) for each class
        # Use diagonal approximation for simplicity
        self.log_diag = nn.Parameter(torch.zeros(num_classes, in_features))
        
        self.s = s
        self.normalize = normalize
        self.mode = mode
        
    def forward(self, x):
        batch_size = x.size(0)
        num_classes = self.prototypes.size(0)
        
        # Optional normalization
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
            prototypes = F.normalize(self.prototypes, p=2, dim=1)
        else:
            prototypes = self.prototypes
        
        # Compute difference vectors
        x_expanded = x.unsqueeze(1).expand(-1, num_classes, -1)
        diff = x_expanded - prototypes.unsqueeze(0)
        
        # Apply learned scaling (simplified precision matrix - diagonal)
        precision_diag = torch.exp(self.log_diag)
        weighted_diff = diff * precision_diag.unsqueeze(0)
        
        # Mahalanobis distance
        mahalanobis_sq = (weighted_diff * diff).sum(dim=2)
        
        if self.mode == 'exponential':
            # Similar to RBF kernel
            logits = self.s * torch.exp(-mahalanobis_sq)
        else:
            # Default: negative distance
            logits = -mahalanobis_sq
        
        return logits


class CosFaceClassifier(nn.Module):
    """CosFace: cosine similarity with additive margin."""
    
    def __init__(self, in_features, num_classes, s=30.0, m=0.35):
        super(CosFaceClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s  # Scale factor
        self.m = m  # Margin
        
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
    def forward(self, x, labels=None):
        # Normalize features and weights
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        
        # Cosine similarity
        cosine = F.linear(x_norm, w_norm)
        
        # Apply margin during training
        if self.training and labels is not None:
            # Create one-hot labels
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            
            # Add margin to target class
            cosine = cosine - one_hot * self.m
        
        # Scale
        logits = self.s * cosine
        return logits


class ArcFaceClassifier(nn.Module):
    """ArcFace: cosine similarity with angular margin."""
    
    def __init__(self, in_features, num_classes, s=30.0, m=0.5):
        super(ArcFaceClassifier, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s  # Scale factor
        self.m = m  # Angular margin
        
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # For numerical stability
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def forward(self, x, labels=None):
        # Normalize features and weights
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.weight, dim=1)
        
        # Cosine similarity
        cosine = F.linear(x_norm, w_norm)
        
        # Apply angular margin during training
        if self.training and labels is not None:
            # Calculate cos(theta + m)
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            
            # Avoid numerical issues
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
            # Create one-hot labels
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1, 1), 1.0)
            
            # Apply margin to target class
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s
        else:
            output = self.s * cosine
            
        return output


class HybridClassifier(nn.Module):
    """Hybrid classifier combining distance and angular similarity.
    
    Args:
        in_features: Input feature dimension
        num_classes: Number of classes
        alpha: Balance between cosine and distance (default: 0.5)
        gamma: RBF bandwidth for distance term (default: 1.0)
        s: Scale parameter (default: 30.0)
        normalize: Whether to normalize for distance calculation (default: False)
        mode: 'distance' or 'rbf_kernel' for distance term
    """
    
    def __init__(self, in_features, num_classes, alpha=0.5, gamma=1.0, 
                 s=30.0, normalize=False, mode='distance'):
        super(HybridClassifier, self).__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, in_features))
        nn.init.xavier_uniform_(self.prototypes)
        
        self.alpha = alpha  # Balance between cosine and distance
        self.gamma = gamma  # RBF bandwidth
        self.s = s
        self.normalize = normalize
        self.mode = mode
        
    def forward(self, x):
        # Always normalize for cosine similarity
        x_norm = F.normalize(x, dim=1)
        p_norm = F.normalize(self.prototypes, dim=1)
        
        # Cosine similarity
        cosine_sim = F.linear(x_norm, p_norm)
        
        # Euclidean distance (optionally on normalized features)
        if self.normalize:
            # Use normalized features for distance too
            x_dist = x_norm
            p_dist = p_norm
        else:
            # Use original features for distance
            x_dist = x
            p_dist = self.prototypes
        
        x_norm_sq = (x_dist ** 2).sum(dim=1, keepdim=True)
        p_norm_sq = (p_dist ** 2).sum(dim=1)
        distances_sq = x_norm_sq + p_norm_sq - 2 * (x_dist @ p_dist.T)
        
        # Compute distance term
        if self.mode == 'rbf_kernel':
            distance_term = torch.exp(-self.gamma * distances_sq)
        else:
            distance_term = -self.gamma * distances_sq
        
        # Combine both similarities
        # Higher cosine = better, appropriate distance term = better
        if self.mode == 'rbf_kernel':
            # Both terms are positive, can directly combine
            logits = self.alpha * cosine_sim + (1 - self.alpha) * distance_term
            logits = self.s * logits
        else:
            # Cosine is in [-1,1], distance_term is negative
            # Normalize cosine to similar scale
            logits = self.alpha * cosine_sim - (1 - self.alpha) * self.gamma * distances_sq
        
        return logits


def get_classifier(classifier_type, in_features, num_classes, **kwargs):
    """Factory function to get the appropriate classifier.
    
    Args:
        classifier_type: Type of classifier
        in_features: Input feature dimension
        num_classes: Number of classes
        **kwargs: Additional arguments for the classifier
        
    Returns:
        Classifier instance
    """
    classifiers = {
        'dotproduct': DotProductClassifier,
        'rbf': RBFClassifier,
        'cosface': CosFaceClassifier,
        'arcface': ArcFaceClassifier,
        'hybrid': HybridClassifier,
        'adaptive_rbf': AdaptiveRBFClassifier,
        'mahalanobis': MahalanobisClassifier,
    }
    
    if classifier_type not in classifiers:
        raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    return classifiers[classifier_type](in_features, num_classes, **kwargs)