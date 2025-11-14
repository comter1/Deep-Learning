"""
Complete model combining backbone and classifier.
"""
import torch
import torch.nn as nn
from .backbones import get_backbone
from .classifiers import get_classifier


class FullModel(nn.Module):
    """Complete model with backbone and classifier head."""
    
    def __init__(self, dataset_name, classifier_type, latent_dim, num_classes, 
                 in_channels=1, classifier_kwargs=None):
        super(FullModel, self).__init__()
        
        self.dataset_name = dataset_name
        self.classifier_type = classifier_type
        self.latent_dim = latent_dim
        
        # Get backbone
        self.backbone = get_backbone(dataset_name, latent_dim, in_channels)
        
        # Get classifier
        if classifier_kwargs is None:
            classifier_kwargs = {}
        self.classifier = get_classifier(classifier_type, latent_dim, num_classes, 
                                        **classifier_kwargs)
        
        # Track if classifier needs labels (for margin-based methods)
        self.needs_labels = classifier_type in ['cosface', 'arcface']
        
    def forward(self, x, labels=None):
        # Extract features
        features = self.backbone(x)
        
        # Classify
        if self.needs_labels and self.training:
            logits = self.classifier(features, labels)
        else:
            logits = self.classifier(features)
        
        return logits, features
    
    def get_features(self, x):
        """Extract only features without classification."""
        with torch.no_grad():
            features = self.backbone(x)
        return features
    
    def get_prototypes(self):
        """Get class prototypes if available."""
        if hasattr(self.classifier, 'prototypes'):
            return self.classifier.prototypes.data.cpu()
        elif hasattr(self.classifier, 'weight'):
            return self.classifier.weight.data.cpu()
        else:
            return None
