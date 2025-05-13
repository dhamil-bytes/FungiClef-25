"""
Diane Hamilton

fusion_model
    fuses a resnet50 and a basic mlp for some topk features :)
"""

import torch
import torch.nn as nn
from constants import *

class ProtoNetModel(nn.Module):
    def __init__(self, image_encoder, tabular_encoder, prototype_dim=128):
        super(ProtoNetModel, self).__init__()
        self.image_encoder = image_encoder
        self.tabular_encoder = tabular_encoder
        # To project combined features into prototype space
        self.fc = nn.Linear(output_dim * 2, prototype_dim)
        self.prototype = {}

    def fuse_data(self, image_data, tabular_data):
        # Exetract image and tabular features
        image_features = self.image_encoder(image_data)
        tabular_features = self.tabular_encoder(tabular_data)
        
        # Combine the features
        combined_features = torch.cat((image_features, tabular_features), dim=1)
        return combined_features

    # calc logits and features
    def forward(self, image_data, tabular_data):
        combined_features = self.fuse_data(image_data, tabular_data)
        
        # Project into the prototype space
        projected_features = self.fc(combined_features)
        return projected_features

    # prototypes calculated at inference time btw
    def calculate_prototypes(self, support_batch_labels, support_batch_examples):
        """
        support_batch_labels: A tensor of shape [batch_size] representing class labels for each feature
        support_batch_examples: batch of coupled image and tabular data already fused
        """

        prototypes = {}

        unique_labels = torch.unique(support_batch_labels)
        for label in unique_labels:
            # examples belonging to the current label
            examples_mask = (support_batch_labels == label)
            examples = support_batch_examples[examples_mask]
            # mean feature vector for the current class
            prototype = torch.mean(examples, dim=0)
            # store prototype for the current class
            prototypes[label.item()] = prototype

        return prototypes
    
    # calc distance of query from prototypes
    def calculate_distance(self, batch_query, prototypes):
        """
        batch_query: batch of coupled image and tabular data already fused
        prototypes: dict of calculated prototypicals 
        """
        labels = list(prototypes.keys())

        # [C, D]
        # get batch dist btwn batches
        prototype_vectors = torch.stack([prototypes[label] for label in labels])
        # [B, C, D]
        diff = batch_query.unsqueeze(1) - prototype_vectors.unsqueeze(0)
        # [B, C]
        dist = torch.norm(diff, dim=2)
        # distances = {labels[idx]: dist[:, idx].tolist() for idx in range(len(labels))}

        return dist

    # predict query from learned prototypes
    def predict_class(self, batch_query, prototypes):
        """
        batch_query: batch of coupled image and tabular data already fused
        prototypes: dict of calculated prototypicals 
        """
        # Calculate distances between the query example and prototypes
        distances = self.calculate_distance(batch_query, prototypes)
        k = list(distances.keys())
        v = list(distances.values())

        batch_to_proto = torch.tensor(v).T
        # return the topk smallest distances calculated across each class against the curr batch
        topk_values, topk_indices = torch.topk(batch_to_proto, topk, dim=1, largest=False, sorted=False)
        topk_labels = torch.tensor(k)[topk_indices]
        
        return topk_labels