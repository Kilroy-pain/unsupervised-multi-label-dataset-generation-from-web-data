import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import random

# Dummy dataset class for demonstration
class DummyImageDataset(Dataset):
    def __init__(self, num_samples=100, img_size=(3, 224, 224)):
        self.num_samples = num_samples
        self.img_size = img_size
        self.data = torch.rand(num_samples, *img_size)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]

# Feature extractor using a pre-trained ResNet model
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])  # Remove the classification layer
    
    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)

# Function to generate single-label dataset using clustering
def generate_single_label_dataset(features, num_clusters=10):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(features)
    
    # Select representative samples (anchors) for each cluster
    anchors = []
    for cluster_id in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_features = features[cluster_indices]
        cluster_center = kmeans.cluster_centers_[cluster_id]
        similarities = cosine_similarity(cluster_features, cluster_center.reshape(1, -1))
        anchor_idx = cluster_indices[np.argmax(similarities)]
        anchors.append(anchor_idx)
    
    return cluster_labels, anchors

# Function to augment labels using class activation maps
def augment_labels(model, dataset, cluster_labels, anchors, num_classes=10):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    augmented_labels = np.zeros((len(dataset), num_classes))
    with torch.no_grad():
        for idx in range(len(dataset)):
            img = dataset[idx].unsqueeze(0).to(device)
            output = model(img)
            probs = F.softmax(output, dim=1).cpu().numpy().squeeze()
            uncertainty = -np.sum(probs * np.log(probs + 1e-10))  # Entropy as uncertainty measure
            
            # Assign labels based on threshold
            threshold = 1.0 / num_classes + uncertainty
            augmented_labels[idx] = (probs > threshold).astype(int)
    
    return augmented_labels

if __name__ == '__main__':
    # Dummy data generation
    dataset = DummyImageDataset(num_samples=100)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    
    # Feature extraction
    feature_extractor = FeatureExtractor()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = feature_extractor.to(device)
    
    features = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            batch_features = feature_extractor(batch)
            features.append(batch_features.cpu().numpy())
    features = np.vstack(features)
    
    # Generate single-label dataset
    num_clusters = 5
    cluster_labels, anchors = generate_single_label_dataset(features, num_clusters=num_clusters)
    print(f"Cluster labels: {cluster_labels}")
    print(f"Anchor indices: {anchors}")
    
    # Augment labels
    classifier = models.resnet18(pretrained=True)
    classifier.fc = nn.Linear(classifier.fc.in_features, num_clusters)  # Adjust output layer for num_clusters
    classifier = classifier.to(device)
    
    augmented_labels = augment_labels(classifier, dataset, cluster_labels, anchors, num_classes=num_clusters)
    print(f"Augmented labels shape: {augmented_labels.shape}")
    print(f"Sample augmented labels: {augmented_labels[:5]}")