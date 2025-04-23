"""
K-Means Clustering Module (Multiple Attributes)

This module implements K-Means clustering for multi-attribute analysis.
It provides functionality for clustering data and visualizing results
using multiple features.

Key Features:
1. Data Loading & Preprocessing
   - Handle numerical data from CSV files
   - Scale/normalize multiple features
   - Handle categorical variables

2. K-Means Clustering
   - Implement clustering with specified number of clusters
   - Support for multiple attributes
   - Automatic feature selection

3. Visualization & Analysis
   - Generate visualizations of clusters in multiple dimensions
   - Create summary statistics of clustering results
   - Feature importance analysis

Author: Furukawa
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional, Dict

class MultiAttributeClusterer:
    """Handles K-Means clustering for multiple attributes."""
    
    def __init__(self, data_path: str = 'machine_learning/data/raw'):
        """Initialize the clusterer.
        
        Args:
            data_path: Base path for data files
        """
        self.data_path = data_path
        self.data = None
        self.scaled_data = None
        self.feature_names = None
        self.kmeans = None
        self.n_clusters = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.pca = None
        self.pca_data = None
    
    def load_data(self, filename: str, features: List[str], 
                 categorical_features: Optional[List[str]] = None) -> bool:
        """Load data from CSV file and extract specified features.
        
        Args:
            filename: Name of the CSV file (without extension)
            features: List of feature names to use for clustering
            categorical_features: List of categorical feature names
            
        Returns:
            bool: True if data loaded successfully
        """
        try:
            path = f'{self.data_path}/{filename}.csv'
            self.data = pd.read_csv(path)
            self.feature_names = features
            
            # Verify all features exist
            missing_features = [f for f in features if f not in self.data.columns]
            if missing_features:
                print(f"Error: Features {missing_features} not found in dataset")
                return False
            
            # Handle categorical features
            if categorical_features:
                for feature in categorical_features:
                    if feature in features:
                        self._encode_categorical(feature)
            
            return True
        except FileNotFoundError:
            print(f"Error: File {path} not found")
            return False
    
    def _encode_categorical(self, feature: str):
        """Encode categorical feature using LabelEncoder.
        
        Args:
            feature: Name of categorical feature to encode
        """
        encoder = LabelEncoder()
        self.data[feature] = encoder.fit_transform(self.data[feature])
        self.label_encoders[feature] = encoder
    
    def preprocess_data(self) -> np.ndarray:
        """Scale the feature data for clustering.
        
        Returns:
            Scaled feature values as numpy array
        """
        feature_data = self.data[self.feature_names].values
        self.scaled_data = self.scaler.fit_transform(feature_data)
        return self.scaled_data
    
    def find_optimal_k(self, max_k: int = 10) -> Tuple[int, list]:
        """Find optimal number of clusters using elbow method.
        
        Args:
            max_k: Maximum number of clusters to try
            
        Returns:
            Tuple of optimal k and inertia values list
        """
        if self.scaled_data is None:
            self.preprocess_data()
        
        inertias = []
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(self.scaled_data)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point (point of maximum curvature)
        diffs = np.diff(inertias)
        optimal_k = np.argmin(diffs) + 1
        
        return optimal_k, inertias
    
    def perform_clustering(self, n_clusters: Optional[int] = None) -> np.ndarray:
        """Perform K-Means clustering.
        
        Args:
            n_clusters: Number of clusters (if None, finds optimal k)
            
        Returns:
            Cluster labels for each data point
        """
        if self.scaled_data is None:
            self.preprocess_data()
        
        if n_clusters is None:
            n_clusters, _ = self.find_optimal_k()
        
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        return self.kmeans.fit_predict(self.scaled_data)
    
    def _perform_pca(self, n_components: int = 2):
        """Perform PCA for visualization.
        
        Args:
            n_components: Number of components for PCA
        """
        if self.scaled_data is None:
            self.preprocess_data()
        
        self.pca = PCA(n_components=n_components)
        self.pca_data = self.pca.fit_transform(self.scaled_data)
    
    def visualize_clusters(self) -> plt.Figure:
        """Create visualization of clustering results.
        
        Returns:
            matplotlib Figure object
        """
        if self.kmeans is None:
            self.perform_clustering()
        
        # Perform PCA for visualization if more than 2 features
        if len(self.feature_names) > 2:
            if self.pca_data is None:
                self._perform_pca()
            plot_data = self.pca_data
            x_label = 'First Principal Component'
            y_label = 'Second Principal Component'
        else:
            plot_data = self.scaled_data
            x_label = self.feature_names[0]
            y_label = self.feature_names[1]
        
        fig = plt.figure(figsize=(15, 5))
        
        # Scatter plot of clusters
        plt.subplot(1, 3, 1)
        scatter = plt.scatter(plot_data[:, 0], plot_data[:, 1], 
                            c=self.kmeans.labels_, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('Cluster Assignments')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        # Elbow curve
        plt.subplot(1, 3, 2)
        _, inertias = self.find_optimal_k()
        plt.plot(range(1, len(inertias) + 1), inertias, 'bo-')
        plt.axvline(x=self.n_clusters, color='r', linestyle='--', 
                   label=f'Selected k={self.n_clusters}')
        plt.title('Elbow Method')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.legend()
        
        # Feature importance (if PCA was used)
        plt.subplot(1, 3, 3)
        if self.pca is not None:
            explained_var = self.pca.explained_variance_ratio_
            plt.bar(range(len(explained_var)), explained_var)
            plt.title('PCA Explained Variance Ratio')
            plt.xlabel('Principal Component')
            plt.ylabel('Explained Variance Ratio')
        else:
            plt.text(0.5, 0.5, 'PCA not performed\n(2 or fewer features)', 
                    ha='center', va='center')
            plt.title('Feature Importance')
        
        plt.tight_layout()
        return fig
    
    def get_cluster_stats(self) -> Dict[str, pd.DataFrame]:
        """Generate summary statistics for each cluster.
        
        Returns:
            Dictionary of DataFrames with cluster statistics for each feature
        """
        if self.kmeans is None:
            self.perform_clustering()
        
        stats = {}
        for feature in self.feature_names:
            feature_stats = []
            for i in range(self.n_clusters):
                cluster_data = self.data[feature][self.kmeans.labels_ == i]
                feature_stats.append({
                    'Cluster': i + 1,
                    'Size': len(cluster_data),
                    'Mean': cluster_data.mean(),
                    'Std': cluster_data.std(),
                    'Min': cluster_data.min(),
                    'Max': cluster_data.max()
                })
            stats[feature] = pd.DataFrame(feature_stats)
        
        return stats

def main():
    """Example usage of the MultiAttributeClusterer class."""
    # Initialize clusterer
    clusterer = MultiAttributeClusterer()
    
    # Load and analyze census data with multiple features
    features = ['age', 'income', 'education']
    categorical_features = ['education']
    
    if clusterer.load_data('census', features, categorical_features):
        # Perform clustering
        clusterer.perform_clustering(n_clusters=4)
        
        # Print cluster statistics
        stats = clusterer.get_cluster_stats()
        for feature, feature_stats in stats.items():
            print(f"\nCluster Statistics for {feature}:")
            print(feature_stats)
        
        # Show visualizations
        fig = clusterer.visualize_clusters()
        plt.show()

if __name__ == "__main__":
    main()
