"""
K-Means Clustering Module (Single Attribute)

This module implements K-Means clustering for single attribute analysis.
It provides functionality for clustering data and visualizing results
using a single feature.

Key Features:
1. Data Loading & Preprocessing
   - Handle numerical data from CSV files
   - Scale/normalize data as needed

2. K-Means Clustering
   - Implement clustering with specified number of clusters
   - Handle single attribute clustering cases

3. Visualization & Analysis
   - Generate visualizations of clusters
   - Create summary statistics of clustering results

Author: Furukawa
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class SingleAttributeClusterer:
    """Handles K-Means clustering for single attribute analysis."""
    
    def __init__(self, data_path: str = 'machine_learning/data/raw'):
        """Initialize the clusterer.
        
        Args:
            data_path: Base path for data files
        """
        self.data_path = data_path
        self.data = None
        self.scaled_data = None
        self.kmeans = None
        self.n_clusters = None
        self.feature_name = None
    
    def load_data(self, filename: str, feature_name: str) -> bool:
        """Load data from CSV file and extract single feature.
        
        Args:
            filename: Name of the CSV file (without extension)
            feature_name: Name of the feature to cluster on
            
        Returns:
            bool: True if data loaded successfully
        """
        try:
            path = f'{self.data_path}/{filename}.csv'
            self.data = pd.read_csv(path)
            self.feature_name = feature_name
            
            if feature_name not in self.data.columns:
                print(f"Error: Feature '{feature_name}' not found in dataset")
                return False
                
            return True
        except FileNotFoundError:
            print(f"Error: File {path} not found")
            return False
    
    def preprocess_data(self) -> np.ndarray:
        """Scale the feature data for clustering.
        
        Returns:
            Scaled feature values as numpy array
        """
        feature_data = self.data[self.feature_name].values.reshape(-1, 1)
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(feature_data)
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
    
    def visualize_clusters(self) -> plt.Figure:
        """Create visualization of clustering results.
        
        Returns:
            matplotlib Figure object
        """
        if self.kmeans is None:
            self.perform_clustering()
        
        fig = plt.figure(figsize=(15, 5))
        
        # Original data distribution
        plt.subplot(1, 3, 1)
        plt.hist(self.data[self.feature_name], bins=30)
        plt.title(f'Distribution of {self.feature_name}')
        plt.xlabel(self.feature_name)
        plt.ylabel('Frequency')
        
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
        
        # Cluster distribution
        plt.subplot(1, 3, 3)
        for i in range(self.n_clusters):
            cluster_data = self.data[self.feature_name][self.kmeans.labels_ == i]
            plt.hist(cluster_data, bins=20, alpha=0.5, 
                    label=f'Cluster {i+1}')
        plt.title('Cluster Distributions')
        plt.xlabel(self.feature_name)
        plt.ylabel('Frequency')
        plt.legend()
        
        plt.tight_layout()
        return fig
    
    def get_cluster_stats(self) -> pd.DataFrame:
        """Generate summary statistics for each cluster.
        
        Returns:
            DataFrame with cluster statistics
        """
        if self.kmeans is None:
            self.perform_clustering()
        
        stats = []
        for i in range(self.n_clusters):
            cluster_data = self.data[self.feature_name][self.kmeans.labels_ == i]
            stats.append({
                'Cluster': i + 1,
                'Size': len(cluster_data),
                'Mean': cluster_data.mean(),
                'Std': cluster_data.std(),
                'Min': cluster_data.min(),
                'Max': cluster_data.max()
            })
        
        return pd.DataFrame(stats)

def main():
    """Example usage of the SingleAttributeClusterer class."""
    # Initialize clusterer
    clusterer = SingleAttributeClusterer()
    
    # Load and analyze age data
    if clusterer.load_data('census', 'age'):
        # Perform clustering
        clusterer.perform_clustering(n_clusters=3)
        
        # Print cluster statistics
        stats = clusterer.get_cluster_stats()
        print("\nCluster Statistics:")
        print(stats)
        
        # Show visualizations
        fig = clusterer.visualize_clusters()
        plt.show()

if __name__ == "__main__":
    main()
