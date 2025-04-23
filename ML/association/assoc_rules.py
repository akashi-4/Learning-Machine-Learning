"""
Association Rules Analysis Module

This module implements association rule mining using the Apriori algorithm.
It provides functionality for analyzing transaction data and discovering
frequent itemsets and association rules.

Key Features:
1. Data Loading & Preprocessing
   - Handle transaction data from CSV files
   - Clean and format data for analysis

2. Association Rule Mining
   - Apply Apriori algorithm
   - Extract rules with support, confidence, and lift metrics

3. Visualization & Analysis
   - Generate visualizations of rule metrics
   - Create summary statistics of discovered rules

Author: Furukawa
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from apyori import apriori
from typing import List, Optional

class AssociationRule:
    """Represents a single association rule with its metrics."""
    
    def __init__(self, items_base: list, items_add: list, 
                 support: float, confidence: float, lift: float):
        """Initialize an association rule.
        
        Args:
            items_base: Antecedent items (left side of rule)
            items_add: Consequent items (right side of rule)
            support: Support value of the rule
            confidence: Confidence value of the rule
            lift: Lift value of the rule
        """
        self.items_base = items_base
        self.items_add = items_add
        self.support = support
        self.confidence = confidence
        self.lift = lift
    
    def __str__(self) -> str:
        """Return string representation of the rule."""
        return (f"Rule: {self.items_base} → {self.items_add}\n"
                f"Support: {self.support:.4f}\n"
                f"Confidence: {self.confidence:.4f}\n"
                f"Lift: {self.lift:.4f}")
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert rule to a pandas DataFrame row."""
        return pd.DataFrame({
            'Items Base': [self.items_base],
            'Items Add': [self.items_add],
            'Support': [self.support],
            'Confidence': [self.confidence],
            'Lift': [self.lift]
        })

class AssociationRuleMiner:
    """Handles association rule mining from transaction data."""
    
    def __init__(self, data_path: str = 'machine_learning/data/raw'):
        """Initialize the association rule miner.
        
        Args:
            data_path: Base path for data files (default: 'machine_learning/data/raw')
        """
        self.data_path = data_path
        self.data = None
        self.rules = []
    
    def load_data(self, filename: str) -> bool:
        """Load transaction data from CSV file.
        
        Args:
            filename: Name of the CSV file (without extension)
            
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            path = f'{self.data_path}/{filename}.csv'
            self.data = pd.read_csv(path, header=None)
            # Replace empty strings and 'nan' values with NaN
            self.data = self.data.replace(['', 'nan', 'NaN'], np.nan)
            return True
        except FileNotFoundError:
            print(f"Error: File {path} not found")
            return False
    
    def _prepare_transactions(self) -> List[List[str]]:
        """Convert DataFrame to list of transactions.
        
        Returns:
            List of transactions, where each transaction is a list of items
        """
        transactions = []
        for i in range(len(self.data)):
            # Include only non-NaN values for each transaction
            items = [str(self.data.values[i, j]) 
                    for j in range(len(self.data.columns))
                    if pd.notna(self.data.values[i, j])]
            if items:  # Only append if transaction has valid items
                transactions.append(items)
        return transactions
    
    def mine_rules(self, min_support: float, min_confidence: float, 
                  min_lift: float) -> List[AssociationRule]:
        """Mine association rules using Apriori algorithm.
        
        Args:
            min_support: Minimum support threshold
            min_confidence: Minimum confidence threshold
            min_lift: Minimum lift threshold
            
        Returns:
            List of AssociationRule objects
        """
        if self.data is None:
            print("Error: No data loaded")
            return []
        
        transactions = self._prepare_transactions()
        apriori_rules = list(apriori(transactions, 
                                   min_support=min_support,
                                   min_confidence=min_confidence,
                                   min_lift=min_lift))
        
        self.rules = self._extract_rules(apriori_rules)
        return self.rules
    
    def _extract_rules(self, apriori_rules: list) -> List[AssociationRule]:
        """Extract rules from apriori results.
        
        Args:
            apriori_rules: Raw rules from apriori algorithm
            
        Returns:
            List of AssociationRule objects
        """
        extracted_rules = []
        for rule in apriori_rules:
            # Filter out NaN values
            items_base = [item for item in list(rule.ordered_statistics[0].items_base)
                         if pd.notna(item) and item != 'nan']
            items_add = [item for item in list(rule.ordered_statistics[0].items_add)
                        if pd.notna(item) and item != 'nan']
            
            if items_base or items_add:
                extracted_rules.append(AssociationRule(
                    items_base=items_base,
                    items_add=items_add,
                    support=rule.support,
                    confidence=rule.ordered_statistics[0].confidence,
                    lift=rule.ordered_statistics[0].lift
                ))
        return extracted_rules
    
    def create_summary(self) -> pd.DataFrame:
        """Create summary statistics of discovered rules.
        
        Returns:
            DataFrame with summary statistics
        """
        if not self.rules:
            return pd.DataFrame({
                'Total Rules': [0],
                'Avg Support': [0],
                'Avg Confidence': [0],
                'Avg Lift': [0],
                'Max Lift': [0],
                'Min Lift': [0]
            })
        
        rules_df = pd.concat([rule.to_dataframe() for rule in self.rules], 
                           ignore_index=True)
        
        return pd.DataFrame({
            'Total Rules': [len(rules_df)],
            'Avg Support': [rules_df['Support'].mean()],
            'Avg Confidence': [rules_df['Confidence'].mean()],
            'Avg Lift': [rules_df['Lift'].mean()],
            'Max Lift': [rules_df['Lift'].max()],
            'Min Lift': [rules_df['Lift'].min()]
        })
    
    def visualize_rules(self) -> Optional[plt.Figure]:
        """Create visualization of rule metrics.
        
        Returns:
            matplotlib Figure object or None if no rules exist
        """
        if not self.rules:
            print("No rules found to visualize!")
            return None
        
        rules_df = pd.concat([rule.to_dataframe() for rule in self.rules], 
                           ignore_index=True)
        
        fig = plt.figure(figsize=(15, 10))
        
        # Support vs Confidence scatter plot
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(rules_df['Support'], 
                            rules_df['Confidence'],
                            c=rules_df['Lift'],
                            cmap='viridis',
                            alpha=0.6)
        plt.colorbar(scatter, label='Lift')
        plt.xlabel('Support')
        plt.ylabel('Confidence')
        plt.title('Support vs Confidence (colored by Lift)')
        
        # Lift distribution
        plt.subplot(2, 2, 2)
        sns.histplot(data=rules_df, x='Lift', bins=20)
        plt.title('Distribution of Lift Values')
        
        # Top 10 rules by lift
        plt.subplot(2, 2, 3)
        top_10_lift = rules_df.nlargest(10, 'Lift')
        base_items = [' , '.join(map(str, x)) for x in top_10_lift['Items Base']]
        add_items = [' , '.join(map(str, x)) for x in top_10_lift['Items Add']]
        rules_text = [f"{b} → {a}" for b, a in zip(base_items, add_items)]
        
        plt.barh(range(len(rules_text)), top_10_lift['Lift'])
        plt.yticks(range(len(rules_text)), rules_text, fontsize=8)
        plt.xlabel('Lift')
        plt.title('Top 10 Rules by Lift')
        
        # Support vs Lift scatter plot
        plt.subplot(2, 2, 4)
        plt.scatter(rules_df['Support'], 
                   rules_df['Lift'],
                   alpha=0.6)
        plt.xlabel('Support')
        plt.ylabel('Lift')
        plt.title('Support vs Lift')
        
        plt.tight_layout()
        return fig

def main():
    """Example usage of the AssociationRuleMiner class."""
    # Initialize miner with default path
    miner = AssociationRuleMiner()
    
    # Load and mine rules
    if miner.load_data('mercado'):
        rules = miner.mine_rules(
            min_support=0.003,
            min_confidence=0.2,
            min_lift=2
        )

        # Get summary statistics
        summary = miner.create_summary()
        print(summary)

        # Visualize the rules
        miner.visualize_rules()
        plt.show()

if __name__ == "__main__":
    main()
