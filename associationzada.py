from apyori import apriori
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def read_data(filename):
    try:
        path = f'content/{filename}.csv'
        # Read the CSV and replace empty strings with NaN
        data = pd.read_csv(path, header=None)
        # Replace empty strings and 'nan' strings with NaN
        data = data.replace(['', 'nan', 'NaN'], np.nan)
        return data
    except FileNotFoundError:
        print(f"Error: File {path} not found")
        return None

def dataframe_to_list(data):
    data_list = []
    for i in range(len(data)):
        # Only include non-NaN values for each transaction
        row_values = [str(data.values[i, j]) for j in range(len(data.columns))
                     if pd.notna(data.values[i, j])]
        if row_values:  # Only append if the row has valid items
            data_list.append(row_values)
    return data_list

def association(filename, min_support, min_confidence, min_lift):
    data = read_data(filename)
    if data is None:
        return []
    data_list = dataframe_to_list(data)
    rules = list(apriori(data_list, min_support=min_support, 
                        min_confidence=min_confidence, min_lift=min_lift))
    return rules

def extract_results(rules):
    results = []
    for rule in rules:
        # Filter out any NaN values that might have made it through
        items_base = [item for item in list(rule.ordered_statistics[0].items_base)
                     if pd.notna(item) and item != 'nan']
        items_add = [item for item in list(rule.ordered_statistics[0].items_add)
                    if pd.notna(item) and item != 'nan']
        
        # Only create a result if both items_base and items_add have valid items
        if items_base or items_add:
            support = rule.support
            confidence = rule.ordered_statistics[0].confidence
            lift = rule.ordered_statistics[0].lift
            results.append(Results(items_base, items_add, support, confidence, lift))
    return results

class Results:
    def __init__(self, items_base, items_add, support, confidence, lift):
        self.items_base = items_base
        self.items_add = items_add
        self.support = support
        self.confidence = confidence
        self.lift = lift
    
    def __str__(self):
        return f"Items Base: {self.items_base} \nItems Add: {self.items_add} \nSupport: {self.support} \nConfidence: {self.confidence} \nLift: {self.lift}"
    
    def to_dataframe(self):
        return pd.DataFrame({
            'Items Base': [self.items_base],
            'Items Add': [self.items_add],
            'Support': [self.support],
            'Confidence': [self.confidence],
            'Lift': [self.lift]
        })

def mercado(min_support, min_confidence, min_lift):
    results = association('mercado', min_support, min_confidence, min_lift)
    results_list = extract_results(results)
    results_assoc = []
    for result in results_list:
        results_assoc.append(result.to_dataframe())
    return results_assoc

def mercado2(min_support, min_confidence, min_lift):
    rules = association('mercado2', min_support, min_confidence, min_lift)
    results_list = extract_results(rules)
    results_assoc = []
    for result in results_list:
        results_assoc.append(result.to_dataframe())
    return results_assoc

def visualize_association_rules(results_list):
    """
    Create visualizations for association rules analysis results.
    """
    if not results_list:
        print("No rules found to visualize!")
        return None
        
    # Combine all dataframes
    combined_df = pd.concat(results_list, ignore_index=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Scatter plot of Support vs Confidence colored by Lift
    plt.subplot(2, 2, 1)
    scatter = plt.scatter(combined_df['Support'], 
                         combined_df['Confidence'],
                         c=combined_df['Lift'],
                         cmap='viridis',
                         alpha=0.6)
    plt.colorbar(scatter, label='Lift')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Support vs Confidence (colored by Lift)')
    
    # 2. Distribution of Lift values
    plt.subplot(2, 2, 2)
    sns.histplot(data=combined_df, x='Lift', bins=20)
    plt.title('Distribution of Lift Values')
    
    # 3. Top 10 rules by lift
    plt.subplot(2, 2, 3)
    top_10_lift = combined_df.nlargest(10, 'Lift')
    base_items = [' , '.join(map(str, x)) for x in top_10_lift['Items Base']]
    add_items = [' , '.join(map(str, x)) for x in top_10_lift['Items Add']]
    rules = [f"{b} → {a}" for b, a in zip(base_items, add_items)]
    
    plt.barh(range(len(rules)), top_10_lift['Lift'])
    plt.yticks(range(len(rules)), rules, fontsize=8)
    plt.xlabel('Lift')
    plt.title('Top 10 Rules by Lift')
    
    # 4. Support vs Lift scatter
    plt.subplot(2, 2, 4)
    plt.scatter(combined_df['Support'], 
               combined_df['Lift'],
               alpha=0.6)
    plt.xlabel('Support')
    plt.ylabel('Lift')
    plt.title('Support vs Lift')
    
    plt.tight_layout()
    return fig

def create_rules_summary(results_list):
    """
    Create a summary DataFrame of the association rules.
    """
    if not results_list:
        return pd.DataFrame({
            'Total Rules': [0],
            'Avg Support': [0],
            'Avg Confidence': [0],
            'Avg Lift': [0],
            'Max Lift': [0],
            'Min Lift': [0]
        })
        
    combined_df = pd.concat(results_list, ignore_index=True)
    
    summary = pd.DataFrame({
        'Total Rules': [len(combined_df)],
        'Avg Support': [combined_df['Support'].mean()],
        'Avg Confidence': [combined_df['Confidence'].mean()],
        'Avg Lift': [combined_df['Lift'].mean()],
        'Max Lift': [combined_df['Lift'].max()],
        'Min Lift': [combined_df['Lift'].min()]
    })
    
    return summary

results = mercado2(0.003, 0.2, 2)
summary = create_rules_summary(results)
print(summary)

fig = visualize_association_rules(results)
if fig:
    plt.show()

