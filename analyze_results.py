"""
Script to analyze and compare results from multiple experiments.
"""
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def load_experiment_results(results_dir):
    """Load all experiment results from a directory."""
    results = []
    
    # Look for JSON summary files
    for filename in os.listdir(results_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    results.extend(data)
                else:
                    results.append(data)
    
    return results


def plot_classifier_comparison(results, save_path=None):
    """Plot comparison of different classifiers."""
    if not results:
        print("No results to plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy comparison
    ax1 = axes[0]
    classifiers = df['classifier'].unique()
    accuracies = [df[df['classifier'] == c]['test_accuracy'].values[0] * 100 
                  for c in classifiers]
    
    bars = ax1.bar(range(len(classifiers)), accuracies, color='steelblue', alpha=0.8)
    ax1.set_xlabel('Classifier', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Classifier Accuracy Comparison', fontsize=14)
    ax1.set_xticks(range(len(classifiers)))
    ax1.set_xticklabels(classifiers, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # ECE comparison
    ax2 = axes[1]
    eces = [df[df['classifier'] == c]['test_ece'].values[0] for c in classifiers]
    
    bars = ax2.bar(range(len(classifiers)), eces, color='coral', alpha=0.8)
    ax2.set_xlabel('Classifier', fontsize=12)
    ax2.set_ylabel('Expected Calibration Error', fontsize=12)
    ax2.set_title('Calibration Error Comparison', fontsize=14)
    ax2.set_xticks(range(len(classifiers)))
    ax2.set_xticklabels(classifiers, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, ece) in enumerate(zip(bars, eces)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{ece:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot to {save_path}")
    
    plt.close()


def plot_latent_dim_comparison(results, save_path=None):
    """Plot comparison of different latent dimensions."""
    if not results:
        print("No results to plot")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('latent_dim')
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy vs latent dimension
    ax1 = axes[0]
    ax1.plot(df['latent_dim'], df['test_accuracy'] * 100, 
            'o-', linewidth=2, markersize=8, color='steelblue')
    ax1.set_xlabel('Latent Dimension', fontsize=12)
    ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy vs Latent Dimension', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for _, row in df.iterrows():
        ax1.annotate(f"{row['test_accuracy']*100:.1f}%",
                    (row['latent_dim'], row['test_accuracy']*100),
                    textcoords="offset points", xytext=(0,10),
                    ha='center', fontsize=9)
    
    # ECE vs latent dimension
    ax2 = axes[1]
    ax2.plot(df['latent_dim'], df['test_ece'],
            'o-', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Latent Dimension', fontsize=12)
    ax2.set_ylabel('Expected Calibration Error', fontsize=12)
    ax2.set_title('Calibration Error vs Latent Dimension', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for _, row in df.iterrows():
        ax2.annotate(f"{row['test_ece']:.4f}",
                    (row['latent_dim'], row['test_ece']),
                    textcoords="offset points", xytext=(0,10),
                    ha='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved latent dim comparison plot to {save_path}")
    
    plt.close()


def create_results_table(results, save_path=None):
    """Create a formatted results table."""
    if not results:
        print("No results to tabulate")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Format the table
    df['test_accuracy'] = df['test_accuracy'] * 100
    df = df.round({'test_accuracy': 2, 'test_ece': 4, 
                   'final_train_acc': 2, 'final_test_acc': 2})
    
    # Reorder columns
    column_order = ['classifier', 'latent_dim', 'test_accuracy', 'test_ece']
    df = df[column_order]
    
    # Rename columns
    df.columns = ['Classifier', 'Latent Dim', 'Test Acc (%)', 'ECE']
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"Saved results table to {save_path}")
    
    return df


def generate_report(results_dir, output_dir=None):
    """Generate a comprehensive analysis report."""
    if output_dir is None:
        output_dir = results_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading experiment results...")
    results = load_experiment_results(results_dir)
    
    if not results:
        print(f"No results found in {results_dir}")
        return
    
    print(f"Found {len(results)} experiment results")
    
    # Check if this is classifier comparison or latent dim comparison
    df = pd.DataFrame(results)
    
    if 'classifier' in df.columns and len(df['classifier'].unique()) > 1:
        # Classifier comparison
        print("\nGenerating classifier comparison plots...")
        plot_classifier_comparison(
            results,
            save_path=os.path.join(output_dir, 'classifier_comparison.png')
        )
    
    if 'latent_dim' in df.columns and len(df['latent_dim'].unique()) > 1:
        # Latent dimension comparison
        print("\nGenerating latent dimension comparison plots...")
        plot_latent_dim_comparison(
            results,
            save_path=os.path.join(output_dir, 'latent_dim_comparison.png')
        )
    
    # Create results table
    print("\nGenerating results table...")
    table = create_results_table(
        results,
        save_path=os.path.join(output_dir, 'results_table.csv')
    )
    
    print("\n" + "=" * 80)
    print("RESULTS TABLE")
    print("=" * 80)
    print(table.to_string(index=False))
    print("=" * 80)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze experiment results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save analysis outputs')
    
    args = parser.parse_args()
    
    generate_report(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
