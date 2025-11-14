"""
Comprehensive Model Evaluation and Comparison
Visualisasi performa semua model SOTA
"""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score, f1_score, precision_score,
                            recall_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

class ModelEvaluator:
    """Comprehensive model evaluation and comparison"""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.y_test = None
        self.preprocessor = None

    def load_all_models(self):
        """Load all trained models"""
        print("\n" + "=" * 80)
        print("LOADING TRAINED MODELS")
        print("=" * 80)

        model_files = {
            'CatBoost': 'model_catboost.pkl',
            'XGBoost+SMOTE': 'model_xgboost_smote.pkl',
            'RandomForest+ADASYN': 'model_randomforest_adasyn.pkl',
            'LightGBM+SMOTE': 'model_lightgbm_smote.pkl',
            'Ensemble': 'model_ensemble.pkl',
            'TabPFN': 'model_tabpfn.pkl'
        }

        for name, filepath in model_files.items():
            try:
                with open(filepath, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"✓ Loaded: {name}")
            except FileNotFoundError:
                print(f"⚠️  Not found: {name} ({filepath})")

        # Load preprocessor
        try:
            with open('preprocessor.pkl', 'rb') as f:
                self.preprocessor = pickle.load(f)
            print(f"✓ Loaded: preprocessor")
        except:
            print(f"⚠️  Preprocessor not found")

        print(f"\nTotal models loaded: {len(self.models)}")

    def load_results(self):
        """Load training results"""
        print("\n" + "=" * 80)
        print("LOADING RESULTS")
        print("=" * 80)

        try:
            with open('training_results.pkl', 'rb') as f:
                self.results = pickle.load(f)
            print(f"✓ Loaded training results")
        except:
            print(f"⚠️  Training results not found")

        # Add TabPFN if available
        try:
            with open('tabpfn_results.pkl', 'rb') as f:
                tabpfn_results = pickle.load(f)
            self.results['TabPFN'] = tabpfn_results
            print(f"✓ Added TabPFN results")
        except:
            print(f"⚠️  TabPFN results not found")

        # Load test data
        try:
            self.y_test = np.load('y_test.npy')
            print(f"✓ Loaded test labels: {len(self.y_test)} samples")
        except:
            print(f"⚠️  Test data not found")

    def plot_model_comparison(self):
        """Plot comparison of all models"""
        print("\n" + "=" * 80)
        print("PLOTTING MODEL COMPARISON")
        print("=" * 80)

        if not self.results:
            print("❌ No results to plot")
            return

        # Prepare data
        models = list(self.results.keys())
        metrics = ['accuracy', 'f1_score', 'auc']
        metric_names = ['Accuracy', 'F1-Score', 'AUC']

        data = {
            'Model': models * 3,
            'Metric': [],
            'Score': []
        }

        for metric, metric_name in zip(metrics, metric_names):
            for model in models:
                data['Metric'].append(metric_name)
                data['Score'].append(self.results[model][metric])

        df = pd.DataFrame(data)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            scores = [self.results[m][metric] for m in models]
            colors = ['#e74c3c' if s == max(scores) else '#3498db' for s in scores]

            axes[idx].barh(models, scores, color=colors)
            axes[idx].set_xlabel(metric_name, fontsize=12, fontweight='bold')
            axes[idx].set_title(f'{metric_name} Comparison', fontsize=14, fontweight='bold')
            axes[idx].set_xlim([0, 1])

            # Add value labels
            for i, v in enumerate(scores):
                axes[idx].text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig('output_model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: output_model_comparison.png")
        plt.close()

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        print("\n" + "=" * 80)
        print("PLOTTING CONFUSION MATRICES")
        print("=" * 80)

        if not self.results or self.y_test is None:
            print("❌ No results or test data available")
            return

        target_names = ['rendah', 'sedang', 'tinggi']
        if self.preprocessor:
            target_names = self.preprocessor.target_encoder.classes_

        n_models = len(self.results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_models > 1 else [axes]

        for idx, (model_name, result) in enumerate(self.results.items()):
            y_pred = result['predictions']
            cm = confusion_matrix(self.y_test, y_pred)

            # Normalize
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

            # Plot
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=target_names, yticklabels=target_names,
                       ax=axes[idx], cbar_kws={'label': 'Proportion'})
            axes[idx].set_title(f'{model_name}\nF1: {result["f1_score"]:.4f}',
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=10)
            axes[idx].set_ylabel('Actual', fontsize=10)

        # Hide empty subplots
        for idx in range(len(self.results), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('output_confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: output_confusion_matrices.png")
        plt.close()

    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        print("\n" + "=" * 80)
        print("PLOTTING ROC CURVES")
        print("=" * 80)

        if not self.results or self.y_test is None:
            print("❌ No results or test data available")
            return

        n_classes = len(np.unique(self.y_test))
        target_names = ['rendah', 'sedang', 'tinggi']
        if self.preprocessor:
            target_names = self.preprocessor.target_encoder.classes_

        # Binarize labels for ROC
        y_test_bin = label_binarize(self.y_test, classes=range(n_classes))

        # Plot for each class
        fig, axes = plt.subplots(1, n_classes, figsize=(18, 5))
        if n_classes == 1:
            axes = [axes]

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

        for class_idx in range(n_classes):
            for model_idx, (model_name, result) in enumerate(self.results.items()):
                try:
                    y_proba = result['probabilities']
                    if y_proba.shape[1] != n_classes:
                        continue

                    fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx],
                                           y_proba[:, class_idx])
                    roc_auc = auc(fpr, tpr)

                    axes[class_idx].plot(fpr, tpr,
                                        color=colors[model_idx % len(colors)],
                                        label=f'{model_name} (AUC={roc_auc:.3f})',
                                        linewidth=2)
                except:
                    continue

            axes[class_idx].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
            axes[class_idx].set_xlabel('False Positive Rate', fontsize=11)
            axes[class_idx].set_ylabel('True Positive Rate', fontsize=11)
            axes[class_idx].set_title(f'ROC Curve - Class: {target_names[class_idx]}',
                                     fontsize=12, fontweight='bold')
            axes[class_idx].legend(loc='lower right', fontsize=8)
            axes[class_idx].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('output_roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: output_roc_curves.png")
        plt.close()

    def plot_per_class_performance(self):
        """Plot per-class performance metrics"""
        print("\n" + "=" * 80)
        print("PLOTTING PER-CLASS PERFORMANCE")
        print("=" * 80)

        if not self.results or self.y_test is None:
            print("❌ No results or test data available")
            return

        target_names = ['rendah', 'sedang', 'tinggi']
        if self.preprocessor:
            target_names = self.preprocessor.target_encoder.classes_

        # Calculate per-class metrics for each model
        data = []
        for model_name, result in self.results.items():
            y_pred = result['predictions']

            for class_idx, class_name in enumerate(target_names):
                mask = self.y_test == class_idx
                if mask.sum() > 0:
                    class_acc = accuracy_score(self.y_test[mask], y_pred[mask])
                    data.append({
                        'Model': model_name,
                        'Class': class_name,
                        'Accuracy': class_acc,
                        'Samples': mask.sum()
                    })

        df = pd.DataFrame(data)

        # Plot
        fig, ax = plt.subplots(figsize=(14, 6))

        df_pivot = df.pivot(index='Model', columns='Class', values='Accuracy')
        df_pivot.plot(kind='bar', ax=ax, width=0.8, color=['#2ecc71', '#f39c12', '#e74c3c'])

        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Accuracy by Model', fontsize=14, fontweight='bold')
        ax.legend(title='Risk Level', fontsize=10)
        ax.set_ylim([0, 1])
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('output_per_class_performance.png', dpi=300, bbox_inches='tight')
        print("✓ Saved: output_per_class_performance.png")
        plt.close()

    def generate_detailed_report(self):
        """Generate detailed evaluation report"""
        print("\n" + "=" * 80)
        print("GENERATING DETAILED REPORT")
        print("=" * 80)

        if not self.results or self.y_test is None:
            print("❌ No results or test data available")
            return

        target_names = ['rendah', 'sedang', 'tinggi']
        if self.preprocessor:
            target_names = self.preprocessor.target_encoder.classes_

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE MODEL EVALUATION REPORT")
        report_lines.append("Prediksi Residivis - Anak Berhadapan dengan Hukum (ABH)")
        report_lines.append("=" * 80)
        report_lines.append("")

        for model_name, result in self.results.items():
            report_lines.append("-" * 80)
            report_lines.append(f"MODEL: {model_name}")
            report_lines.append("-" * 80)
            report_lines.append("")

            y_pred = result['predictions']

            # Overall metrics
            report_lines.append("Overall Metrics:")
            report_lines.append(f"  Accuracy:  {result['accuracy']:.4f}")
            report_lines.append(f"  F1-Score:  {result['f1_score']:.4f}")
            report_lines.append(f"  AUC:       {result['auc']:.4f}")
            report_lines.append("")

            # Classification report
            report_lines.append("Classification Report:")
            report_lines.append(classification_report(self.y_test, y_pred,
                                                     target_names=target_names))

            # Confusion matrix
            report_lines.append("Confusion Matrix:")
            cm = confusion_matrix(self.y_test, y_pred)
            cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
            report_lines.append(cm_df.to_string())
            report_lines.append("")

            # Per-class accuracy
            report_lines.append("Per-Class Accuracy:")
            for class_idx, class_name in enumerate(target_names):
                mask = self.y_test == class_idx
                if mask.sum() > 0:
                    class_acc = accuracy_score(self.y_test[mask], y_pred[mask])
                    report_lines.append(f"  {class_name}: {class_acc:.4f} ({mask.sum()} samples)")
            report_lines.append("")

        # Save report
        with open('evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print("✓ Saved: evaluation_report.txt")

        # Also print to console
        print("\n" + '\n'.join(report_lines[:50]))  # Print first 50 lines

    def create_summary_table(self):
        """Create summary comparison table"""
        print("\n" + "=" * 80)
        print("CREATING SUMMARY TABLE")
        print("=" * 80)

        if not self.results:
            print("❌ No results available")
            return

        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'F1-Score': result['f1_score'],
                'AUC': result['auc'],
                'Precision (weighted)': precision_score(self.y_test,
                                                       result['predictions'],
                                                       average='weighted'),
                'Recall (weighted)': recall_score(self.y_test,
                                                 result['predictions'],
                                                 average='weighted')
            })

        df = pd.DataFrame(summary_data)
        df = df.sort_values('F1-Score', ascending=False)

        print("\n" + "=" * 80)
        print("SUMMARY TABLE")
        print("=" * 80)
        print(df.to_string(index=False, float_format='%.4f'))

        # Save
        df.to_csv('summary_comparison.csv', index=False, float_format='%.4f')
        print("\n✓ Saved: summary_comparison.csv")

        # Best model
        best_model = df.iloc[0]['Model']
        best_f1 = df.iloc[0]['F1-Score']
        print(f"\n🏆 BEST MODEL: {best_model}")
        print(f"   F1-Score: {best_f1:.4f}")

        return df

def main():
    """Main evaluation pipeline"""
    print("\n" + "=" * 80)
    print("MODEL EVALUATION & COMPARISON - PREDIKSI RESIDIVIS")
    print("=" * 80)

    evaluator = ModelEvaluator()

    # Load models and results
    evaluator.load_all_models()
    evaluator.load_results()

    # Generate visualizations
    evaluator.plot_model_comparison()
    evaluator.plot_confusion_matrices()
    evaluator.plot_roc_curves()
    evaluator.plot_per_class_performance()

    # Generate reports
    evaluator.generate_detailed_report()
    summary_df = evaluator.create_summary_table()

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - output_model_comparison.png")
    print("  - output_confusion_matrices.png")
    print("  - output_roc_curves.png")
    print("  - output_per_class_performance.png")
    print("  - evaluation_report.txt")
    print("  - summary_comparison.csv")

if __name__ == "__main__":
    main()
