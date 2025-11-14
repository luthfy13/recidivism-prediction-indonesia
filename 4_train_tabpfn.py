"""
Training TabPFN - State-of-the-Art Foundation Model untuk Small Tabular Data
TabPFN adalah foundation model yang tidak memerlukan hyperparameter tuning
dan sangat optimal untuk dataset kecil (<10,000 samples)
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                            classification_report, confusion_matrix)
import warnings
warnings.filterwarnings('ignore')

# TabPFN might not be installed, so we'll handle import gracefully
try:
    from tabpfn import TabPFNClassifier
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False
    print("⚠️  TabPFN not installed. Install with: pip install tabpfn")

class TabPFNTrainer:
    """Trainer untuk TabPFN model"""

    def __init__(self):
        self.model = None
        self.results = {}

    def load_data(self):
        """Load preprocessed data"""
        print("\n" + "=" * 80)
        print("LOADING DATA FOR TABPFN")
        print("=" * 80)

        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')

        print(f"✓ Data loaded:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Classes: {len(np.unique(y_train))}")

        return X_train, X_test, y_train, y_test

    def train_tabpfn(self, X_train, y_train, X_test, y_test):
        """Train TabPFN model"""
        print("\n" + "=" * 80)
        print("TRAINING TABPFN")
        print("=" * 80)

        if not TABPFN_AVAILABLE:
            print("❌ TabPFN is not installed.")
            print("\nTo install TabPFN:")
            print("  pip install tabpfn")
            print("\nAlternatively, install from GitHub:")
            print("  pip install git+https://github.com/PriorLabs/TabPFN.git")
            return None

        # Check constraints
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))

        print(f"Checking TabPFN constraints:")
        print(f"  ✓ Samples: {n_samples} (limit: 10,000)")
        print(f"  ✓ Features: {n_features} (limit: 500)")
        print(f"  ✓ Classes: {n_classes}")

        if n_samples > 10000:
            print(f"⚠️  Warning: TabPFN optimal for ≤10,000 samples")

        if n_features > 500:
            print(f"❌ Error: TabPFN does not support >500 features")
            print(f"   Current features: {n_features}")
            return None

        # Initialize TabPFN
        # TabPFN doesn't need hyperparameter tuning!
        print("\nInitializing TabPFN (no hyperparameter tuning needed)...")
        self.model = TabPFNClassifier(
            device='cpu',  # Use 'cuda' if GPU available
            N_ensemble_configurations=32  # Default ensemble size
        )

        # Train
        print("Training TabPFN...")
        import time
        start_time = time.time()

        self.model.fit(X_train, y_train)

        training_time = time.time() - start_time
        print(f"✓ Training completed in {training_time:.2f} seconds")

        # Predict
        print("Making predictions...")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0

        self.results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'training_time': training_time,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        print(f"\n✓ TabPFN Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"  Training Time: {training_time:.2f}s")

        return self.model

    def detailed_evaluation(self, y_test):
        """Detailed evaluation report"""
        if self.model is None:
            print("❌ Model not trained yet")
            return

        print("\n" + "=" * 80)
        print("DETAILED EVALUATION - TABPFN")
        print("=" * 80)

        y_pred = self.results['predictions']

        # Load preprocessor for label names
        try:
            with open('preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            target_names = preprocessor.target_encoder.classes_
        except:
            target_names = ['rendah', 'sedang', 'tinggi']

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))

        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        print(cm_df)

        # Per-class metrics
        print("\nPer-Class Performance:")
        for idx, class_name in enumerate(target_names):
            mask = y_test == idx
            if mask.sum() > 0:
                class_acc = accuracy_score(y_test[mask], y_pred[mask])
                print(f"  {class_name}: {class_acc:.4f} ({mask.sum()} samples)")

    def compare_with_others(self):
        """Compare TabPFN with other models"""
        print("\n" + "=" * 80)
        print("COMPARISON WITH OTHER MODELS")
        print("=" * 80)

        try:
            with open('training_results.pkl', 'rb') as f:
                other_results = pickle.load(f)

            # Add TabPFN to comparison
            all_results = {
                'TabPFN': self.results,
                **other_results
            }

            # Create comparison DataFrame
            comparison_df = pd.DataFrame({
                'Model': list(all_results.keys()),
                'Accuracy': [all_results[m]['accuracy'] for m in all_results.keys()],
                'F1-Score': [all_results[m]['f1_score'] for m in all_results.keys()],
                'AUC': [all_results[m]['auc'] for m in all_results.keys()]
            })

            comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
            print(comparison_df.to_string(index=False))

            # Highlight best
            best_model = comparison_df.iloc[0]['Model']
            best_f1 = comparison_df.iloc[0]['F1-Score']
            print(f"\n🏆 Best Model: {best_model}")
            print(f"   F1-Score: {best_f1:.4f}")

            # Save updated comparison
            comparison_df.to_csv('model_comparison_with_tabpfn.csv', index=False)
            print(f"\n✓ Saved: model_comparison_with_tabpfn.csv")

            return comparison_df

        except FileNotFoundError:
            print("⚠️  Other models not trained yet. Run 3_train_models.py first.")
            return None

    def save_model(self):
        """Save TabPFN model"""
        if self.model is None:
            print("❌ No model to save")
            return

        print("\n" + "=" * 80)
        print("SAVING MODEL")
        print("=" * 80)

        # Save model
        with open('model_tabpfn.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("✓ Saved: model_tabpfn.pkl")

        # Save results
        with open('tabpfn_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        print("✓ Saved: tabpfn_results.pkl")

def main():
    """Main TabPFN training pipeline"""
    print("\n" + "=" * 80)
    print("TABPFN TRAINING - STATE-OF-THE-ART FOR SMALL TABULAR DATA")
    print("=" * 80)

    if not TABPFN_AVAILABLE:
        print("\n❌ TabPFN is not available.")
        print("\nInstallation instructions:")
        print("  1. Install via pip:")
        print("     pip install tabpfn")
        print("\n  2. Or install from GitHub:")
        print("     pip install git+https://github.com/PriorLabs/TabPFN.git")
        print("\n  3. If you encounter issues, you may need PyTorch:")
        print("     pip install torch")
        print("\nAfter installation, run this script again.")
        return

    # Initialize trainer
    trainer = TabPFNTrainer()

    # Load data
    X_train, X_test, y_train, y_test = trainer.load_data()

    # Train TabPFN
    model = trainer.train_tabpfn(X_train, y_train, X_test, y_test)

    if model is not None:
        # Detailed evaluation
        trainer.detailed_evaluation(y_test)

        # Compare with other models
        trainer.compare_with_others()

        # Save model
        trainer.save_model()

        print("\n" + "=" * 80)
        print("TABPFN TRAINING COMPLETED")
        print("=" * 80)
        print("\n💡 Key Advantages of TabPFN:")
        print("  ✓ No hyperparameter tuning needed")
        print("  ✓ Optimal for small datasets (<10K samples)")
        print("  ✓ Fast training (seconds vs hours)")
        print("  ✓ State-of-the-art performance")
        print("  ✓ Published in Nature 2025")

if __name__ == "__main__":
    main()
