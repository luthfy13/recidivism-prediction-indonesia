"""
Training SOTA Models untuk Prediksi Residivis
Models: CatBoost, XGBoost, Random Forest, LightGBM
Dengan handling imbalanced data menggunakan SMOTE
"""

import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                            accuracy_score, f1_score, roc_auc_score,
                            precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
from catboost import CatBoostClassifier
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class SOTAModels:
    """State-of-the-Art Models untuk Prediksi Residivis"""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def load_data(self):
        """Load preprocessed data"""
        print("\n" + "=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        # Load encoded data for XGBoost, RF, LightGBM
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')

        print(f"✓ Encoded data loaded:")
        print(f"  X_train: {X_train.shape}")
        print(f"  X_test: {X_test.shape}")

        # Load CatBoost data
        X_train_cat = pd.read_pickle('X_train_catboost.pkl')
        X_test_cat = pd.read_pickle('X_test_catboost.pkl')
        y_train_cat = np.load('y_train_catboost.npy')
        y_test_cat = np.load('y_test_catboost.npy')
        cat_indices = np.load('cat_indices.npy')

        print(f"\n✓ CatBoost data loaded:")
        print(f"  X_train_cat: {X_train_cat.shape}")
        print(f"  X_test_cat: {X_test_cat.shape}")
        print(f"  Categorical indices: {len(cat_indices)}")

        return (X_train, X_test, y_train, y_test,
                X_train_cat, X_test_cat, y_train_cat, y_test_cat, cat_indices)

    def train_catboost(self, X_train, y_train, X_test, y_test, cat_indices):
        """Train CatBoost with categorical features"""
        print("\n" + "=" * 80)
        print("TRAINING CATBOOST")
        print("=" * 80)

        # CatBoost handles imbalance internally
        model = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            border_count=128,
            cat_features=cat_indices,
            auto_class_weights='Balanced',  # Handle imbalance
            random_seed=self.random_state,
            verbose=False,
            early_stopping_rounds=50
        )

        print("Training CatBoost...")
        model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            verbose=False
        )

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Multiclass ROC AUC (OvR)
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0

        self.models['CatBoost'] = model
        self.results['CatBoost'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        print(f"✓ CatBoost trained successfully")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")

        return model

    def train_xgboost_with_smote(self, X_train, y_train, X_test, y_test):
        """Train XGBoost with SMOTE"""
        print("\n" + "=" * 80)
        print("TRAINING XGBOOST + SMOTE")
        print("=" * 80)

        # Apply SMOTE
        print("Applying SMOTE...")
        smote = SMOTE(random_state=self.random_state, k_neighbors=2)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        print(f"✓ After SMOTE:")
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} samples")

        # XGBoost
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            eval_metric='mlogloss',
            early_stopping_rounds=50
        )

        print("Training XGBoost...")
        model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0

        self.models['XGBoost_SMOTE'] = model
        self.results['XGBoost_SMOTE'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        print(f"✓ XGBoost trained successfully")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")

        return model

    def train_random_forest_with_adasyn(self, X_train, y_train, X_test, y_test):
        """Train Random Forest with ADASYN"""
        print("\n" + "=" * 80)
        print("TRAINING RANDOM FOREST + ADASYN")
        print("=" * 80)

        # Apply ADASYN
        print("Applying ADASYN...")
        try:
            adasyn = ADASYN(random_state=self.random_state, n_neighbors=2)
            X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train, y_train)
        except:
            # Fallback to SMOTE if ADASYN fails
            print("ADASYN failed, using SMOTE instead...")
            smote = SMOTE(random_state=self.random_state, k_neighbors=2)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        print(f"✓ After balancing:")
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} samples")

        # Random Forest
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )

        print("Training Random Forest...")
        model.fit(X_train_balanced, y_train_balanced)

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0

        self.models['RandomForest_ADASYN'] = model
        self.results['RandomForest_ADASYN'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        print(f"✓ Random Forest trained successfully")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")

        return model

    def train_lightgbm_with_smote(self, X_train, y_train, X_test, y_test):
        """Train LightGBM with SMOTE"""
        print("\n" + "=" * 80)
        print("TRAINING LIGHTGBM + SMOTE")
        print("=" * 80)

        # Apply SMOTE
        print("Applying SMOTE...")
        smote = SMOTE(random_state=self.random_state, k_neighbors=2)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        print(f"✓ After SMOTE:")
        unique, counts = np.unique(y_train_balanced, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"  Class {u}: {c} samples")

        # LightGBM
        model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            class_weight='balanced',
            random_state=self.random_state,
            verbose=-1
        )

        print("Training LightGBM...")
        model.fit(
            X_train_balanced, y_train_balanced,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0

        self.models['LightGBM_SMOTE'] = model
        self.results['LightGBM_SMOTE'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        print(f"✓ LightGBM trained successfully")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")

        return model

    def create_ensemble(self, X_train, y_train, X_test, y_test):
        """Create voting ensemble of best models"""
        print("\n" + "=" * 80)
        print("CREATING ENSEMBLE MODEL")
        print("=" * 80)

        # Apply SMOTE for ensemble
        smote = SMOTE(random_state=self.random_state, k_neighbors=2)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        # Base models
        xgb_model = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=self.random_state, eval_metric='mlogloss'
        )

        rf_model = RandomForestClassifier(
            n_estimators=300, max_depth=10, random_state=self.random_state
        )

        lgb_model = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=self.random_state, verbose=-1
        )

        # Voting ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('lgb', lgb_model)
            ],
            voting='soft'
        )

        print("Training ensemble...")
        ensemble.fit(X_train_balanced, y_train_balanced)

        # Predictions
        y_pred = ensemble.predict(X_test)
        y_pred_proba = ensemble.predict_proba(X_test)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc = 0.0

        self.models['Ensemble'] = ensemble
        self.results['Ensemble'] = {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc': auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

        print(f"✓ Ensemble trained successfully")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  AUC: {auc:.4f}")

        return ensemble

    def print_comparison(self):
        """Print model comparison"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)

        results_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Accuracy': [self.results[m]['accuracy'] for m in self.results.keys()],
            'F1-Score': [self.results[m]['f1_score'] for m in self.results.keys()],
            'AUC': [self.results[m]['auc'] for m in self.results.keys()]
        })

        results_df = results_df.sort_values('F1-Score', ascending=False)
        print(results_df.to_string(index=False))

        # Best model
        best_model = results_df.iloc[0]['Model']
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   F1-Score: {results_df.iloc[0]['F1-Score']:.4f}")

        return results_df

    def save_models(self):
        """Save all trained models"""
        print("\n" + "=" * 80)
        print("SAVING MODELS")
        print("=" * 80)

        for name, model in self.models.items():
            filename = f"model_{name.lower().replace(' ', '_')}.pkl"
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"✓ Saved: {filename}")

        # Save results
        with open('training_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        print(f"✓ Saved: training_results.pkl")

def main():
    """Main training pipeline"""
    print("\n" + "=" * 80)
    print("SOTA MODELS TRAINING - PREDIKSI RESIDIVIS")
    print("=" * 80)

    # Initialize
    trainer = SOTAModels(random_state=42)

    # Load data
    (X_train, X_test, y_train, y_test,
     X_train_cat, X_test_cat, y_train_cat, y_test_cat, cat_indices) = trainer.load_data()

    # Train models
    trainer.train_catboost(X_train_cat, y_train_cat, X_test_cat, y_test_cat, cat_indices)
    trainer.train_xgboost_with_smote(X_train, y_train, X_test, y_test)
    trainer.train_random_forest_with_adasyn(X_train, y_train, X_test, y_test)
    trainer.train_lightgbm_with_smote(X_train, y_train, X_test, y_test)
    trainer.create_ensemble(X_train, y_train, X_test, y_test)

    # Compare models
    results_df = trainer.print_comparison()
    results_df.to_csv('model_comparison.csv', index=False)
    print(f"\n✓ Saved: model_comparison.csv")

    # Save models
    trainer.save_models()

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
