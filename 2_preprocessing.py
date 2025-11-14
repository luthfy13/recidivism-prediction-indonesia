"""
Preprocessing dan Feature Engineering untuk Prediksi Residivis
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from pathlib import Path

class ResidivismPreprocessor:
    """Preprocessor untuk dataset prediksi residivis"""

    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_encoder = LabelEncoder()

    def load_data(self, filepath='data.csv'):
        """Load dataset"""
        df = pd.read_csv(filepath)
        print(f"✓ Loaded dataset: {df.shape}")
        return df

    def create_features(self, df):
        """Feature engineering"""
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING")
        print("=" * 80)

        df = df.copy()

        # 1. Risk score berbasis frekuensi dan usia pertama
        df['risk_score_frequency'] = df['frekuensi_tindak_pidana'] * 2
        df['early_crime_indicator'] = (df['usia_pertama_kali'] <= 14).astype(int)

        # 2. Support system score
        support_mapping = {'rendah': 0, 'sedang': 1, 'tinggi': 2}
        df['dukungan_keluarga_encoded'] = df['dukungan_keluarga'].map(support_mapping)

        # 3. Socioeconomic indicator
        income_mapping = {'rendah': 0, 'menengah': 1, 'tinggi': 2}
        df['pendapatan_encoded'] = df['pendapatan_keluarga'].map(income_mapping)

        # 4. Family stability score
        df['family_broken'] = df['struktur_keluarga'].isin(['broken_home', 'yatim', 'piatu']).astype(int)

        # 5. Educational status
        df['dropout_indicator'] = (df['status_sekolah'] == 'putus_sekolah').astype(int)

        # 6. Environmental risk
        df['high_risk_environment'] = (df['lingkungan_tempat_tinggal'] == 'kumuh').astype(int)
        df['negative_peers'] = (df['pengaruh_teman_sebaya'] == 'negatif').astype(int)

        # 7. Post-intervention support
        df['post_support_count'] = (
            df['pendampingan_pasca'].map({'ya': 1, 'tidak': 0}) +
            df['akses_pendidikan_pasca'].map({'ya': 1, 'tidak': 0}) +
            df['akses_pekerjaan_pasca'].map({'ya': 1, 'tidak': 0})
        )

        # 8. Rehabilitation effectiveness
        kepatuhan_mapping = {'rendah': 0, 'sedang': 1, 'tinggi': 2}
        df['kepatuhan_encoded'] = df['kepatuhan_rehabilitasi'].map(kepatuhan_mapping)
        df['rehab_score'] = df['kepatuhan_encoded'] * df['durasi_rehabilitasi_bulan'] / 12

        # 9. Crime severity proxy
        severe_crimes = ['narkoba', 'penganiayaan', 'pencabulan']
        df['severe_crime'] = df['jenis_tindak_pidana'].isin(severe_crimes).astype(int)

        # 10. Age-related features
        df['age_group'] = pd.cut(df['usia'], bins=[0, 14, 16, 18],
                                  labels=['early_teen', 'mid_teen', 'late_teen'])

        # 11. Composite risk score
        df['composite_risk_score'] = (
            df['risk_score_frequency'] +
            df['early_crime_indicator'] * 3 +
            df['family_broken'] * 2 +
            df['dropout_indicator'] * 2 +
            df['high_risk_environment'] * 2 +
            df['negative_peers'] * 2 -
            df['post_support_count'] * 2 -
            df['rehab_score'] * 2
        )

        print(f"✓ Created {df.shape[1] - 22} new features")
        print(f"  Total features now: {df.shape[1]}")

        return df

    def encode_features(self, df, fit=True):
        """Encode categorical features"""
        print("\n" + "=" * 80)
        print("ENCODING CATEGORICAL FEATURES")
        print("=" * 80)

        df = df.copy()

        # Categorical columns to encode
        categorical_cols = [
            'jenis_kelamin', 'pendidikan_terakhir', 'status_sekolah',
            'pekerjaan_ortu', 'pendapatan_keluarga', 'struktur_keluarga',
            'dukungan_keluarga', 'lingkungan_tempat_tinggal',
            'pengaruh_teman_sebaya', 'jenis_tindak_pidana',
            'jenis_intervensi', 'kepatuhan_rehabilitasi',
            'pendampingan_pasca', 'akses_pendidikan_pasca',
            'akses_pekerjaan_pasca', 'age_group'
        ]

        for col in categorical_cols:
            if col not in df.columns:
                continue

            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                if col in self.label_encoders:
                    # Handle unseen labels
                    le = self.label_encoders[col]
                    df[f'{col}_encoded'] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        print(f"✓ Encoded {len(categorical_cols)} categorical features")

        return df

    def prepare_data(self, df, target_col='tingkat_risiko', test_size=0.2, random_state=42):
        """Prepare data for modeling"""
        print("\n" + "=" * 80)
        print("PREPARING DATA FOR MODELING")
        print("=" * 80)

        df = df.copy()

        # Drop ID column
        if 'id' in df.columns:
            df = df.drop('id', axis=1)

        # Separate features and target
        X = df.drop(target_col, axis=1)
        y = df[target_col]

        # Encode target
        y_encoded = self.target_encoder.fit_transform(y)

        # Select only encoded features and engineered numerical features
        feature_cols = [col for col in X.columns if
                       col.endswith('_encoded') or
                       col in ['usia', 'frekuensi_tindak_pidana', 'usia_pertama_kali',
                              'lama_pidana_bulan', 'durasi_rehabilitasi_bulan',
                              'risk_score_frequency', 'early_crime_indicator',
                              'family_broken', 'dropout_indicator',
                              'high_risk_environment', 'negative_peers',
                              'post_support_count', 'rehab_score',
                              'severe_crime', 'composite_risk_score',
                              'dukungan_keluarga_encoded', 'pendapatan_encoded',
                              'kepatuhan_encoded']]

        X_selected = X[feature_cols]
        self.feature_names = feature_cols

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y_encoded, test_size=test_size,
            random_state=random_state, stratify=y_encoded
        )

        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        print(f"✓ Number of features: {len(feature_cols)}")
        print(f"\nFeatures used:")
        for i, feat in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {feat}")

        return X_train, X_test, y_train, y_test

    def get_original_data_for_catboost(self, df, target_col='tingkat_risiko'):
        """Prepare data dengan categorical features untuk CatBoost"""
        print("\n" + "=" * 80)
        print("PREPARING DATA FOR CATBOOST (with categorical features)")
        print("=" * 80)

        df = df.copy()

        # Drop ID
        if 'id' in df.columns:
            df = df.drop('id', axis=1)

        # Get target
        y = df[target_col]
        y_encoded = self.target_encoder.fit_transform(y)

        # Get features (exclude original categorical that are already processed)
        X = df.drop(target_col, axis=1)

        # Categorical features for CatBoost
        cat_features = [
            'jenis_kelamin', 'pendidikan_terakhir', 'status_sekolah',
            'pekerjaan_ortu', 'pendapatan_keluarga', 'struktur_keluarga',
            'dukungan_keluarga', 'lingkungan_tempat_tinggal',
            'pengaruh_teman_sebaya', 'jenis_tindak_pidana',
            'jenis_intervensi', 'kepatuhan_rehabilitasi',
            'pendampingan_pasca', 'akses_pendidikan_pasca',
            'akses_pekerjaan_pasca', 'age_group'
        ]

        # Keep only existing cat features
        cat_features = [f for f in cat_features if f in X.columns]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        cat_indices = [X.columns.get_loc(f) for f in cat_features]

        print(f"✓ Training set: {X_train.shape}")
        print(f"✓ Test set: {X_test.shape}")
        print(f"✓ Categorical features: {len(cat_features)}")

        return X_train, X_test, y_train, y_test, cat_indices

    def save_preprocessor(self, filepath='preprocessor.pkl'):
        """Save preprocessor"""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"\n✓ Saved preprocessor to {filepath}")

    @staticmethod
    def load_preprocessor(filepath='preprocessor.pkl'):
        """Load preprocessor"""
        with open(filepath, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"✓ Loaded preprocessor from {filepath}")
        return preprocessor

def main():
    """Main preprocessing pipeline"""
    print("\n" + "=" * 80)
    print("PREPROCESSING PIPELINE - PREDIKSI RESIDIVIS")
    print("=" * 80)

    # Initialize preprocessor
    preprocessor = ResidivismPreprocessor()

    # Load data
    df = preprocessor.load_data('data.csv')

    # Feature engineering
    df_engineered = preprocessor.create_features(df)

    # Encode features
    df_encoded = preprocessor.encode_features(df_engineered, fit=True)

    # Prepare data (encoded version)
    X_train, X_test, y_train, y_test = preprocessor.prepare_data(df_encoded)

    # Save processed data
    np.save('X_train.npy', X_train.values)
    np.save('X_test.npy', X_test.values)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)

    print("\n✓ Saved processed data:")
    print("  - X_train.npy")
    print("  - X_test.npy")
    print("  - y_train.npy")
    print("  - y_test.npy")

    # Prepare CatBoost version
    X_train_cat, X_test_cat, y_train_cat, y_test_cat, cat_indices = \
        preprocessor.get_original_data_for_catboost(df_engineered)

    # Save CatBoost data
    X_train_cat.to_pickle('X_train_catboost.pkl')
    X_test_cat.to_pickle('X_test_catboost.pkl')
    np.save('y_train_catboost.npy', y_train_cat)
    np.save('y_test_catboost.npy', y_test_cat)
    np.save('cat_indices.npy', cat_indices)

    print("\n✓ Saved CatBoost data:")
    print("  - X_train_catboost.pkl")
    print("  - X_test_catboost.pkl")
    print("  - y_train_catboost.npy")
    print("  - y_test_catboost.npy")
    print("  - cat_indices.npy")

    # Save preprocessor
    preprocessor.save_preprocessor('preprocessor.pkl')

    # Show label mapping
    print("\n" + "=" * 80)
    print("TARGET LABEL MAPPING")
    print("=" * 80)
    for idx, label in enumerate(preprocessor.target_encoder.classes_):
        print(f"  {idx} -> {label}")

    print("\n" + "=" * 80)
    print("PREPROCESSING COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
