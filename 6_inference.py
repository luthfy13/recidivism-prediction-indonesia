"""
Model Inference - Prediksi untuk Data Baru
Script untuk melakukan prediksi menggunakan model terbaik yang sudah dilatih
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ResidivismPredictor:
    """Predictor untuk prediksi residivis pada data baru"""

    def __init__(self, model_path=None, preprocessor_path='preprocessor.pkl'):
        """
        Initialize predictor

        Args:
            model_path: Path ke model yang akan digunakan (None = auto-select best)
            preprocessor_path: Path ke preprocessor
        """
        self.model = None
        self.preprocessor = None
        self.model_name = None

        # Load preprocessor
        self.load_preprocessor(preprocessor_path)

        # Load model
        if model_path:
            self.load_model(model_path)
        else:
            self.auto_select_best_model()

    def load_preprocessor(self, path):
        """Load preprocessor"""
        try:
            with open(path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            print(f"✓ Loaded preprocessor from {path}")
        except FileNotFoundError:
            print(f"❌ Preprocessor not found: {path}")
            raise

    def load_model(self, path):
        """Load model from file"""
        try:
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
            self.model_name = Path(path).stem
            print(f"✓ Loaded model from {path}")
        except FileNotFoundError:
            print(f"❌ Model not found: {path}")
            raise

    def auto_select_best_model(self):
        """Automatically select best model based on comparison results"""
        print("\n" + "=" * 80)
        print("AUTO-SELECTING BEST MODEL")
        print("=" * 80)

        try:
            # Try to load comparison results
            comparison_path = 'model_comparison_with_tabpfn.csv'
            if not Path(comparison_path).exists():
                comparison_path = 'model_comparison.csv'

            df = pd.read_csv(comparison_path)
            df = df.sort_values('F1-Score', ascending=False)

            best_model_name = df.iloc[0]['Model']
            best_f1 = df.iloc[0]['F1-Score']

            print(f"\nBest model: {best_model_name}")
            print(f"F1-Score: {best_f1:.4f}")

            # Map model name to file
            model_files = {
                'CatBoost': 'model_catboost.pkl',
                'XGBoost_SMOTE': 'model_xgboost_smote.pkl',
                'XGBoost+SMOTE': 'model_xgboost_smote.pkl',
                'RandomForest_ADASYN': 'model_randomforest_adasyn.pkl',
                'RandomForest+ADASYN': 'model_randomforest_adasyn.pkl',
                'LightGBM_SMOTE': 'model_lightgbm_smote.pkl',
                'LightGBM+SMOTE': 'model_lightgbm_smote.pkl',
                'Ensemble': 'model_ensemble.pkl',
                'TabPFN': 'model_tabpfn.pkl'
            }

            model_path = model_files.get(best_model_name)
            if model_path and Path(model_path).exists():
                self.load_model(model_path)
            else:
                # Fallback to any available model
                for name, path in model_files.items():
                    if Path(path).exists():
                        print(f"⚠️  Best model not found, using: {name}")
                        self.load_model(path)
                        break

        except Exception as e:
            print(f"⚠️  Could not auto-select: {e}")
            print("Please specify model_path manually")
            raise

    def preprocess_input(self, data):
        """
        Preprocess input data

        Args:
            data: DataFrame or dict with input features

        Returns:
            Preprocessed features ready for prediction
        """
        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # Feature engineering
        data = self.preprocessor.create_features(data)

        # Encode features
        data = self.preprocessor.encode_features(data, fit=False)

        # Select features
        if self.preprocessor.feature_names:
            # Check if all required features exist
            missing_features = [f for f in self.preprocessor.feature_names
                              if f not in data.columns]
            if missing_features:
                print(f"⚠️  Missing features: {missing_features}")
                # Fill with defaults
                for feat in missing_features:
                    data[feat] = 0

            data = data[self.preprocessor.feature_names]

        return data

    def predict(self, data, return_proba=False):
        """
        Make prediction

        Args:
            data: Input data (DataFrame or dict)
            return_proba: If True, return probabilities instead of class

        Returns:
            Prediction (class label or probabilities)
        """
        if self.model is None:
            raise ValueError("No model loaded")

        # Preprocess
        X = self.preprocess_input(data)

        # Predict
        if return_proba:
            proba = self.model.predict_proba(X)
            return proba
        else:
            pred = self.model.predict(X)
            # Decode labels
            pred_labels = self.preprocessor.target_encoder.inverse_transform(pred)
            return pred_labels

    def predict_with_explanation(self, data):
        """
        Predict with detailed explanation

        Args:
            data: Input data (DataFrame or dict)

        Returns:
            Dictionary with prediction, probabilities, and risk factors
        """
        # Get probabilities
        proba = self.predict(data, return_proba=True)

        # Get class prediction
        pred = self.predict(data, return_proba=False)

        # Class names
        class_names = self.preprocessor.target_encoder.classes_

        # Create result
        result = {
            'prediction': pred[0],
            'probabilities': {
                class_names[i]: float(proba[0, i])
                for i in range(len(class_names))
            },
            'confidence': float(proba[0].max()),
            'model_used': self.model_name
        }

        # Analyze risk factors
        if isinstance(data, dict):
            data_df = pd.DataFrame([data])
        else:
            data_df = data

        risk_factors = self._analyze_risk_factors(data_df)
        result['risk_factors'] = risk_factors

        return result

    def _analyze_risk_factors(self, data):
        """Analyze key risk factors from input data"""
        risk_factors = {
            'high_risk': [],
            'moderate_risk': [],
            'protective': []
        }

        # Get first row
        row = data.iloc[0]

        # Check various risk factors
        if 'status_sekolah' in row:
            if row['status_sekolah'] == 'putus_sekolah':
                risk_factors['high_risk'].append('Putus sekolah')
            else:
                risk_factors['protective'].append('Masih bersekolah')

        if 'dukungan_keluarga' in row:
            if row['dukungan_keluarga'] == 'rendah':
                risk_factors['high_risk'].append('Dukungan keluarga rendah')
            elif row['dukungan_keluarga'] == 'tinggi':
                risk_factors['protective'].append('Dukungan keluarga tinggi')

        if 'struktur_keluarga' in row:
            if row['struktur_keluarga'] in ['broken_home', 'yatim', 'piatu']:
                risk_factors['high_risk'].append('Struktur keluarga tidak utuh')

        if 'frekuensi_tindak_pidana' in row:
            if row['frekuensi_tindak_pidana'] >= 3:
                risk_factors['high_risk'].append(f'Frekuensi tinggi ({int(row["frekuensi_tindak_pidana"])}x)')
            elif row['frekuensi_tindak_pidana'] == 1:
                risk_factors['protective'].append('Frekuensi rendah (1x)')

        if 'pengaruh_teman_sebaya' in row:
            if row['pengaruh_teman_sebaya'] == 'negatif':
                risk_factors['high_risk'].append('Pengaruh teman sebaya negatif')
            elif row['pengaruh_teman_sebaya'] == 'positif':
                risk_factors['protective'].append('Pengaruh teman sebaya positif')

        if 'pendampingan_pasca' in row:
            if row['pendampingan_pasca'] == 'tidak':
                risk_factors['moderate_risk'].append('Tidak ada pendampingan pasca')
            else:
                risk_factors['protective'].append('Ada pendampingan pasca')

        if 'lingkungan_tempat_tinggal' in row:
            if row['lingkungan_tempat_tinggal'] == 'kumuh':
                risk_factors['moderate_risk'].append('Lingkungan kumuh')

        return risk_factors

def print_prediction_result(result):
    """Pretty print prediction result"""
    print("\n" + "=" * 80)
    print("HASIL PREDIKSI")
    print("=" * 80)

    print(f"\n🎯 Prediksi Tingkat Risiko: {result['prediction'].upper()}")
    print(f"📊 Confidence: {result['confidence']*100:.2f}%")
    print(f"🤖 Model: {result['model_used']}")

    print("\n📈 Probabilitas untuk setiap kelas:")
    for class_name, prob in result['probabilities'].items():
        bar_length = int(prob * 40)
        bar = '█' * bar_length + '░' * (40 - bar_length)
        print(f"  {class_name:8s}: {bar} {prob*100:5.2f}%")

    print("\n⚠️  Faktor Risiko:")

    if result['risk_factors']['high_risk']:
        print("\n  Risiko Tinggi:")
        for factor in result['risk_factors']['high_risk']:
            print(f"    ❌ {factor}")

    if result['risk_factors']['moderate_risk']:
        print("\n  Risiko Sedang:")
        for factor in result['risk_factors']['moderate_risk']:
            print(f"    ⚠️  {factor}")

    if result['risk_factors']['protective']:
        print("\n  Faktor Protektif:")
        for factor in result['risk_factors']['protective']:
            print(f"    ✅ {factor}")

    print("\n" + "=" * 80)

def example_usage():
    """Example of how to use the predictor"""
    print("\n" + "=" * 80)
    print("EXAMPLE: PREDIKSI UNTUK DATA BARU")
    print("=" * 80)

    # Create predictor
    predictor = ResidivismPredictor()

    # Example 1: High risk case
    print("\n\nCONTOH 1: Kasus Risiko Tinggi")
    print("-" * 80)

    high_risk_case = {
        'id': 999,
        'usia': 15,
        'jenis_kelamin': 'L',
        'pendidikan_terakhir': 'SMP',
        'status_sekolah': 'putus_sekolah',
        'pekerjaan_ortu': 'tidak_bekerja',
        'pendapatan_keluarga': 'rendah',
        'struktur_keluarga': 'broken_home',
        'dukungan_keluarga': 'rendah',
        'lingkungan_tempat_tinggal': 'kumuh',
        'pengaruh_teman_sebaya': 'negatif',
        'jenis_tindak_pidana': 'narkoba',
        'frekuensi_tindak_pidana': 4,
        'usia_pertama_kali': 13,
        'lama_pidana_bulan': 12,
        'jenis_intervensi': 'LPKA',
        'durasi_rehabilitasi_bulan': 3,
        'kepatuhan_rehabilitasi': 'rendah',
        'pendampingan_pasca': 'tidak',
        'akses_pendidikan_pasca': 'tidak',
        'akses_pekerjaan_pasca': 'tidak'
    }

    result = predictor.predict_with_explanation(high_risk_case)
    print_prediction_result(result)

    # Example 2: Low risk case
    print("\n\nCONTOH 2: Kasus Risiko Rendah")
    print("-" * 80)

    low_risk_case = {
        'id': 998,
        'usia': 16,
        'jenis_kelamin': 'P',
        'pendidikan_terakhir': 'SMA',
        'status_sekolah': 'aktif',
        'pekerjaan_ortu': 'PNS',
        'pendapatan_keluarga': 'tinggi',
        'struktur_keluarga': 'utuh',
        'dukungan_keluarga': 'tinggi',
        'lingkungan_tempat_tinggal': 'perumahan',
        'pengaruh_teman_sebaya': 'positif',
        'jenis_tindak_pidana': 'penganiayaan',
        'frekuensi_tindak_pidana': 1,
        'usia_pertama_kali': 16,
        'lama_pidana_bulan': 0,
        'jenis_intervensi': 'diversi',
        'durasi_rehabilitasi_bulan': 6,
        'kepatuhan_rehabilitasi': 'tinggi',
        'pendampingan_pasca': 'ya',
        'akses_pendidikan_pasca': 'ya',
        'akses_pekerjaan_pasca': 'ya'
    }

    result = predictor.predict_with_explanation(low_risk_case)
    print_prediction_result(result)

def batch_predict_from_csv(input_file, output_file='predictions.csv'):
    """
    Batch prediction from CSV file

    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file with predictions
    """
    print("\n" + "=" * 80)
    print(f"BATCH PREDICTION FROM: {input_file}")
    print("=" * 80)

    # Load data
    data = pd.read_csv(input_file)
    print(f"✓ Loaded {len(data)} cases")

    # Create predictor
    predictor = ResidivismPredictor()

    # Predict
    predictions = []
    probabilities = []

    for idx, row in data.iterrows():
        pred = predictor.predict(row.to_dict(), return_proba=False)
        proba = predictor.predict(row.to_dict(), return_proba=True)

        predictions.append(pred[0])
        probabilities.append(proba[0])

    # Add to dataframe
    data['predicted_risk'] = predictions

    class_names = predictor.preprocessor.target_encoder.classes_
    for i, class_name in enumerate(class_names):
        data[f'prob_{class_name}'] = [p[i] for p in probabilities]

    # Save
    data.to_csv(output_file, index=False)
    print(f"\n✓ Saved predictions to: {output_file}")

    # Summary
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    print(data['predicted_risk'].value_counts())

def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("INFERENCE - PREDIKSI RESIDIVIS")
    print("=" * 80)

    # Run examples
    example_usage()

    print("\n" + "=" * 80)
    print("CARA PENGGUNAAN")
    print("=" * 80)
    print("""
Untuk prediksi data baru:

1. Import module:
   from inference import ResidivismPredictor, print_prediction_result

2. Buat predictor:
   predictor = ResidivismPredictor()

3. Siapkan data (dictionary dengan fitur yang diperlukan):
   data = {
       'usia': 15,
       'jenis_kelamin': 'L',
       'status_sekolah': 'aktif',
       ...
   }

4. Prediksi:
   result = predictor.predict_with_explanation(data)
   print_prediction_result(result)

5. Untuk batch prediction dari CSV:
   batch_predict_from_csv('data_baru.csv', 'hasil_prediksi.csv')
""")

if __name__ == "__main__":
    main()
