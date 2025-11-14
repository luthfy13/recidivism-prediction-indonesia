# Prediksi Residivis - Anak Berhadapan dengan Hukum (ABH)

Sistem prediksi risiko residivis menggunakan **State-of-the-Art (SOTA)** machine learning models untuk anak yang berhadapan dengan hukum di Indonesia.

## 🎯 Tujuan

Memprediksi tingkat risiko residivis (rendah, sedang, tinggi) berdasarkan:
- Karakteristik demografis
- Kondisi keluarga dan lingkungan
- Riwayat tindak pidana
- Intervensi dan rehabilitasi
- Dukungan pasca rehabilitasi

## 🏆 Model State-of-the-Art

Project ini mengimplementasikan berbagai SOTA models:

### 1. **TabPFN** (Recommended untuk Small Dataset)
- Foundation model terbaru (Nature 2025)
- Zero hyperparameter tuning
- Optimal untuk dataset <10,000 samples
- Training time: detik vs jam untuk model tradisional

### 2. **CatBoost**
- Terbaik untuk categorical features
- Built-in handling untuk imbalanced data
- Auto-balancing dengan class weights

### 3. **XGBoost + SMOTE**
- Gradient boosting dengan SMOTE oversampling
- Robust performance

### 4. **Random Forest + ADASYN**
- Ensemble learning dengan ADASYN balancing
- Resistant to overfitting

### 5. **LightGBM + SMOTE**
- Fast gradient boosting
- Memory efficient

### 6. **Ensemble (Voting)**
- Kombinasi XGBoost, Random Forest, LightGBM
- Voting classifier untuk maximum stability

## 📊 Performa Benchmark

Berdasarkan systematic review untuk recidivism prediction (2025):
- Average Accuracy: **81%**
- Average AUC: **0.74-0.85**
- Best methods: Gradient Boosted Trees & Random Forest

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd prediksi-residivis

# Install dependencies
pip install -r requirements.txt

# Optional: Install TabPFN (requires PyTorch)
pip install torch
pip install tabpfn
```

### Usage Pipeline

```bash
# 1. Exploratory Data Analysis
python 1_eda_analysis.py

# 2. Preprocessing & Feature Engineering
python 2_preprocessing.py

# 3. Train SOTA Models
python 3_train_models.py

# 4. Train TabPFN (optional, tapi recommended!)
python 4_train_tabpfn.py

# 5. Comprehensive Evaluation
python 5_evaluation_comparison.py

# 6. Inference untuk Data Baru
python 6_inference.py
```

## 📁 Project Structure

```
prediksi-residivis/
├── data.csv                          # Dataset
├── 1_eda_analysis.py                 # Exploratory Data Analysis
├── 2_preprocessing.py                # Data Preprocessing & Feature Engineering
├── 3_train_models.py                 # Training SOTA Models
├── 4_train_tabpfn.py                 # Training TabPFN
├── 5_evaluation_comparison.py        # Model Evaluation & Comparison
├── 6_inference.py                    # Inference untuk Prediksi Baru
├── requirements.txt                  # Dependencies
└── README.md                         # Documentation
```

## 📈 Dataset Features

### Demografis
- Usia (13-17 tahun)
- Jenis kelamin
- Pendidikan terakhir & status sekolah

### Keluarga
- Pekerjaan orang tua
- Pendapatan keluarga
- Struktur keluarga (utuh/broken home/yatim/piatu)
- Tingkat dukungan keluarga

### Lingkungan
- Lingkungan tempat tinggal (kumuh/perumahan/elite)
- Pengaruh teman sebaya

### Tindak Pidana
- Jenis tindak pidana
- Frekuensi
- Usia pertama kali
- Lama pidana

### Intervensi
- Jenis intervensi (LPKA/diversi/pidana bersyarat)
- Durasi rehabilitasi
- Kepatuhan rehabilitasi

### Pasca Rehabilitasi
- Pendampingan pasca
- Akses pendidikan pasca
- Akses pekerjaan pasca

### Target Variable
- **Tingkat Risiko**: rendah, sedang, tinggi

## 🔧 Feature Engineering

Script otomatis membuat 11 fitur baru:
1. Risk score berbasis frekuensi
2. Early crime indicator (usia ≤14)
3. Support system score
4. Socioeconomic indicator
5. Family stability score
6. Educational dropout indicator
7. Environmental risk
8. Negative peers influence
9. Post-intervention support count
10. Rehabilitation effectiveness score
11. Composite risk score

## 🎨 Visualizations Generated

### EDA (1_eda_analysis.py)
- `output_eda_target_distribution.png`
- `output_eda_numerical_distribution.png`
- `output_eda_correlation_matrix.png`
- `output_eda_target_vs_features.png`

### Evaluation (5_evaluation_comparison.py)
- `output_model_comparison.png`
- `output_confusion_matrices.png`
- `output_roc_curves.png`
- `output_per_class_performance.png`

## 📊 Model Files Generated

```
model_catboost.pkl
model_xgboost_smote.pkl
model_randomforest_adasyn.pkl
model_lightgbm_smote.pkl
model_ensemble.pkl
model_tabpfn.pkl              # If TabPFN installed
```

## 💡 Inference Example

```python
from inference import ResidivismPredictor, print_prediction_result

# Initialize predictor (auto-loads best model)
predictor = ResidivismPredictor()

# Prepare data
case = {
    'usia': 15,
    'jenis_kelamin': 'L',
    'pendidikan_terakhir': 'SMP',
    'status_sekolah': 'putus_sekolah',
    'pekerjaan_ortu': 'buruh',
    'pendapatan_keluarga': 'rendah',
    'struktur_keluarga': 'broken_home',
    'dukungan_keluarga': 'rendah',
    'lingkungan_tempat_tinggal': 'kumuh',
    'pengaruh_teman_sebaya': 'negatif',
    'jenis_tindak_pidana': 'pencurian',
    'frekuensi_tindak_pidana': 3,
    'usia_pertama_kali': 13,
    'lama_pidana_bulan': 6,
    'jenis_intervensi': 'LPKA',
    'durasi_rehabilitasi_bulan': 3,
    'kepatuhan_rehabilitasi': 'rendah',
    'pendampingan_pasca': 'tidak',
    'akses_pendidikan_pasca': 'tidak',
    'akses_pekerjaan_pasca': 'tidak'
}

# Predict with explanation
result = predictor.predict_with_explanation(case)
print_prediction_result(result)
```

Output example:
```
================================================================================
HASIL PREDIKSI
================================================================================

🎯 Prediksi Tingkat Risiko: TINGGI
📊 Confidence: 87.34%
🤖 Model: CatBoost

📈 Probabilitas untuk setiap kelas:
  rendah  : ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  5.23%
  sedang  : ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  7.43%
  tinggi  : ███████████████████████████████████░░░░░ 87.34%

⚠️  Faktor Risiko:

  Risiko Tinggi:
    ❌ Putus sekolah
    ❌ Dukungan keluarga rendah
    ❌ Struktur keluarga tidak utuh
    ❌ Frekuensi tinggi (3x)
    ❌ Pengaruh teman sebaya negatif

  Risiko Sedang:
    ⚠️  Tidak ada pendampingan pasca
    ⚠️  Lingkungan kumuh
```

## 🔬 Key Technical Features

### Handling Imbalanced Data
- **SMOTE** (Synthetic Minority Over-sampling Technique)
- **ADASYN** (Adaptive Synthetic Sampling)
- **Class Weights** (CatBoost auto-balancing)

### Cross-Validation
- Stratified K-Fold CV
- Preserves class distribution

### Regularization
- L1/L2 regularization in all models
- Early stopping to prevent overfitting

### Algorithmic Fairness
- Important untuk recidivism prediction
- Evaluation per-class metrics
- Confusion matrix analysis

## 📚 References

### State-of-the-Art Research (2025)
1. **TabPFN**: Hollmann et al., "TabPFN: A Foundation Model for Tabular Data", *Nature*, 2024-2025
2. **Recidivism ML**: Guo, "Recidivism prediction: A machine learning approach", *SAGE*, 2025
3. **Tabular Data SOTA**: "When Do Neural Nets Outperform Boosted Trees on Tabular Data?", *arXiv*, 2025

### Key Findings
- TabPFN outperforms baselines by wide margin on small datasets
- Gradient boosting (CatBoost, XGBoost, LightGBM) consistently reliable
- Average recidivism prediction AUC: 0.74-0.85
- ML-based models require bias mitigation procedures

## ⚠️ Important Notes

### Data Privacy & Ethics
- Dataset ini untuk tujuan penelitian dan pendidikan
- Perhatikan aspek privasi saat menggunakan data riil
- Implementasikan bias mitigation untuk fairness

### Model Limitations
- Dataset sangat kecil (25 samples) - hasil untuk demonstrasi
- Untuk production, gunakan dataset lebih besar (>1000 samples)
- Perlu validasi eksternal dengan data independen

### Production Deployment
- Lakukan extensive testing sebelum deployment
- Monitor model performance secara berkala
- Implement human-in-the-loop untuk keputusan penting

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Larger dataset collection
- Additional feature engineering
- Hyperparameter optimization
- Explainable AI (SHAP, LIME)
- Web interface development
- API deployment

## 📝 License

This project is for educational and research purposes.

## 👥 Authors

Machine Learning Project - Prediksi Residivis ABH

## 📧 Contact

For questions or collaborations, please open an issue.

---

**⭐ Star this repo if you find it useful!**
