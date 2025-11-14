"""
Exploratory Data Analysis (EDA) untuk Dataset Prediksi Residivis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def load_data():
    """Load dataset"""
    df = pd.read_csv('data.csv')
    print("=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"\nShape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    return df

def basic_info(df):
    """Informasi dasar dataset"""
    print("\n" + "=" * 80)
    print("BASIC INFORMATION")
    print("=" * 80)
    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("✓ No missing values found")
    else:
        print(missing[missing > 0])

    print("\nDuplicate Rows:")
    duplicates = df.duplicated().sum()
    print(f"Found {duplicates} duplicate rows")

def target_distribution(df):
    """Analisis distribusi target variable"""
    print("\n" + "=" * 80)
    print("TARGET VARIABLE DISTRIBUTION")
    print("=" * 80)

    target_counts = df['tingkat_risiko'].value_counts()
    target_pct = df['tingkat_risiko'].value_counts(normalize=True) * 100

    print("\nCounts:")
    print(target_counts)
    print("\nPercentage:")
    for level, pct in target_pct.items():
        print(f"{level}: {pct:.2f}%")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Count plot
    target_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#f39c12', '#e74c3c'])
    axes[0].set_title('Distribusi Tingkat Risiko (Count)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Tingkat Risiko', fontsize=12)
    axes[0].set_ylabel('Jumlah', fontsize=12)
    axes[0].tick_params(axis='x', rotation=0)

    # Pie chart
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    axes[1].pie(target_counts, labels=target_counts.index, autopct='%1.1f%%',
                colors=colors, startangle=90)
    axes[1].set_title('Proporsi Tingkat Risiko', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('output_eda_target_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: output_eda_target_distribution.png")
    plt.close()

def categorical_analysis(df):
    """Analisis fitur kategorikal"""
    print("\n" + "=" * 80)
    print("CATEGORICAL FEATURES ANALYSIS")
    print("=" * 80)

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.remove('tingkat_risiko')  # Remove target

    print(f"\nFound {len(categorical_cols)} categorical features")

    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())

def numerical_analysis(df):
    """Analisis fitur numerikal"""
    print("\n" + "=" * 80)
    print("NUMERICAL FEATURES ANALYSIS")
    print("=" * 80)

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove('id')  # Remove ID

    print("\nDescriptive Statistics:")
    print(df[numerical_cols].describe())

    # Visualization
    n_cols = len(numerical_cols)
    n_rows = (n_cols + 2) // 3

    fig, axes = plt.subplots(n_rows, 3, figsize=(15, n_rows * 4))
    axes = axes.flatten() if n_cols > 1 else [axes]

    for idx, col in enumerate(numerical_cols):
        df[col].hist(bins=15, ax=axes[idx], color='#3498db', edgecolor='black')
        axes[idx].set_title(f'Distribusi {col}', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col, fontsize=10)
        axes[idx].set_ylabel('Frequency', fontsize=10)

    # Hide empty subplots
    for idx in range(len(numerical_cols), len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('output_eda_numerical_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: output_eda_numerical_distribution.png")
    plt.close()

def correlation_analysis(df):
    """Analisis korelasi fitur numerikal"""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS")
    print("=" * 80)

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numerical_cols.remove('id')

    corr_matrix = df[numerical_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('Correlation Matrix - Numerical Features', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('output_eda_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: output_eda_correlation_matrix.png")
    plt.close()

def target_vs_features(df):
    """Analisis hubungan fitur dengan target"""
    print("\n" + "=" * 80)
    print("TARGET vs FEATURES ANALYSIS")
    print("=" * 80)

    # Kategorikal penting
    important_cat_features = [
        'status_sekolah', 'pendapatan_keluarga', 'struktur_keluarga',
        'dukungan_keluarga', 'pengaruh_teman_sebaya', 'pendampingan_pasca'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, col in enumerate(important_cat_features):
        ct = pd.crosstab(df[col], df['tingkat_risiko'], normalize='index') * 100
        ct.plot(kind='bar', ax=axes[idx], color=['#2ecc71', '#f39c12', '#e74c3c'])
        axes[idx].set_title(f'{col} vs Tingkat Risiko', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(col, fontsize=10)
        axes[idx].set_ylabel('Percentage (%)', fontsize=10)
        axes[idx].legend(title='Tingkat Risiko', fontsize=9)
        axes[idx].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('output_eda_target_vs_features.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: output_eda_target_vs_features.png")
    plt.close()

def risk_profile_analysis(df):
    """Analisis profil risiko"""
    print("\n" + "=" * 80)
    print("RISK PROFILE ANALYSIS")
    print("=" * 80)

    for risk_level in ['tinggi', 'sedang', 'rendah']:
        subset = df[df['tingkat_risiko'] == risk_level]
        print(f"\n{'='*50}")
        print(f"PROFIL RISIKO {risk_level.upper()} ({len(subset)} kasus)")
        print('='*50)

        print(f"\nStatus Sekolah:")
        print(subset['status_sekolah'].value_counts())

        print(f"\nDukungan Keluarga:")
        print(subset['dukungan_keluarga'].value_counts())

        print(f"\nFrekuensi Tindak Pidana (rata-rata): {subset['frekuensi_tindak_pidana'].mean():.2f}")

        print(f"\nPendampingan Pasca:")
        print(subset['pendampingan_pasca'].value_counts())

def main():
    """Main function"""
    print("\n" + "=" * 80)
    print("ANALISIS EKSPLORATORI DATA - PREDIKSI RESIDIVIS")
    print("=" * 80)

    # Load data
    df = load_data()

    # Run analyses
    basic_info(df)
    target_distribution(df)
    categorical_analysis(df)
    numerical_analysis(df)
    correlation_analysis(df)
    target_vs_features(df)
    risk_profile_analysis(df)

    print("\n" + "=" * 80)
    print("EDA COMPLETED")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - output_eda_target_distribution.png")
    print("  - output_eda_numerical_distribution.png")
    print("  - output_eda_correlation_matrix.png")
    print("  - output_eda_target_vs_features.png")

if __name__ == "__main__":
    main()
