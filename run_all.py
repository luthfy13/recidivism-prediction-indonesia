"""
Master Script - Run All Pipeline
Jalankan seluruh pipeline training dan evaluation
"""

import subprocess
import sys
from pathlib import Path
import time

def run_script(script_name, description):
    """Run a Python script and track execution"""
    print("\n" + "=" * 80)
    print(f"RUNNING: {description}")
    print(f"Script: {script_name}")
    print("=" * 80)

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time
        print(f"\n✓ {script_name} completed successfully in {elapsed:.2f}s")
        return True

    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f"\n❌ {script_name} failed after {elapsed:.2f}s")
        print(f"Error: {e}")
        return False

def main():
    """Run complete pipeline"""
    print("\n" + "=" * 80)
    print("PREDIKSI RESIDIVIS - COMPLETE PIPELINE")
    print("=" * 80)
    print("\nThis will run the entire ML pipeline:")
    print("  1. Exploratory Data Analysis")
    print("  2. Data Preprocessing & Feature Engineering")
    print("  3. Model Training (SOTA Models)")
    print("  4. TabPFN Training (if available)")
    print("  5. Model Evaluation & Comparison")
    print("  6. Inference Examples")
    print("\n" + "=" * 80)

    # Check if data exists
    if not Path('data.csv').exists():
        print("\n❌ Error: data.csv not found!")
        print("Please ensure data.csv is in the current directory.")
        return

    response = input("\nProceed with pipeline execution? (y/n): ")
    if response.lower() != 'y':
        print("Pipeline execution cancelled.")
        return

    pipeline_start = time.time()
    results = {}

    # Pipeline steps
    steps = [
        ('1_eda_analysis.py', 'Exploratory Data Analysis'),
        ('2_preprocessing.py', 'Data Preprocessing & Feature Engineering'),
        ('3_train_models.py', 'Training SOTA Models'),
        ('4_train_tabpfn.py', 'Training TabPFN (optional)'),
        ('5_evaluation_comparison.py', 'Model Evaluation & Comparison'),
        ('6_inference.py', 'Inference Examples')
    ]

    # Execute pipeline
    for script, description in steps:
        success = run_script(script, description)
        results[script] = success

        # Continue even if TabPFN fails (it's optional)
        if not success and script != '4_train_tabpfn.py':
            print(f"\n⚠️  Pipeline stopped at {script}")
            break

    # Summary
    total_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)

    for script, success in results.items():
        status = "✓ SUCCESS" if success else "❌ FAILED"
        print(f"{status:12s} - {script}")

    print(f"\nTotal execution time: {total_elapsed:.2f}s ({total_elapsed/60:.2f} minutes)")

    # Check generated files
    print("\n" + "=" * 80)
    print("GENERATED FILES")
    print("=" * 80)

    expected_files = {
        'Data': [
            'X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy',
            'X_train_catboost.pkl', 'X_test_catboost.pkl',
            'preprocessor.pkl'
        ],
        'Models': [
            'model_catboost.pkl', 'model_xgboost_smote.pkl',
            'model_randomforest_adasyn.pkl', 'model_lightgbm_smote.pkl',
            'model_ensemble.pkl'
        ],
        'Results': [
            'model_comparison.csv', 'summary_comparison.csv',
            'evaluation_report.txt'
        ],
        'Visualizations': [
            'output_eda_target_distribution.png',
            'output_eda_numerical_distribution.png',
            'output_eda_correlation_matrix.png',
            'output_eda_target_vs_features.png',
            'output_model_comparison.png',
            'output_confusion_matrices.png',
            'output_roc_curves.png',
            'output_per_class_performance.png'
        ]
    }

    for category, files in expected_files.items():
        print(f"\n{category}:")
        for filename in files:
            exists = Path(filename).exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {filename}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Review evaluation_report.txt for detailed results")
    print("  2. Check visualizations (output_*.png)")
    print("  3. Use 6_inference.py for predictions on new data")
    print("  4. Read README.md for usage examples")

if __name__ == "__main__":
    main()
