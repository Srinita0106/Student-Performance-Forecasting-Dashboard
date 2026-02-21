"""
Main Orchestration Script
Runs all modules in sequence for complete pipeline execution.
"""

import os
import sys
import subprocess
import warnings
warnings.filterwarnings('ignore')

def run_module(module_name, description):
    """Run a module and handle errors."""
    print("\n" + "="*70)
    print(f"Running {description}")
    print("="*70)
    
    try:
        # Run each module in its own Python process.
        # Using exec() inside a function can break imports (they land in a local scope),
        # which then causes NameError inside classes/functions defined by the executed file.
        if module_name.endswith(".py"):
            subprocess.run(
                [sys.executable, module_name],
                check=True,
                cwd=os.getcwd(),
            )
        else:
            subprocess.run(
                [sys.executable, "-m", module_name],
                check=True,
                cwd=os.getcwd(),
            )
        print(f"\n[OK] {description} completed successfully!")
        return True
    except Exception as e:
        print(f"\n[ERROR] {description}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("STUDENT PERFORMANCE FORECASTING PIPELINE")
    print("="*70)
    
    # Check if data file exists
    if not os.path.exists('StudentPerformanceFactors.csv'):
        print("\n[ERROR] StudentPerformanceFactors.csv not found!")
        print("Please ensure the dataset is in the project directory.")
        return
    
    modules = [
        ('module1_data_preprocessing.py', 'Module 1: Data Preprocessing'),
        ('module2_advanced_eda.py', 'Module 2: Advanced EDA'),
        ('module3_feature_engineering.py', 'Module 3: Feature Engineering'),
        ('module4_baseline_models.py', 'Module 4: Baseline Model Comparison'),
        ('module5_adaptive_ensemble.py', 'Module 5: Adaptive Meta-Ensemble'),
        ('module6_concept_drift.py', 'Module 6: Concept Drift Detection'),
        ('module7_explainable_ai.py', 'Module 7: Explainable AI & Counterfactuals'),
    ]
    
    results = {}
    
    for module_file, description in modules:
        if os.path.exists(module_file):
            success = run_module(module_file, description)
            results[description] = success
        else:
            print(f"\n[WARN] {module_file} not found. Skipping...")
            results[description] = False
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*70)
    
    for description, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{status}: {description}")
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"\nCompleted: {successful}/{total} modules")
    
    if successful == total:
        print("\nAll modules completed successfully!")
        print("\nNext steps:")
        print("1. Review generated CSV files and HTML visualizations")
        print("2. Run 'streamlit run streamlit_dashboard.py' to launch the dashboard")
    else:
        print("\nSome modules failed. Please review errors above.")

if __name__ == "__main__":
    main()
