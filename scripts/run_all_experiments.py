"""Master experiment runner: executes Development, Test, 3 Ablations, and Baseline."""
import os
import subprocess
import sys
import time

def run_cmd(cmd_list, description):
    print(f"\n========================================================")
    print(f"[RUNNING] {description}")
    print(f"Command: {' '.join(cmd_list)}")
    print(f"========================================================")
    t0 = time.time()
    res = subprocess.run([sys.executable] + cmd_list, capture_output=True, text=True, encoding="utf-8")
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[FAILED] {description} in {dt:.1f}s")
        print(res.stderr)
    else:
        print(f"[COMPLETED] {description} in {dt:.1f}s")
        print(res.stdout.strip())
    return res.returncode == 0

def main():
    os.makedirs("outputs", exist_ok=True)
    
    # 1. Pipeline on Development (42 images)
    run_cmd([
        "run_pipeline.py",
        "--manifest", "data/manifest_development_full.csv",
        "--images-dir", "data/development",
        "--output-csv", "outputs/results_development.csv",
        "--intermediate-dir", "outputs/intermediates_dev",
    ], "Experiment 1: Full Pipeline on Development (42 images)")
    
    # 2. Pipeline on Test (30 images)
    run_cmd([
        "run_pipeline.py",
        "--manifest", "data/manifest_test_full.csv",
        "--images-dir", "data/test",
        "--output-csv", "outputs/results_test.csv",
        "--intermediate-dir", "outputs/intermediates_test",
    ], "Experiment 2: Full Pipeline on Test (30 images)")
    
    # 3. Ablation 1: No Gaussian smoothing
    run_cmd([
        "run_pipeline.py",
        "--manifest", "data/manifest_development_full.csv",
        "--images-dir", "data/development",
        "--output-csv", "outputs/ablation_no_gaussian.csv",
        "--no-gaussian",
    ], "Experiment 3: Ablation 1 - No Gaussian Smoothing")
    
    # 4. Ablation 2: No Hough duplicate suppression
    run_cmd([
        "run_pipeline.py",
        "--manifest", "data/manifest_development_full.csv",
        "--images-dir", "data/development",
        "--output-csv", "outputs/ablation_no_hough_suppression.csv",
        "--no-hough-suppression",
    ], "Experiment 4: Ablation 2 - No Hough Line Suppression")
    
    # 5. Ablation 3: No Edge Support in scoring
    run_cmd([
        "run_pipeline.py",
        "--manifest", "data/manifest_development_full.csv",
        "--images-dir", "data/development",
        "--output-csv", "outputs/ablation_no_edge_support.csv",
        "--no-edge-support",
    ], "Experiment 5: Ablation 3 - No Edge Support (Area Only)")
    
    # 6. Baseline on Development
    run_cmd([
        "scripts/run_baseline_batch.py",
        "data/manifest_development_full.csv",
        "data/development",
        "outputs/results_baseline_dev.csv",
    ], "Experiment 6: OpenCV Baseline on Development")
    
    # 7. Baseline on Test
    run_cmd([
        "scripts/run_baseline_batch.py",
        "data/manifest_test_full.csv",
        "data/test",
        "outputs/results_baseline_test.csv",
    ], "Experiment 7: OpenCV Baseline on Test")
    
    print("\nAll experiments finished successfully!")

if __name__ == "__main__":
    main()
