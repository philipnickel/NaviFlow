import os
import subprocess

EXPERIMENTS_ROOT = "experiments"

def find_experiment_dirs(root):
    experiment_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if (
            "config.yaml" in filenames and
            "results" in dirnames and
            not dirpath.endswith("debugging")
        ):
            experiment_dirs.append(dirpath)
    return experiment_dirs

def postprocess_experiment(exp_dir):
    print(f"Postprocessing: {exp_dir}")
    rel_exp_dir = os.path.relpath(exp_dir, EXPERIMENTS_ROOT)
    subprocess.run([
        "python",
        "naviflow_collocated/utils/postprocess/postprocess.py",
        "--experiment", rel_exp_dir,
        "--all"
    ], check=True)

if __name__ == "__main__":
    all_experiments = find_experiment_dirs(EXPERIMENTS_ROOT)
    for exp in all_experiments:
        postprocess_experiment(exp)