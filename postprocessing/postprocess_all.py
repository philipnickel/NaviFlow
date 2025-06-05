import os
import subprocess
from multiprocessing import Pool, cpu_count

EXPERIMENTS_ROOT = "experiments"

def find_config_files(root):
    config_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "config.yaml" in filenames:
            config_path = os.path.join(dirpath, "config.yaml")
            config_files.append(config_path)
    return config_files

def postprocess_config(config_path):
    print(f"Starting postprocessing: {config_path}")
    try:
        subprocess.run([
            "python",
            "postprocessing/postprocess.py",
            "--config", config_path,
            "--all"
        ], check=True)
        print(f"✅ Completed: {config_path}")
        return f"SUCCESS: {config_path}"
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed: {config_path} - {e}")
        return f"FAILED: {config_path} - {e}"
    except Exception as e:
        print(f"❌ Error: {config_path} - {e}")
        return f"ERROR: {config_path} - {e}"

if __name__ == "__main__":
    all_configs = find_config_files(EXPERIMENTS_ROOT)
    print(f"Found {len(all_configs)} config files to process")
    
    # Use all available CPU cores, but cap at 8 to avoid overwhelming the system
    num_processes = min(cpu_count(), 8, len(all_configs))
    print(f"Using {num_processes} parallel processes")
    
    with Pool(processes=num_processes) as pool:
        results = pool.map(postprocess_config, all_configs)
    
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    for result in results:
        print(result)