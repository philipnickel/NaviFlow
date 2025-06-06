import os
import subprocess
import concurrent.futures
from pathlib import Path

# Define the base directory for results
OVERLEAF_DIR = "OVERLEAFCOLOFULFD"

# Define the commands and their target directories
commands = [
    # LDC cases
    {
        "cmd": "python postprocessing/postprocess.py --config experiments/Collocated/lidDrivenCavity/TVD/uniform/Re_100/medium/config.yaml --all",
        "target_dir": f"{OVERLEAF_DIR}/LDC/Re_100"
    },
    {
        "cmd": "python postprocessing/postprocess.py --config experiments/Collocated/lidDrivenCavity/TVD/uniform/Re_400/medium/config.yaml --all",
        "target_dir": f"{OVERLEAF_DIR}/LDC/Re_400"
    },
    {
        "cmd": "python postprocessing/postprocess.py --config experiments/Collocated/lidDrivenCavity/TVD/uniform/Re_1000/medium/config.yaml --all",
        "target_dir": f"{OVERLEAF_DIR}/LDC/Re_1000"
    },
    {
        "cmd": "python postprocessing/postprocess.py --config experiments/Collocated/lidDrivenCavity/TVD/uniform/Re_3200/medium/config.yaml --all",
        "target_dir": f"{OVERLEAF_DIR}/LDC/Re_3200"
    },
    {
        "cmd": "python postprocessing/postprocess.py --config experiments/Collocated/lidDrivenCavity/TVD/uniform/Re_5000/medium/config.yaml --all",
        "target_dir": f"{OVERLEAF_DIR}/LDC/Re_5000"
    },
    # Channel flow cases
    {
        "cmd": "python postprocessing/postprocess.py --config experiments/Collocated/channelFlow/ForReport/unstructured/medium/Re_10/config.yaml --all",
        "target_dir": f"{OVERLEAF_DIR}/channelflow/Re_10"
    },
    # Cylinder flow cases
    {
        "cmd": "python postprocessing/postprocess.py --config experiments/Collocated/cylinderFlow/ForReport/unstructured/medium/Re_5/fine/config.yaml --all",
        "target_dir": f"{OVERLEAF_DIR}/cylinderflow/Re_5"
    },
    {
        "cmd": "python postprocessing/postprocess.py --config experiments/Collocated/cylinderFlow/ForReport/unstructured/medium/Re_20/fine/config.yaml --all",
        "target_dir": f"{OVERLEAF_DIR}/cylinderflow/Re_20"
    }
]

def run_command_and_move_plots(cmd_info):
    cmd = cmd_info["cmd"]
    target_dir = cmd_info["target_dir"]
    
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        # Run the command
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True, check=True)
        
        # Move the plots directory to the target location
        # Extract the config path from the command
        config_path = cmd.split("--config ")[1].split(" --all")[0]
        base_dir = os.path.dirname(config_path)
        plots_dir = os.path.join(base_dir, "results", "plots")
        
        if os.path.exists(plots_dir):
            # Move all contents from plots to target directory
            for item in os.listdir(plots_dir):
                src = os.path.join(plots_dir, item)
                dst = os.path.join(target_dir, item)
                if os.path.exists(dst):
                    if os.path.isdir(dst):
                        subprocess.run(f"rm -rf {dst}", shell=True)
                    else:
                        os.remove(dst)
                subprocess.run(f"mv {src} {dst}", shell=True)
            print(f"Successfully moved plots to {target_dir}")
        else:
            print(f"Warning: No plots directory found at {plots_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Error details: {str(e)}")
    except Exception as e:
        print(f"Unexpected error for command {cmd}: {str(e)}")

def main():
    # Create the main OVERLEAF directory
    os.makedirs(OVERLEAF_DIR, exist_ok=True)
    
    # Run commands in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(run_command_and_move_plots, commands)

if __name__ == "__main__":
    main() 