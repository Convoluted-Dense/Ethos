import time
import subprocess
import sys
import os

def get_diffusion_process():
    try:
        # Use PowerShell to check for running processes containing generate_diffused_3.py
        cmd = 'powershell -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like \'*generate_diffused_3.py*\' } | Select-Object -ExpandProperty CommandLine"'
        output = subprocess.check_output(cmd, shell=True, text=True, errors='ignore')
        lines = [line.strip() for line in output.split('\n') if 'generate_diffused_3.py' in line and 'wait_and_run_train_v2.py' not in line]
        if len(lines) > 0:
            return lines[0]
        return None
    except Exception as e:
        return None

print("Monitoring system for 'generate_diffused_3.py'...")
time.sleep(5) # buffer

while True:
    cmd = get_diffusion_process()
    if cmd is None:
        break
    print("Diffusion is still running. Waiting 30 seconds...", flush=True)
    time.sleep(30)

print("\nDiffusion has finished! Launching train_steering_v2.py...")

# Launch the training script
cmd_to_run = [".venv\\Scripts\\python.exe", "train_steering_v2.py"]

# Run the training script in-place
subprocess.run(cmd_to_run)

print("Training completed. Cleaning up watcher script...")
try:
    os.remove(__file__)
except:
    pass
