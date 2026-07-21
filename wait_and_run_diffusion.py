import time
import subprocess
import sys
import os

def get_collect_process():
    try:
        output = subprocess.check_output('wmic process get CommandLine', shell=True, text=True, errors='ignore')
        lines = [line for line in output.split('\n') if 'beamng_collect.py' in line and 'wait_and_run' not in line]
        if len(lines) > 0:
            return lines[0]
        return None
    except Exception as e:
        return None

print("Monitoring system for 'beamng_collect.py'...")
time.sleep(5) # buffer

# Try to detect which folder is being used based on --out parameter
dataset_arg = None
running_cmd = get_collect_process()
if running_cmd:
    print(f"Detected running collection: {running_cmd.strip()}")
    # parse --out argument
    parts = running_cmd.strip().split()
    if '--out' in parts:
        try:
            idx = parts.index('--out')
            dataset_arg = parts[idx+1]
            print(f"Detected custom dataset folder: {dataset_arg}")
        except:
            pass

while True:
    cmd = get_collect_process()
    if cmd is None:
        break
    print("Collection is still running. Waiting 30 seconds...", flush=True)
    time.sleep(30)

print("\nData collection has finished! Launching generate_diffused_3.py...")
print("The script will now display the progress window.")

# Launch the generation script
cmd_to_run = [".venv\\Scripts\\python.exe", "generate_diffused_3.py"]
if dataset_arg:
    cmd_to_run.append(dataset_arg)

subprocess.run(cmd_to_run)

print("Diffusion completed. Cleaning up watcher script...")
try:
    os.remove(__file__)
except:
    pass
