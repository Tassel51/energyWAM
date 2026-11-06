"""Wait for the bounded comparison and build the final report with current code."""
import json
from pathlib import Path
import time
import run_k_training_comparison as experiment

def main():
    deadline=time.monotonic()+14*3600
    while not (experiment.OUT/'complete.json').exists():
        if time.monotonic()>deadline:raise TimeoutError('comparison did not finish within 14 hours')
        time.sleep(15)
    experiment.summarize(json.loads((experiment.OUT/'status.json').read_text()))
    print((experiment.OUT/'RESULTS_CN.md').read_text(),flush=True)
    import subprocess
    subprocess.run([str(experiment.ROOT/'.venv-wandb/bin/python'),str(experiment.ROOT/'scripts/upload_k_training_comparison.py')],check=True,timeout=300)

if __name__=='__main__':main()
