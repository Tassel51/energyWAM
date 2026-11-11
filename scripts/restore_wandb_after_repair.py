"""Restore only the project cloud uploader after an isolated repair exits."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    receipt = Path(args.receipt)
    while True:
        record = json.loads(receipt.read_text())
        proc = Path('/proc') / str(int(record['pid']))
        try:
            status = (proc / 'status').read_text()
            command = (proc / 'cmdline').read_bytes()
            state = next(line.split()[1] for line in status.splitlines() if line.startswith('State:'))
            running = state != 'Z' and b'repair_wandb_source.py' in command
        except FileNotFoundError:
            running = False
        if not running:
            break
        time.sleep(15)
    result = subprocess.run(['supervisorctl', 'start', 'energywam-wandb'],
                            capture_output=True, text=True, timeout=30)
    report = {'checked_unix': time.time(), 'repair_phase': record.get('phase'),
              'service': 'energywam-wandb', 'returncode': result.returncode,
              'output': result.stdout.strip(), 'training_changed': False}
    receipt.with_name(receipt.stem + '_uploader_restored.json').write_text(
        json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
