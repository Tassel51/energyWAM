"""Read-only terminal status for the bounded head-LR study."""
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('/cloud/zzc/energyWAM/experiments/head_lr_tiny16_20261106')


def main():
    controller = OUT/'controller_status.json'
    if controller.exists():
        print(json.loads(controller.read_text()))
    for job in json.loads((OUT/'arms.json').read_text()):
        cfg = yaml.safe_load((ROOT/job['config']).read_text())
        folder = Path(cfg['trainer']['output_dir'])
        speed = folder/'speed_report.json'
        try:
            done = json.loads(speed.read_text())['completed_steps'] if speed.exists() else 0
        except (ValueError, OSError):
            done = 0
        bar = '#' * min(30, done*30//200)
        print(f"head_lr={cfg['trainer']['energy_head_learning_rate']:g} [{bar:<30}] {done}/200")
        for steps in [50,200]:
            status = OUT/f'status_{steps}.json'
            if status.exists():
                state = json.loads(status.read_text()).get(job['name'])
                if state:
                    print(' ',steps,state['phase'],state.get('reason',''))
        selection = folder/'best_selection.json'
        if selection.exists():
            selected = json.loads(selection.read_text())
            print('  best VAL:',selected['global_step'],selected['metrics'])


if __name__=='__main__':
    main()
