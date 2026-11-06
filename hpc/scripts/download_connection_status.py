"""Read download status on HPC, repairing forwarding across login nodes."""
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess

root = Path.home() / 'energywam_storage'
data = json.loads((root / 'logs/full_robocasa365_status.json').read_text())
host_file = root / 'downloads/full_robocasa365_host.json'
target = json.loads(host_file.read_text())['hostname'] if host_file.exists() else socket.gethostname()
control = '/tmp/energywam-box-bridge-' + str(os.getuid())
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
       '-o', 'ConnectTimeout=5']
if target != socket.gethostname():
    check = subprocess.run(ssh + ['-S', control, '-O', 'check', target],
                           capture_output=True, timeout=7)
    if check.returncode:
        bridge = subprocess.run(ssh + [
            '-fNT', '-o', 'ExitOnForwardFailure=yes',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
            '-o', 'ControlMaster=yes', '-o', 'ControlPath=' + control,
            '-R', '127.0.0.1:38741:127.0.0.1:38741', target],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=10)
        if bridge.returncode:
            raise RuntimeError('Cross-login-node download bridge could not be restored')
    code = 'from pathlib import Path; print(Path("/proc/' + str(int(data['pipeline_pid'])) + '").exists())'
    result = subprocess.run(ssh + [target, 'python3 -c ' + shlex.quote(code)],
                            capture_output=True, text=True, timeout=8)
    if result.returncode or result.stdout.strip() not in ('True', 'False'):
        raise RuntimeError('Could not verify pipeline on its original login node')
    data['pipeline_alive'] = result.stdout.strip() == 'True'
else:
    data['pipeline_alive'] = Path('/proc/' + str(int(data['pipeline_pid']))).exists()
data['pipeline_host'] = target
data['connection_host'] = socket.gethostname()
data['robosuite_assets_ready'] = (root / 'logs/ROBOSUITE_ASSETS_READY.json').exists()
if data.get('stage') == 'complete' and not data['robosuite_assets_ready']:
    data['stage'] = 'waiting_for_robosuite_assets'
print(json.dumps(data))
