"""Back up ignored historical source versions without datasets or checkpoints."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = Path('/cloud/zzc/energyWAM/experiments')
DEST = ROOT / 'docs/reproducibility/source_archives'
SOURCE_SUFFIXES = {'.py', '.sh', '.ps1'}


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    records = []

    def copy(source, relative):
        target = DEST / relative
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f'Archive collision: {relative}')
        else:
            shutil.copy2(source, target)
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        records.append({'original_path': str(source), 'archive_path': str(relative),
                        'bytes': len(payload), 'sha256': digest})

    for source in sorted((ROOT / 'logs/frozen_sources').rglob('*')):
        if source.is_file() and '__pycache__' not in source.parts:
            if source.is_symlink():
                raise RuntimeError('Review frozen-source symlink before exporting')
            copy(source, Path('frozen_sources') / source.relative_to(ROOT / 'logs/frozen_sources'))
    for source in sorted(EXPERIMENTS.rglob('*.zip')):
        if 'source' not in source.name and 'diagnostic' not in source.name:
            continue
        with zipfile.ZipFile(source) as archive:
            for member in archive.infolist():
                path = Path(member.filename)
                if path.is_absolute() or '..' in path.parts:
                    raise RuntimeError(f'Unsafe member in {source}')
                if not member.is_dir() and path.suffix not in {'.py', '.yaml', '.json'}:
                    raise RuntimeError(f'Unexpected non-source archive member: {path}')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        copy(source, Path('bundles') / f'{digest}.zip')
    for base, prefix in [(ROOT / 'logs', 'local_auxiliary'), (EXPERIMENTS, 'experiment_auxiliary')]:
        for source in sorted(base.rglob('*')):
            if (source.is_file() and source.suffix in SOURCE_SUFFIXES
                    and 'frozen_sources' not in source.parts and '__pycache__' not in source.parts):
                copy(source, Path(prefix) / source.relative_to(base))
    manifest = {'purpose': 'Preserve all discovered historical project source versions before server retirement',
                'excludes': ['datasets', 'model weights', 'checkpoints', 'credentials', 'virtual environments'],
                'file_references': len(records),
                'unique_archive_files': len({r['archive_path'] for r in records}),
                'entries': records}
    (DEST / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value for key, value in manifest.items() if key != 'entries'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
