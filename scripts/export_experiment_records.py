#!/usr/bin/env python3
"""Back up auditable experiment text records without weights or credentials.

Archive members are byte-for-byte originals, never redacted or truncated. Files
with credential indicators are listed as excluded for a separate private backup.
This is a per-file capture, not an atomic snapshot of a live filesystem.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import time


EXTENSIONS = {'.json', '.jsonl', '.yaml', '.yml', '.csv', '.tsv', '.md',
              '.txt', '.log', '.py', '.sh', '.ps1'}
EXCLUDED_DIRS = {'.git', '__pycache__', '.cache', 'cache', '.pytest_cache',
                 '.mypy_cache', '.ruff_cache', 'wandb', 'node_modules',
                 'datasets', 'dataset', 'venv', '.venv', 'feature_cache',
                 'cached_features'}
EXCLUDED_NAMES = {'.netrc', '.git-credentials', 'credentials.json',
                  'secrets.json', 'debug.log', 'debug-internal.log',
                  'debug-core.log'}
PATTERNS = {
    'private_key_material': re.compile(
        rb'-----BEGIN (?:OPENSSH |RSA |EC |DSA |ENCRYPTED )?PRIVATE KEY-----'),
    'github_token': re.compile(
        rb'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'huggingface_token': re.compile(rb'\bhf_[A-Za-z0-9]{25,}\b'),
    'wandb_token': re.compile(rb'\bwandb_v1_[A-Za-z0-9_\-]{20,}\b'),
    'embedded_url_credentials': re.compile(rb'https?://[^\s/@:]+:[^\s/@]+@'),
    'signed_network_url': re.compile(
        rb'(?:X-Goog-Signature|X-Amz-Signature|GoogleAccessId|AWSAccessKeyId)=', re.I),
    'literal_secret_assignment': re.compile(
        rb'''(?:["']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|secret[_-]?key|WANDB_API_KEY|HF_TOKEN|GITHUB_TOKEN|MODELSCOPE_TOKEN)["']?)\s*(?:\])?\s*[:=]\s*["']([A-Za-z0-9_.\-]{24,})["']''',
        re.I),
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def known_credentials(project):
    """Use existing credential references only for in-memory exclusion checks."""
    values = []
    for name in ['wandb_bridge.json', 'lab_wandb_bridge.json']:
        path = project / 'configs' / name
        if not path.is_file():
            continue
        config = json.loads(path.read_text())
        filename = config.get('api_key_file')
        if not filename:
            continue
        secret_file = Path(filename).expanduser()
        if not secret_file.is_absolute():
            secret_file = project / secret_file
        if secret_file.is_file():
            value = secret_file.read_bytes().strip()
            if len(value) >= 16 and value not in values:
                values.append(value)
    return values


def sensitive_categories(data, credentials):
    categories = [name for name, regex in PATTERNS.items() if regex.search(data)]
    if any(value in data for value in credentials):
        categories.append('known_project_credential_value')
    return categories


def capture_file(path):
    """Read twice; record retries and explicitly identify continuously changing files."""
    observations = []
    for attempt in range(1, 4):
        before = path.stat(follow_symlinks=False)
        first = path.read_bytes()
        middle = path.stat(follow_symlinks=False)
        second = path.read_bytes()
        after = path.stat(follow_symlinks=False)
        digest_first, digest_second = sha256(first), sha256(second)
        same_identity = (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino)
        metadata_stable = (before.st_size, before.st_mtime_ns) == (middle.st_size, middle.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        stable = same_identity and metadata_stable and first == second
        observations.append({'attempt': attempt, 'first_sha256': digest_first,
                             'second_sha256': digest_second, 'stable': stable})
        metadata = {'bytes': len(second), 'sha256': digest_second,
                    'source_mtime_ns': after.st_mtime_ns,
                    'captured_at_utc': timestamp(), 'capture_attempts': attempt,
                    'changed_during_capture': any(not row['stable'] for row in observations),
                    'stable_double_read': stable}
        if stable:
            if attempt > 1:
                metadata['capture_observations'] = observations
            return second, metadata
        time.sleep(0.05)
    metadata['capture_observations'] = observations
    metadata['snapshot_note'] = 'Latest read retained; source kept changing. Not an atomic or stable filesystem snapshot.'
    return second, metadata


def candidate_files(root, allowed_extensions=EXTENSIONS):
    candidates, skipped = [], []
    for directory, dirs, files in os.walk(root, followlinks=False):
        kept = []
        for name in sorted(dirs):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                skipped.append({'path': relative, 'reason': 'symlink_directory_not_followed'})
            elif name in EXCLUDED_DIRS or name.startswith('.venv'):
                skipped.append({'path': relative, 'reason': 'cache_dataset_or_sdk_directory'})
            else:
                kept.append(name)
        dirs[:] = kept
        for name in sorted(files):
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if path.suffix.lower() not in allowed_extensions:
                continue
            if path.is_symlink():
                skipped.append({'path': relative, 'reason': 'symlink_file_not_followed'})
            elif name in EXCLUDED_NAMES or name.startswith('.env'):
                skipped.append({'path': relative, 'reason': 'credential_or_sdk_debug_filename'})
            elif path.is_file():
                candidates.append(path)
    return sorted(candidates), skipped


def export(root, output, project):
    if not root.is_dir():
        raise ValueError('Source root must be an existing directory')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Refusing to overwrite an existing record archive')
    output.mkdir(parents=True, exist_ok=True)
    started = timestamp()
    source_roots = {'cloud_experiments': root,
                    'lab_remote_setup': project / 'logs' / 'lab_remote_setup'}
    candidates, exclusions = [], []
    for group, source_root in source_roots.items():
        allowed = EXTENSIONS if group == 'cloud_experiments' else {'.json', '.yaml', '.yml', '.md'}
        if not source_root.is_dir():
            exclusions.append({'source_group': group, 'path': group,
                               'reason': 'source_root_missing'})
            continue
        files, skipped = candidate_files(source_root, allowed)
        candidates.extend((group, source_root, path) for path in files)
        exclusions.extend({**row, 'source_group': group,
                           'source_relative_path': row['path'],
                           'path': group + '/' + row['path']} for row in skipped)
    secrets = known_credentials(project)
    included, archives = [], []
    # A 40 MiB raw-member budget bounds even incompressible gzip archives well
    # below the requested 45 MiB limit (each original file remains whole).
    raw_limit = 40 * 1024 ** 2
    hard_limit = 45 * 1024 ** 2
    archive = None
    archive_path = None
    archive_raw = 0

    def finish_archive():
        nonlocal archive
        if archive is None:
            return
        archive.close()
        size = archive_path.stat().st_size
        if size >= hard_limit:
            raise RuntimeError('Archive exceeds required compressed size bound')
        members_expected = {row['path']: row for row in included
                            if row['archive'] == archive_path.name}
        with tarfile.open(archive_path, 'r:gz') as restored:
            members = restored.getmembers()
            if set(member.name for member in members) != set(members_expected):
                raise RuntimeError('Archive membership verification failed')
            for member in members:
                if not member.isfile():
                    raise RuntimeError('Archive contains an unexpected non-file entry')
                data = restored.extractfile(member).read()
                expected = members_expected[member.name]
                if len(data) != expected['bytes'] or sha256(data) != expected['sha256']:
                    raise RuntimeError('Restored archive member differs from captured source')
        archives.append({'path': archive_path.name, 'bytes': size,
                         'sha256': sha256(archive_path.read_bytes()),
                         'members': len(members_expected), 'verified': True})
        archive = None

    for group, source_root, path in candidates:
        source_relative = path.relative_to(source_root).as_posix()
        relative = group + '/' + source_relative
        identity = {'path': relative, 'source_group': group,
                    'source_relative_path': source_relative}
        try:
            data, metadata = capture_file(path)
        except OSError as error:
            exclusions.append({**identity, 'reason': 'read_failed',
                               'error_type': type(error).__name__})
            continue
        categories = sensitive_categories(data, secrets)
        try:
            data.decode('utf-8')
        except UnicodeDecodeError:
            categories.append('non_utf8_text_candidate')
        if categories:
            exclusions.append({**identity, **metadata,
                               'reason': 'quarantined_do_not_publish', 'categories': categories})
            continue
        if len(data) + 2048 >= raw_limit:
            exclusions.append({**identity, **metadata,
                               'reason': 'single_file_exceeds_archive_raw_budget_not_truncated'})
            continue
        member_cost = ((len(data) + 511) // 512) * 512 + 2048
        if archive is not None and archive_raw + member_cost >= raw_limit:
            finish_archive()
        if archive is None:
            archive_path = output / f'experiment-records-{len(archives) + 1:03d}.tar.gz'
            archive = tarfile.open(archive_path, 'w:gz', compresslevel=6)
            archive_raw = 0
        info = tarfile.TarInfo(relative)
        info.size = len(data)
        info.mtime = metadata['source_mtime_ns'] // 1_000_000_000
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(data))
        included.append({**identity, **metadata, 'archive': archive_path.name})
        archive_raw += member_cost
        if len(included) % 500 == 0:
            print(json.dumps({'captured_files': len(included), 'excluded': len(exclusions)}), flush=True)
    finish_archive()
    manifest = {'format_version': 1, 'source_roots': {key: str(value) for key, value in source_roots.items()},
                'capture_started_at_utc': started, 'capture_finished_at_utc': timestamp(),
                'snapshot_semantics': 'Per-file verified capture; not a globally atomic live-filesystem snapshot.',
                'allowed_extensions': sorted(EXTENSIONS),
                'credential_scan': {'categories': list(PATTERNS) + ['known_project_credential_value'],
                                    'known_secret_values_exported': False,
                                    'matched_files_quarantined_without_source_deletion': True},
                'included_count': len(included), 'source_bytes': sum(row['bytes'] for row in included),
                'compressed_bytes': sum(row['bytes'] for row in archives),
                'changed_during_capture_count': sum(row['changed_during_capture'] for row in included),
                'unstable_last_reads': sum(not row['stable_double_read'] for row in included),
                'excluded_count': len(exclusions), 'archives': archives,
                'files': included, 'excluded': exclusions}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# 原始实验记录备份\n\n'
        f'来源：`cloud_experiments` = `{root}`；`lab_remote_setup` = `{source_roots["lab_remote_setup"]}`。'
        f'采集区间（UTC）：{started} 至 {manifest["capture_finished_at_utc"]}。\n\n'
        f'保留 {len(included)} 个原始文本文件，原始总大小 {manifest["source_bytes"]:,} bytes，'
        f'压缩后 {manifest["compressed_bytes"]:,} bytes，分为 {len(archives)} 个归档，每个小于45MiB。'
        '全部归档已逐成员恢复并核对SHA256与字节数。\n\n'
        '这是逐文件采集，不是活跃文件系统的原子快照。每个文件双读核验，发生变化时重读并在manifest中记录；'
        '持续变化者明确标为最后读取版本。没有重启训练、删除源记录、截断文件或改写日志内容。\n\n'
        f'排除项 {len(exclusions)} 个，详见manifest的excluded。任何命中凭据或签名URL的记录均只列路径和原因，'
        '原件仍留在服务器，需要另行私密备份。模型权重、数据集、大型二进制和SDK调试目录不属于本归档。\n\n'
        '学校部署记录仅选择JSON/YAML/Markdown，不笼统打包本机其他logs。'
        '各压缩包用两个恢复根前缀区分来源：`cloud_experiments/` 对应实验目录，'
        '`lab_remote_setup/` 对应项目 `logs/lab_remote_setup/`。在新目录解压全部包即可合并恢复：\n\n'
        '```bash\nmkdir -p restored_records\nfor archive in experiment-records-*.tar.gz; do\n'
        '  tar -xzf "$archive" -C restored_records\ndone\n```\n\n'
        '`manifest.json` 逐文件记录原路径、字节数、SHA256、采集时间、变化检查和所属归档。\n', encoding='utf-8')
    print(json.dumps({key: manifest[key] for key in ['included_count', 'source_bytes', 'compressed_bytes',
                                                    'changed_during_capture_count', 'unstable_last_reads',
                                                    'excluded_count']}, ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/cloud/zzc/energyWAM/experiments'))
    parser.add_argument('--output', type=Path, default=Path('docs/reproducibility/experiment_records'))
    parser.add_argument('--project', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    export(args.source.resolve(), args.output.resolve(), args.project.resolve())


if __name__ == '__main__':
    main()
