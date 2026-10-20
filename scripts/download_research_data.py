"""Restore the owner-authorized data Release with archive and per-file hashes."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://github.com/tyyao989666/fixed-income-plus-allocation/releases/download/research-snapshot-20261020/'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fetch(name, cache):
    target = cache / name
    if not target.exists():
        print('Downloading ' + name, flush=True)
        temporary = target.with_suffix(target.suffix + '.part')
        with urllib.request.urlopen(BASE + name, timeout=120) as response, temporary.open('wb') as output:
            shutil.copyfileobj(response, output, 4 * 1024 * 1024)
        temporary.replace(target)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=Path, default=ROOT / '.download-cache')
    parser.add_argument('--destination', type=Path, default=ROOT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    destination = args.destination.resolve()
    manifest = json.loads(fetch('snapshot_manifest.json', args.cache).read_text(encoding='utf-8'))
    expected = {row['path']: row for row in manifest['files']}
    checked = set()
    for asset in manifest['assets']:
        path = fetch(asset['name'], args.cache)
        if path.stat().st_size != asset['bytes'] or digest(path) != asset['sha256']:
            raise ValueError('Archive hash mismatch: ' + asset['name'])
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                rel = PurePosixPath(member.filename)
                target = (destination / member.filename).resolve()
                if rel.is_absolute() or '..' in rel.parts or '\\' in member.filename or ':' in member.filename:
                    raise ValueError('Unsafe archive path')
                if destination not in target.parents or rel.parts[0] not in {'data_raw', 'data_processed', 'output'}:
                    raise ValueError('Archive path outside data directories')
                row = expected.get(member.filename)
                if row is None or member.filename in checked or row['archive'] != asset['name']:
                    raise ValueError('Unregistered or duplicate archive entry')
                if not args.verify_only:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if digest(target) != row['sha256']:
                            raise FileExistsError('Local file differs; restore in a fresh checkout: ' + member.filename)
                        checked.add(member.filename)
                        continue
                h, size = hashlib.sha256(), 0
                temporary = target.with_name(target.name + '.snapshot-part')
                out = temporary.open('wb') if not args.verify_only else None
                try:
                    with archive.open(member) as stream:
                        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                            h.update(block)
                            size += len(block)
                            if out is not None:
                                out.write(block)
                finally:
                    if out is not None:
                        out.close()
                if size != row['bytes'] or h.hexdigest() != row['sha256']:
                    raise ValueError('Restored file hash mismatch: ' + member.filename)
                if not args.verify_only:
                    temporary.replace(target)
                checked.add(member.filename)
        print('Verified ' + asset['name'], flush=True)
    if checked != set(expected):
        raise ValueError('Incomplete snapshot')
    print(json.dumps(dict(files_verified=len(checked), restored=not args.verify_only, formal_performance_certified=False)), flush=True)


if __name__ == '__main__':
    main()
