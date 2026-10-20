"""Package an owner-authorized data snapshot without credentials; publish via Releases."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'dist/github_snapshot_20261020'
REPO = 'tyyao989666/fixed-income-plus-allocation'
TAG = 'research-snapshot-20261020'
PATTERNS = [re.compile(p) for p in [
    rb'\bsk-[A-Za-z0-9_-]{20,}',
    rb'\b(?:ghp_|github_pat_|gho_)[A-Za-z0-9_]{20,}',
    rb'\b[0-9a-fA-F]{32}\.[A-Za-z0-9_-]{12,}',
    rb'\beyJ[A-Za-z0-9_-]{12,}(?:\.[A-Za-z0-9_-]*){2,4}',
    rb'(?i)(?:bearer|basic)\s+[A-Za-z0-9_.-]{20,}',
    rb'(?i)[\"\']?(?:api_key|access_token|refresh_token|password|tushare_token)[\"\']?\s*[:=]\s*[\"\'][A-Za-z0-9_.-]{16,}[\"\']',
]]
TEXT = {'.json', '.jsonl', '.csv', '.txt', '.md', '.html', '.yaml', '.yml', '.log', '.xml', '.ndjson', '.ini', '.cfg'}


def hits(raw):
    return {m.group() for pattern in PATTERNS for m in pattern.finditer(raw)}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build():
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / 'snapshot_manifest.json').exists():
        raise FileExistsError('Snapshot is already frozen; reuse it rather than rebuilding')
    files = sorted(p for folder in ['data_raw', 'data_processed', 'output'] for p in (ROOT / folder).rglob('*') if p.is_file())
    known = {os.environ[n].encode() for n in ['BIGMODEL_API_KEY', 'DEEPSEEK_API_KEY', 'TUSHARE_TOKEN', 'WIND_API_KEY', 'IFIND_TOKEN'] if os.environ.get(n)}
    records, excluded, assets = [], [], []
    count, source_bytes = len(files), sum(p.stat().st_size for p in files)
    print(json.dumps(dict(phase='scan_and_package', files=count, source_bytes=source_bytes)), flush=True)
    archive, archive_path, logical_size, part = None, None, 0, 0

    def close_archive():
        if archive is not None:
            archive.close()
            with zipfile.ZipFile(archive_path) as check:
                bad = check.testzip()
                if bad:
                    raise ValueError('Archive CRC failure')
            assets.append(dict(name=archive_path.name, bytes=archive_path.stat().st_size, sha256=sha(archive_path)))

    try:
        for index, p in enumerate(files, 1):
            rel = p.relative_to(ROOT).as_posix()
            if p.name == '.env' or p.name.startswith('.env.') or p.suffix.lower() in {'.pem', '.key'} or p.name.lower() in {'credentials.json', 'token.json', 'tokens.json'}:
                excluded.append(dict(path=rel, reason='credential_file'))
                continue
            before = p.stat()
            raw = p.read_bytes()
            found = hits(raw) | {value for value in known if value in raw}
            # Compressed checkpoint/workbook archives can hide literal credentials.
            if raw.startswith(b'PK\x03\x04'):
                with zipfile.ZipFile(io.BytesIO(raw)) as nested:
                    for item in nested.infolist():
                        if item.file_size < 64 * 1024 * 1024:
                            found.update(hits(nested.read(item)))
            if p.suffix.lower() == '.parquet':
                import pyarrow.parquet as pq
                import pyarrow as pa
                table = pq.ParquetFile(p)
                string_cols = [field.name for field in table.schema_arrow if pa.types.is_string(field.type) or pa.types.is_large_string(field.type)]
                for batch in table.iter_batches(batch_size=8192, columns=string_cols) if string_cols else []:
                    found.update(hits(json.dumps(batch.to_pydict(), ensure_ascii=False).encode()))
            if b'PRIVATE KEY-----' in raw:
                found.add(b'PRIVATE KEY-----')
            sanitized = bool(found)
            if found and p.suffix.lower() not in TEXT:
                excluded.append(dict(path=rel, reason='credential_candidate_in_binary_quarantined'))
                continue
            original_hash = hashlib.sha256(raw).hexdigest()
            if found:
                for value in sorted(found, key=len, reverse=True):
                    raw = raw.replace(value, b'REDACTED_CREDENTIAL')
                if hits(raw):
                    raise ValueError('Credential pattern remains after sanitization: ' + rel)
            after = p.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Source changed while packaging: ' + rel)
            if archive is None or logical_size + len(raw) > 384 * 1024 * 1024:
                close_archive()
                part += 1
                archive_path = DEST / f'data-part-{part:03d}.zip'
                archive = zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True)
                logical_size = 0
            archive.writestr(rel, raw)
            logical_size += len(raw)
            records.append(dict(path=rel, archive=archive_path.name, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                source_sha256=original_hash, credentials_sanitized=sanitized))
            if index % 300 == 0:
                print(json.dumps(dict(phase='packaging', completed=index, total=count, part=part, excluded=len(excluded))), flush=True)
        close_archive()
        archive = None
    finally:
        if archive is not None:
            archive.close()
    manifest = dict(snapshot_date='2026-10-20', owner_redistribution_authorized=True,
                    dataset_cutoff_note='Most refreshed core market sources through 2026-10-14; inspect per-file dates.',
                    source_bytes=source_bytes, original_files_modified=False, formal_performance_certified=False,
                    contains_historical_and_quarantined_experiments=True,
                    restored_hash_bound_sources_may_require_original_local_credentials_removed_evidence=True,
                    files=records, excluded=excluded, assets=assets)
    (DEST / 'snapshot_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(phase='build_complete', files=len(records), assets=len(assets), excluded=len(excluded),
                          sanitized=sum(r['credentials_sanitized'] for r in records), compressed_bytes=sum(a['bytes'] for a in assets))), flush=True)


def github_session():
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
    result = subprocess.run(['git', 'credential', 'fill'], input='protocol=https\nhost=github.com\n\n', text=True, capture_output=True, env=env, check=True)
    credential = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
    token = credential.get('password')
    if not token:
        raise RuntimeError('GitHub credential is unavailable; authenticate with Git Credential Manager')
    session = requests.Session()
    session.headers.update({'Authorization': 'Bearer ' + token, 'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'})
    return session


def upload(commit, publish):
    session = github_session()
    api = f'https://api.github.com/repos/{REPO}'
    manifest = json.loads((DEST / 'snapshot_manifest.json').read_text(encoding='utf-8'))
    response = session.get(api + '/releases/tags/' + TAG, timeout=30)
    if response.status_code == 404:
        response = session.post(api + '/releases', json=dict(tag_name=TAG, target_commitish=commit, draft=True,
                                name='Complete research snapshot - 2026-10-20',
                                body='Code plus owner-authorized raw/processed data, evidence, models and historical reports. Download all data-part ZIPs and snapshot_manifest.json, or run scripts/download_research_data.py from the repository. API credentials removed. V95 accounting remains uncertified; abandoned and historical experiments are forensic artifacts, not validated performance.'), timeout=30)
    response.raise_for_status()
    release = response.json()
    release_id = release['id']
    response = session.get(api + f'/releases/{release_id}/assets', params={'per_page':100}, timeout=30)
    response.raise_for_status()
    existing = {a['name']:a for a in response.json()}
    for p in [*(DEST / a['name'] for a in manifest['assets']), DEST / 'snapshot_manifest.json']:
        digest = sha(p)
        if p.name in existing:
            item = existing[p.name]
            if item['size'] != p.stat().st_size or item.get('digest') != 'sha256:' + digest:
                raise ValueError('Existing uploaded asset differs; preserve it and investigate')
            print('Verified existing ' + p.name, flush=True)
            continue
        print('Uploading ' + p.name + ' bytes=' + str(p.stat().st_size), flush=True)
        with p.open('rb') as data:
            response = session.post(release['upload_url'].split('{')[0], params={'name':p.name}, data=data,
                                    headers={'Content-Type':'application/octet-stream'}, timeout=(30, 1800))
        response.raise_for_status()
        item = response.json()
        if item['size'] != p.stat().st_size or item.get('digest') != 'sha256:' + digest:
            raise ValueError('GitHub asset verification failed: ' + p.name)
        print('Uploaded and SHA-256 verified ' + p.name, flush=True)
    if publish:
        response = session.patch(api + f'/releases/{release_id}', json={'draft':False}, timeout=30)
        response.raise_for_status()
        release = response.json()
    (DEST / 'upload_receipt.json').write_text(json.dumps(dict(url=release['html_url'], draft=release['draft'],
                    release_id=release_id, source_commit=commit, uploaded_assets=len(manifest['assets'])+1), indent=2), encoding='utf-8')
    print(json.dumps(dict(phase='upload_complete', url=release['html_url'], draft=release['draft'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--upload', action='store_true')
    parser.add_argument('--publish', action='store_true')
    parser.add_argument('--commit')
    args = parser.parse_args()
    if args.build:
        build()
    if args.upload:
        if not args.commit:
            parser.error('--commit is required for upload')
        upload(args.commit, args.publish)
