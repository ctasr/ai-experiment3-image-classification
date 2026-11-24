"""Fetch the five published best checkpoints; never overwrite a different file."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def destination(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError('Checkpoint destination escapes this checkout')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teachers-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true',
                        help='check existing files; make no network requests or writes')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'release_checkpoints.json').read_text(encoding='utf-8'))
    entries = list(manifest['teachers'].values())
    if not args.teachers_only:
        entries += list(manifest['checkpoints'].values())
    missing = []
    for entry in entries:
        path = destination(entry['path'])
        if path.exists():
            if path.stat().st_size != entry['bytes'] or digest(path) != entry['sha256']:
                raise ValueError(f"Existing checkpoint differs; not overwritten: {entry['path']}")
            print(f"Verified {entry['path']}")
            continue
        if args.verify_only:
            missing.append(entry['path'])
            continue
        asset = entry['asset']
        if '/' in asset or '\\' in asset:
            raise ValueError('Invalid release asset name')
        url = (f"https://github.com/{manifest['repository']}/releases/download/"
               f"{manifest['tag']}/{asset}")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + '.download')
        request = urllib.request.Request(url, headers={'User-Agent': 'ai-experiment3-checkpoints/1.0'})
        print(f"Downloading {asset} ({entry['bytes']} bytes)", flush=True)
        try:
            with urllib.request.urlopen(request, timeout=90) as response, partial.open('wb') as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            if partial.stat().st_size != entry['bytes'] or digest(partial) != entry['sha256']:
                raise ValueError(f'Checkpoint download failed size/SHA256 validation: {asset}')
            if path.exists():
                raise FileExistsError(f'Destination appeared during download: {path}')
            partial.rename(path)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
        print(f"Verified {entry['path']}")
    if missing:
        print('Missing checkpoints: ' + ', '.join(missing), file=sys.stderr)
        return 1
    print(f'PASS: {len(entries)} checkpoint identities verified.')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f'error: {error}')
