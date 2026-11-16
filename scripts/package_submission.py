"""Archive the clean HEAD commit and verify every ZIP entry against its Git blob."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def package(output: Path):
    if git('status', '--porcelain').strip():
        raise RuntimeError('Commit all changes before packaging; the working tree must be clean')
    commit = git('rev-parse', 'HEAD').decode().strip()
    output = output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError('Write the archive outside the repository')
    output.mkdir(parents=True, exist_ok=True)
    archive = output / f'submission-{commit[:12]}.zip'
    prefix = 'constrained-rf/'
    git('archive', '--format=zip', f'--prefix={prefix}', '-o', str(archive), commit)
    expected = {}
    for entry in git('ls-tree', '-r', '-z', commit).split(b'\0'):
        if not entry: continue
        info, name = entry.split(b'\t', 1)
        mode, kind, oid = info.decode().split()
        if kind != 'blob' or mode not in {'100644', '100755'}:
            raise AssertionError('Only regular tracked files are permitted in this submission')
        expected[name.decode('utf-8')] = oid
    hashes = {}
    with zipfile.ZipFile(archive) as z:
        entries = [entry for entry in z.infolist() if not entry.is_dir()]
        if len(entries) != len(expected): raise AssertionError('ZIP file count differs from commit')
        for entry in entries:
            if not entry.filename.startswith(prefix): raise AssertionError('Unexpected archive prefix')
            name = entry.filename[len(prefix):]
            data = z.read(entry)
            blob = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
            if expected.get(name) != blob: raise AssertionError(f'ZIP bytes differ from commit: {name}')
            hashes[name] = hashlib.sha256(data).hexdigest()
        if z.comment.decode() != commit: raise AssertionError('ZIP commit comment differs')
    result = {'commit': commit, 'archive': archive.name, 'files': len(hashes),
              'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
              'entry_sha256': hashes}
    inventory = archive.with_suffix('.inventory.json')
    inventory.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8', newline='\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'entry_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT.parent/'submission')
    package(parser.parse_args().output_dir)
