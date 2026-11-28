"""Acquire only named frozen asset groups; different existing files are never overwritten."""
from __future__ import annotations
import argparse
import hashlib
import io
import os
import re
import json
from pathlib import Path
import shutil
import sys
import urllib.request

ROOT=Path(__file__).resolve().parent
GROUPS=('alignment-best','kl-last','weighted-last','budget-best','confidence-mass','references')
def digest(path):
    with Path(path).open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def destination(relative):
    if not isinstance(relative,str)or not relative:raise ValueError('Asset path must be nonempty')
    path=(ROOT/relative).resolve()
    if not path.is_relative_to(ROOT.resolve()):raise ValueError('Asset path escapes this checkout')
    return path
def verify(path,entry):
    if type(entry['bytes'])is not int or entry['bytes']<=0:raise ValueError('Invalid asset size')
    value=entry['sha256']
    if not isinstance(value,str)or len(value)!=64 or any(x not in '0123456789abcdef'for x in value):raise ValueError('Invalid SHA256')
    if not path.is_file()or path.stat().st_size!=entry['bytes']or digest(path)!=value:raise ValueError('Missing or differing frozen asset; no overwrite: '+str(path))
def sources(manifest):
    rows=[{'path':p,'sha256':s}for p,s in manifest['training_source_sha256'].items()]
    rows+=manifest.get('historical_source_bindings',{}).get('entries',[])
    for row in rows:
        file=destination(row['path'])
        if not file.is_file()or digest(file)!=row['sha256']:raise ValueError('Frozen source differs: '+row['path'])
    print('PASS: '+str(len(rows))+' exact source bindings')
def identity(path,entry):
    """CPU-only loading of SHA-verified PTs; no historical driver is imported."""
    if not path.name.endswith('.pt'):return
    if 'expected_training_sources'not in entry:
        raise ValueError('Explicit checkpoint identity metadata required for this optional check')
    import torch
    if torch.cuda.is_initialized():raise RuntimeError('Run checkpoint identity checks in a fresh CPU process')
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['source_sha256']!=entry['expected_training_sources']or saved['config']!=entry['expected_config']:
        raise ValueError('Checkpoint source/config role differs: '+str(path))
    if saved['epoch']!=entry['epoch']or saved['config']['seed']!=entry['seed']or saved['stats']!=entry['normalization']:
        raise ValueError('Checkpoint endpoint/seed/normalization differs')
    for name,expected in entry['expected_training_sources'].items():
        bound=entry['source_file_bindings'][name]
        if digest(destination(bound))!=expected:raise ValueError('Historical source binding differs: '+name)
    for name,expected in entry.get('expected_provenance_sources',{}).items():
        if digest(destination(entry['provenance_source_file_bindings'][name]))!=expected:raise ValueError('Analysis provenance binding differs: '+name)
    if torch.cuda.is_initialized():raise RuntimeError('CPU identity check initialized CUDA')
def response_header(response,name,required=True):
    values=response.headers.get_all(name,[])
    if len(values)>1 or (required and len(values)!=1):raise ValueError('Missing/duplicate HTTP header: '+name)
    return values[0].strip() if values else None

def resume_download(partial,url,entry):
    """Append only a fully validated suffix to the same unchanged own partial."""
    total=entry['bytes']
    if type(total)is not int or total<=0:raise ValueError('Invalid frozen asset size')
    if partial.is_symlink()or not partial.is_file():raise ValueError('Resume requires a regular own .download file')
    with partial.open('rb')as prefix:
        before=os.fstat(prefix.fileno());prefix_hash=hashlib.file_digest(prefix,'sha256')
        after=os.fstat(prefix.fileno())
    stamp=lambda st:(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
    if stamp(before)!=stamp(after)or stamp(before)!=stamp(partial.stat()):raise ValueError('Partial changed during prefix hashing')
    offset=before.st_size;prefix_sha=prefix_hash.hexdigest()
    if offset>total:raise ValueError('Partial exceeds the frozen asset size; retained unchanged')
    if offset==total:verify(partial,entry);return
    print(f'Resuming {entry["asset"]}: offset={offset}, total={total}, prefix_sha256={prefix_sha}',flush=True)
    request=urllib.request.Request(url,headers={'User-Agent':'ai-experiment3-checkpoints/1.0','Range':f'bytes={offset}-','Accept-Encoding':'identity'})
    # Buffer only this one remaining suffix; invalid/truncated responses never modify the prefix.
    with urllib.request.urlopen(request,timeout=90)as response,io.BytesIO()as suffix:
        if response.status!=206:raise ValueError('Resume requires HTTP 206; the partial is retained unchanged')
        value=response_header(response,'Content-Range')
        match=re.fullmatch(r'bytes ([0-9]+)-([0-9]+)/([0-9]+)',value)
        if not match or tuple(map(int,match.groups()))!=(offset,total-1,total):raise ValueError('HTTP Content-Range does not match the exact frozen suffix')
        length=response_header(response,'Content-Length')
        if not re.fullmatch(r'[0-9]+',length)or int(length)!=total-offset:raise ValueError('HTTP Content-Length differs from the frozen suffix')
        if response_header(response,'Content-Encoding',False)not in(None,'identity')or response_header(response,'Transfer-Encoding',False)is not None:raise ValueError('Encoded/ambiguous HTTP suffix refused')
        remaining=total-offset;combined=prefix_hash.copy()
        while remaining:
            block=response.read(min(1024*1024,remaining))
            if not block or len(block)>remaining:raise ValueError('HTTP suffix length differs; partial retained unchanged')
            suffix.write(block);combined.update(block);remaining-=len(block)
        if response.read(1):raise ValueError('HTTP suffix has trailing bytes; partial retained unchanged')
        if combined.hexdigest()!=entry['sha256']:raise ValueError('Combined prefix/suffix SHA256 differs; partial retained unchanged')
        if partial.is_symlink():raise ValueError('Partial path changed before append')
        with partial.open('r+b')as target:
            if stamp(os.fstat(target.fileno()))!=stamp(before)or hashlib.file_digest(target,'sha256').hexdigest()!=prefix_sha:raise ValueError('Partial prefix size/SHA/identity changed before append')
            target.seek(0,os.SEEK_END)
            if target.tell()!=offset:raise ValueError('Partial size changed before append')
            suffix.seek(0);shutil.copyfileobj(suffix,target);target.flush();os.fsync(target.fileno())

def acquire(entry,manifest,args,group):
    path=destination(entry['path']);asset=entry['asset']
    if not asset or '/'in asset or '\\'in asset or asset in('.','..'):raise ValueError('Invalid asset filename')
    if path.exists():verify(path,entry);return True
    if args.verify_only:return False
    if args.local_assets is not None:
        source=(args.local_assets/asset).resolve()
        if not source.is_relative_to(args.local_assets.resolve()):raise ValueError('Local asset path escapes input directory')
        verify(source,entry)
    else:
        if group.get('publication_state')!='published'or not group.get('tag'):
            raise ValueError('This asset group is not published; no remote URL is invented. Explicit --local-assets is private acquisition, not a remote-download proof.')
        source=None
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.download')
    retained=temporary.exists()
    if retained and (not args.resume or source is not None):raise FileExistsError('Retained earlier partial download; use explicit --resume for the same remote asset: '+str(temporary))
    if source is not None:
        with source.open('rb')as left,temporary.open('xb')as right:shutil.copyfileobj(left,right)
    else:
        url=f"https://github.com/{manifest['repository']}/releases/download/{group['tag']}/{asset}"
        if retained:resume_download(temporary,url,entry)
        else:
            request=urllib.request.Request(url,headers={'User-Agent':'ai-experiment3-checkpoints/1.0'})
            with urllib.request.urlopen(request,timeout=90)as left,temporary.open('xb')as right:shutil.copyfileobj(left,right)
    verify(temporary,entry)
    if path.exists():raise FileExistsError('Destination appeared during acquisition')
    temporary.rename(path);return True
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teachers-only',action='store_true')
    parser.add_argument('--references-only',action='store_true')
    parser.add_argument('--group',choices=GROUPS,action='append',default=[],help='Only explicitly requested groups; default remains the five original release assets')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Explicitly continue the same remote .download partial using a checked HTTP Range; failures retain it')
    parser.add_argument('--verify-sources',action='store_true')
    parser.add_argument('--verify-identities',action='store_true',help='CPU-check downloaded named checkpoint groups against their own original source/config/endpoint')
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--local-assets',type=Path,help='Explicit local acquisition only; never silently falls back to remote')
    args=parser.parse_args()
    if args.resume and args.local_assets is not None:parser.error('--resume is only for the same remote asset, not --local-assets')
    manifest=json.loads((ROOT/'release_checkpoints.json').read_text(encoding='utf-8'))
    if args.verify_sources:sources(manifest);return 0
    names=list(dict.fromkeys(args.group+(['references']if args.references_only else[])))
    if args.teachers_only and names:parser.error('--teachers-only cannot be combined with named groups')
    selected=[]
    if not names:
        entries=list(manifest['teachers'].values())+([]if args.teachers_only else list(manifest['checkpoints'].values()))
        selected=[('original-five',{'publication_state':'published','tag':manifest['tag'],'entries':entries})]
    for name in names:
        group=manifest['checkpoint_groups'].get(name)
        if group is None or group.get('inputs_ready')is not True or not group.get('entries'):
            raise ValueError('Actual complete frozen inputs are not ready for group: '+name)
        selected.append((name,group))
    if args.verify_identities and not names:raise ValueError('--verify-identities uses an explicit named historical/control checkpoint group')
    missing=[];verified=0
    for name,group in selected:
        for entry in group['entries']:
            if not acquire(entry,manifest,args,group):missing.append(entry['path']);continue
            if args.verify_identities:identity(destination(entry['path']),entry)
            verified+=1;print('Verified '+entry['path'])
        if name=='references'and not missing:
            info=manifest['explanation_references'];meta=info['legacy42_reconstruction'];target=destination(info['reconstructed_destination'])
            row={'bytes':meta['original_complete_npz_bytes'],'sha256':meta['original_complete_npz_sha256']}
            if target.exists():verify(target,row)
            elif args.verify_only:missing.append(info['reconstructed_destination'])
            else:
                if args.data_dir is None:raise ValueError('--data-dir is required for local exact reference reconstruction')
                from reconstruct_reference import reconstruct
                print(json.dumps(reconstruct(destination(group['entries'][0]['path']),ROOT/'release_checkpoints.json',args.data_dir,target),ensure_ascii=False))
    if missing:print('Missing assets: '+', '.join(missing),file=sys.stderr);return 1
    print(f'PASS: {verified} frozen asset identities; acquisition='+('verify_only'if args.verify_only else'local_explicit'if args.local_assets else'remote'))
    return 0
if __name__=='__main__':
    try:raise SystemExit(main())
    except(OSError,ValueError,KeyError,RuntimeError)as error:raise SystemExit('error: '+str(error))
