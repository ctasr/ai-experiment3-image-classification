"""Reconstruct the exact legacy42 cache from pixel-free arrays and CIFAR train.

No official test file is opened. Success requires the complete original NPZ SHA.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import pickle
import zipfile

import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def zipinfo(row):
    info=zipfile.ZipInfo(row['filename'],tuple(row['date_time']))
    for name in ('compress_type','create_system','create_version','extract_version','reserved','flag_bits','volume','internal_attr','external_attr'):
        setattr(info,name,row[name])
    info.comment=base64.b64decode(row['comment_base64']);info.extra=base64.b64decode(row['extra_base64']);info._compresslevel=row['compresslevel']
    return info

def reconstruct(payload,manifest_file,data_dir,destination):
    document=json.loads(Path(manifest_file).read_text(encoding='utf-8'))
    manifest=document.get('explanation_references',{}).get('legacy42_reconstruction',document)
    payload,data_dir,destination=Path(payload).resolve(),Path(data_dir).resolve(),Path(destination).resolve()
    if destination.exists():raise FileExistsError('Never overwrite an existing original/reference file')
    if destination.is_relative_to(data_dir):raise ValueError('Reconstructed cache must be outside immutable raw data')
    if sha(payload)!=manifest['pixel_free_payload_sha256']:raise RuntimeError('Pixel-free array payload changed')
    with np.load(payload,allow_pickle=False)as stored:indices=stored['indices'].copy()
    if indices.dtype!=np.int64 or indices.shape!=(200,)or np.any(indices<0)or np.any(indices>=50000):raise ValueError('Unexpected original fixed200 IDs')
    images=np.empty((200,32,32,3),dtype=np.uint8)
    original_data_locks={}
    for row in manifest['official_training_batches']:
        relative=Path(row['file']);relative=Path(*relative.parts[1:])if relative.parts[0]=='data'else relative
        source=(data_dir/relative).resolve()
        if not source.is_relative_to(data_dir)or source.name.startswith('test'):raise ValueError('Only frozen original training batches are allowed')
        original_data_locks[str(source)]=sha(source)
        if original_data_locks[str(source)]!=row['sha256']:raise RuntimeError('Original training file hash differs')
        number=int(source.name.rsplit('_',1)[1]);chosen=np.flatnonzero(indices//10000==number-1)
        if not len(chosen):continue
        with source.open('rb')as stream:batch=pickle.load(stream,encoding='bytes')
        data=np.asarray(batch[b'data'])
        if data.shape!=(10000,3072)or data.dtype!=np.uint8:raise ValueError('Unexpected official train batch schema')
        images[chosen]=data[indices[chosen]%10000].reshape(-1,3,32,32).transpose(0,2,3,1)
    # This header is format/dtype/shape metadata only, never image values.
    image_payload=base64.b64decode(manifest['images_npy_header_base64'])+np.ascontiguousarray(images).tobytes(order='C')
    if hashlib.sha256(image_payload).hexdigest()!=manifest['images_npy_sha256']:
        raise RuntimeError('Images reconstructed from normal CIFAR files do not match the original member')
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(destination.suffix+'.reconstruction_partial')
    if temporary.exists():raise FileExistsError('Preserve an earlier failed reconstruction for review')
    with zipfile.ZipFile(payload,'r')as stored,zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True,compresslevel=None)as output:
        output.comment=base64.b64decode(manifest['archive_comment_base64'])
        for row in manifest['original_zip_members']:
            member=image_payload if row['filename']=='images.npy'else stored.read(row['filename'])
            if len(member)!=row['file_size']or hashlib.sha256(member).hexdigest()!=row['npy_member_sha256']:raise RuntimeError('A complete original member differs')
            with output.open(zipinfo(row),mode='w',force_zip64=True)as stream:stream.write(member)
    actual=sha(temporary)
    if actual!=manifest['original_complete_npz_sha256']or temporary.stat().st_size!=manifest['original_complete_npz_bytes']:
        raise RuntimeError('Exact complete original NPZ reconstruction failed; partial retained, never pretend the old SHA')
    temporary.rename(destination)
    if any(sha(path)!=digest for path,digest in original_data_locks.items()):raise RuntimeError('An original training input changed')
    return {'state':'exact_original_legacy42_NPZ_reconstructed_from_pixel_free_payload_and_local_CIFAR_train',
            'destination':str(destination),'complete_original_sha256':actual,'bytes':destination.stat().st_size,
            'original_train_file_hashes':original_data_locks,'original_raw_data_unchanged':True,'official_test_opened':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--pixel-free',type=Path,required=True);parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(reconstruct(args.pixel_free,args.manifest,args.data_dir,args.output),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
