"""Official torchvision0.24 operators, independently sampled for each image.

Only policy sampling is new: a private stateless PCG64 keyed by run/epoch/batch.
Every row is an image-in-batch coordinate, not one policy shared by the batch.
"""
from pathlib import Path
import hashlib
import numpy as np
import torch
import torchvision
from torchvision.transforms import RandAugment, InterpolationMode
from torchvision.transforms.autoaugment import _apply_op
import torchvision.transforms.autoaugment as official
import torchvision.transforms.functional as functional
import torchvision.transforms._functional_tensor as functional_tensor

NUM_OPS=2
MAGNITUDE=9
BINS=31
FILL=(125.0,123.0,114.0)
INTERPOLATION=InterpolationMode.BILINEAR
SEED_TAG=58193
SPACE=RandAugment(NUM_OPS,MAGNITUDE,BINS,INTERPOLATION,list(FILL))._augmentation_space(BINS,(32,32))
NAMES=tuple(SPACE)
assert NAMES==('Identity','ShearX','ShearY','TranslateX','TranslateY','Rotate','Brightness','Color','Contrast','Sharpness','Posterize','Solarize','AutoContrast','Equalize')


def library_manifest():
    return {'torchvision':torchvision.__version__, 'torch':torch.__version__,
            'sha256':{Path(m.__file__).name:hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                      for m in (official,functional,functional_tensor)}}


def policy(seed,epoch,batch,size):
    assert 0<size<=256 and epoch>=0 and batch>=0
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([int(seed),int(epoch),int(batch),SEED_TAG])))
    # Fixed256 rows make the meaning of image-in-batch index independent of a short last batch.
    operations=rng.integers(0,len(NAMES),size=(256,NUM_OPS),dtype=np.int64)[:size].copy()
    signs=(rng.integers(0,2,size=(256,NUM_OPS),dtype=np.int64)*2-1)[:size].copy()
    for i,name in enumerate(NAMES):
        if not SPACE[name][1]: signs[operations==i]=1
    return operations,signs


def magnitude(op_index,sign):
    values,signed=SPACE[NAMES[int(op_index)]]
    value=float(values[MAGNITUDE].item()) if values.ndim else 0.0
    return value*int(sign) if signed else value


def reference(images,operations,signs):
    if images.is_cuda:return reference(images.cpu(),operations,signs).to(images.device)
    assert images.dtype==torch.uint8 and tuple(images.shape[1:])==(3,32,32)
    output=[]
    for i,image in enumerate(images):
        image=image.clone()
        for slot in range(NUM_OPS):
            op=int(operations[i,slot]); image=apply_op(image,NAMES[op],magnitude(op,signs[i,slot]),INTERPOLATION,list(FILL))
        output.append(image)
    return torch.stack(output).contiguous(memory_format=torch.channels_last)


def grouped(images,operations,signs):
    # Canonical per-image official CPU uint8 operator, deliberately no batch
    # affine approximation: float affine grids can round differently by shape.
    return reference(images,operations,signs)


def digest(operations,signs):
    return hashlib.sha256(operations.astype('<i8').tobytes()+signs.astype('<i8').tobytes()).hexdigest()


def apply_op(images, name, value, interpolation, fill):
    # torchvision Equalize uses nondeterministic floating CUDA histc. Preserve
    # its exact uint8 operator on CPU only; all other GPU operations stay intact.
    if name == "Equalize" and images.is_cuda:
        return _apply_op(images.cpu(), name, value, interpolation, fill).to(images.device)
    return _apply_op(images, name, value, interpolation, fill)
