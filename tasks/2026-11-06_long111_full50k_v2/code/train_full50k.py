"""Reproduce the fixed full-50k fit using the executed training loop.

Run on a free CUDA GPU. Outputs must be outside the immutable release.
--check-only verifies all 50k original PNGs, normalization and epoch sampling on CPU.
"""
import argparse,hashlib,io,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import engine,local_runtime,operators

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'evidence/config.json')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args();torch.set_num_threads(4)
    release=Path(__file__).resolve().parents[1];out=args.output_dir.resolve()
    assert not out.is_relative_to(release),'Keep outputs outside the immutable release'
    out.mkdir(parents=True,exist_ok=True)
    config=engine.read(args.config);assert config['seed']==42 and config['training_images']==50000
    assert config['epochs']==config['recipe']['epochs']==240
    source=engine.rows(args.data_dir/'train_labels.csv')
    assert len(source)==len({r['id'] for r in source})==50000
    assert engine.sha(args.data_dir/'train_labels.csv')==config['source_hashes']['data\\competition\\train_labels.csv']
    labels=np.array([int(r['label']) for r in source],dtype=np.int64)
    assert np.array_equal(np.bincount(labels),np.full(10,5000))
    pixels=np.empty((50000,32,32,3),dtype=np.uint8)
    for i,r in enumerate(source):
        assert Path(r['filename']).name==r['filename']
        with Image.open(args.data_dir/'train_images'/r['filename']) as im:
            assert im.size==(32,32) and im.mode=='RGB';pixels[i]=np.asarray(im)
    b=io.BytesIO();np.save(b,pixels)
    assert hashlib.sha256(b.getbuffer()).hexdigest()==config['source_hashes']['.cache\\baseline\\train.npy']
    sums=np.zeros(3);sq=np.zeros(3)
    for part in np.array_split(pixels,100):
        x=part.astype(np.float64)/255;sums+=x.sum((0,1,2));sq+=(x*x).sum((0,1,2))
    mean=sums/(50000*32*32);std=np.sqrt(sq/(50000*32*32)-mean**2)
    assert np.array_equal(mean,np.array(config['mean'])) and np.array_equal(std,np.array(config['std']))
    for epoch in range(1,241):
        order,counts,_=operators.sampler(labels,42,epoch,128)
        assert len(order)==50000 and len(counts)==391 and counts.min()>0
    model=operators.build(config['recipe'])
    assert sum(p.numel() for p in model.parameters())==config['parameters']
    report=dict(passed=True,training_images=50000,epochs_checked=240,parameters=config['parameters'],normalization_matches=True)
    local_runtime.atomic_json(out/'preflight.json',report)
    if args.check_only:print(json.dumps(report));return
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA/BF16 is required for the original training loop'
    if (out/'config.json').exists():assert (out/'config.json').read_bytes()==args.config.read_bytes()
    else:(out/'config.json').write_bytes(args.config.read_bytes())
    engine.HERE=out;engine.DATA=args.data_dir.resolve();local_runtime.OUTPUT=out
    engine.accepted=lambda:True # Fixed, already accepted recipe; never rerank experiments here.
    engine.prepare=lambda:(config,pixels,labels)
    engine.main()

if __name__=='__main__':main()
