"""Evaluate supplied independent labeled images; full50k has no remaining local holdout."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import operators

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--labels',type=Path,required=True,help='CSV with filename,label for an independent labeled set')
    p.add_argument('--images',type=Path,required=True)
    p.add_argument('--policy',choices=['raw','tta'],default='tta')
    args=p.parse_args();torch.set_num_threads(4)
    if args.labels.name=='train_labels.csv':raise ValueError('These 50k labels were used for training; they are not independent test data')
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model=operators.build(cp['recipe']).eval();model.load_state_dict(cp['model_state'],strict=True)
    mean,std=[torch.tensor(cp[k]).view(1,3,1,1) for k in ['mean','std']]
    with args.labels.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    assert rows and len({r['filename'] for r in rows})==len(rows)
    correct=0
    for start in range(0,len(rows),128):
        batch=rows[start:start+128];ims=[]
        for r in batch:
            assert Path(r['filename']).name==r['filename'] and 0<=int(r['label'])<=9
            with Image.open(args.images/r['filename']) as im:
                assert im.size==(32,32) and im.mode=='RGB';ims.append(np.array(im))
        x=(torch.from_numpy(np.stack(ims)).permute(0,3,1,2).float()/255-mean)/std
        scores=model(x)
        if args.policy=='tta':scores=.5*(scores.softmax(1)+model(x.flip(-1)).softmax(1))
        correct+=int((scores.argmax(1)==torch.tensor([int(r['label']) for r in batch])).sum())
    print(json.dumps(dict(correct=correct,n=len(rows),accuracy=correct/len(rows),parameters=sum(p.numel() for p in model.parameters()),independence='Caller must establish data independence; filename guards are not a provenance check')))

if __name__=='__main__':main()
