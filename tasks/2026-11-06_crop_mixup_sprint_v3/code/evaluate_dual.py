"""Evaluate the complete dual network on caller-provided independent labeled images."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from dual_residual import DualResidual

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parents[1]/'ensemble.pt')
    p.add_argument('--labels',type=Path,required=True,help='Independent labeled CSV: filename,label')
    p.add_argument('--images',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2)
    if a.labels.name=='train_labels.csv':raise ValueError('These 50k labels were used for training, not independent testing')
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=True);assert cp['mixture_weight']==.5
    model=DualResidual(cp['new_checkpoint'],cp['reference_checkpoint'],cp['new_policy']).eval()
    with a.labels.open(encoding='utf-8-sig',newline='') as f:rs=list(csv.DictReader(f))
    assert rs and len(rs)==len({r['filename'] for r in rs});correct=0
    for pos in range(0,len(rs),128):
        batch=rs[pos:pos+128];pixels=[]
        for r in batch:
            assert Path(r['filename']).name==r['filename'] and r['label'] in list('0123456789')
            with Image.open(a.images/r['filename']) as im:
                assert im.size==(32,32) and im.mode=='RGB';pixels.append(np.array(im))
        scores=model(torch.from_numpy(np.stack(pixels)))
        correct+=int((scores.argmax(1)==torch.tensor([int(r['label']) for r in batch])).sum())
    print(json.dumps(dict(correct=correct,n=len(rs),accuracy=correct/len(rs),parameters=model.parameter_count,independence='Caller must establish data independence; filename guards are not a provenance check')))
if __name__=='__main__':main()
