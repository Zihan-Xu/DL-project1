"""Reproduce the fixed dual residual network's predictions; no weight search."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from dual_residual import DualResidual

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parents[1]/'ensemble.pt')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');p.add_argument('--compare',type=Path);p.add_argument('--report',type=Path)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    assert cp['mixture_weight']==.5 and cp['parameters']==4872692
    model=DualResidual(cp['new_checkpoint'],cp['reference_checkpoint'],cp['new_policy']).to(a.device,memory_format=torch.channels_last).eval()
    def rows(path):
        with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
    test=rows(a.data_dir/'test.csv');sample=rows(a.data_dir/'sample_submission.csv');ids=[r['id'] for r in test]
    assert len(ids)==len(set(ids))==2000 and ids==[r['id'] for r in sample]
    labels=[]
    with torch.inference_mode():
        for pos in range(0,2000,128):
            pixels=[]
            for r in test[pos:pos+128]:
                assert Path(r['filename']).name==r['filename']
                with Image.open(a.data_dir/'test_images'/r['filename']) as im:
                    assert im.mode=='RGB' and im.size==(32,32);pixels.append(np.array(im))
            scores=model(torch.as_tensor(np.stack(pixels),device=a.device));assert scores.shape[1]==10 and torch.isfinite(scores).all()
            labels+=scores.argmax(1).cpu().tolist()
    if a.compare:
        reference=rows(a.compare);assert ids==[r['id'] for r in reference] and labels==[int(r['label']) for r in reference]
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x',encoding='utf8',newline='') as f:w=csv.writer(f);w.writerow(['id','label']);w.writerows(zip(ids,labels))
    result=dict(passed=True,rows=2000,parameters=model.parameter_count,device=a.device,labels_match_reference=bool(a.compare),model_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),csv_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())
    if a.report:a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
