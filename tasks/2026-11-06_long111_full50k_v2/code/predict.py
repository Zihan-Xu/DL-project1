"""Reproduce the final full-50k model's raw or fixed two-view predictions."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import operators

def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

@torch.inference_mode()
def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parents[1]/'model.pt')
    ap.add_argument('--output',type=Path,default=Path('submission_reproduced.csv'))
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--compare',type=Path)
    ap.add_argument('--policy',choices=['raw','tta'],help='Optional explicit alternative to stored policy')
    ap.add_argument('--report',type=Path)
    args=ap.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    assert cp['training_images']==50000 and cp['policy'] in ('raw','tta')
    if args.policy:cp['policy']=args.policy
    model=operators.build(cp['recipe']);model.load_state_dict(cp['model_state'],strict=True)
    parameters=sum(p.numel() for p in model.parameters());assert parameters==cp['parameters']<=5000000
    model=model.to(device=args.device,memory_format=torch.channels_last).eval()
    mean,std=[torch.tensor(cp[k],device=args.device).view(1,3,1,1) for k in ['mean','std']]
    test=rows(args.data_dir/'test.csv');sample=rows(args.data_dir/'sample_submission.csv')
    ids=[r['id'] for r in test]
    assert len(test)==len(set(ids))==2000 and ids==[r['id'] for r in sample]
    predictions=[]
    for start in range(0,2000,128):
        ims=[]
        for row in test[start:start+128]:
            assert Path(row['filename']).name==row['filename']
            with Image.open(args.data_dir/'test_images'/row['filename']) as im:
                assert im.size==(32,32) and im.mode=='RGB';ims.append(np.array(im))
        x=torch.from_numpy(np.stack(ims)).to(args.device).permute(0,3,1,2).float()/255
        x=((x-mean)/std).contiguous(memory_format=torch.channels_last)
        scores=model(x)
        if cp['policy']=='tta':scores=.5*(scores.softmax(1)+model(x.flip(-1)).softmax(1))
        assert scores.dtype==torch.float32 and torch.isfinite(scores).all()
        predictions.extend(scores.argmax(1).cpu().tolist())
    matches=None
    if args.compare:
        ref=rows(args.compare)
        matches=ids==[r['id'] for r in ref] and predictions==[int(r['label']) for r in ref]
        assert matches,'Predictions differ from reference'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',newline='',encoding='utf8') as f:
        w=csv.writer(f);w.writerow(['id','label']);w.writerows(zip(ids,predictions))
    report=dict(passed=True,rows=2000,unique_ids=2000,parameters=parameters,policy=cp['policy'],device=args.device,
        original_pngs_read=2000,labels_match_reference=matches,model_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        submission_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
    if args.report:args.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
