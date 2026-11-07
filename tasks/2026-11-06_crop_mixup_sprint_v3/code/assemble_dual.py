"""Assemble two separately trained residual networks with fixed equal weights."""
import argparse, hashlib
from pathlib import Path
import torch
from dual_residual import DualResidual

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--policy',choices=['raw','tta'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();torch.set_num_threads(2)
    n=torch.load(a.new,map_location='cpu',weights_only=True);r=torch.load(a.reference,map_location='cpu',weights_only=True)
    assert n['training_images']==r['training_images']==50000 and n['epoch']==80 and r['epoch']==240
    assert n['seed']==r['seed']==42
    model=DualResidual(n,r,a.policy)
    cp=dict(new_checkpoint=n,reference_checkpoint=r,new_policy=a.policy,mixture_weight=.5,parameters=model.parameter_count,
        new_sha256=hashlib.sha256(a.new.read_bytes()).hexdigest(),reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest())
    a.output.parent.mkdir(parents=True,exist_ok=True);torch.save(cp,a.output)
    print('Saved fixed dual residual model; parameters:',model.parameter_count)
if __name__=='__main__':main()
