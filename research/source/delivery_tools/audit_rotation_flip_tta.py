import csv,json,sys,math,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'experiments/research_sweep/architecture'))
from models import ResidualNet
out=ROOT/'delivery_reports/rotation_flip_tta_20261106'
out.mkdir(exist_ok=True)
plan={'seeds':[42,43,44],'angles':[5,-5,10,-10],'policies':['raw','flip','rotation5','flip_rotation5','rotation10','flip_rotation10'],'averaging':'equal softmax probabilities; rotation policies include original; combined includes all horizontal mirrors','interpolation':'bilinear, reflection padding, align_corners=False; rotate then flip','split':'validation only','device':'cpu','created_at_utc':datetime.now(timezone.utc).isoformat()}
(out/'plan.json').write_text(json.dumps(plan,indent=2))
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)
rows=[r for r in csv.DictReader((ROOT/'baseline/splits.csv').open()) if r['split']=='validation']
ids=np.array([int(r['id']) for r in rows]); y=np.array([int(r['label']) for r in rows])
images=np.load(ROOT/'.cache/baseline/train.npy',mmap_mode='r')
release=torch.load(ROOT.parent/'DL-project1_Kaggle_best_2026-11-06/model.pt',map_location='cpu',weights_only=True)
mean=torch.tensor(release['mean']).reshape(1,3,1,1);std=torch.tensor(release['std']).reshape(1,3,1,1)
results=[]
for seed in plan['seeds']:
 ck=ROOT/f'experiments/research_sweep/combinations/runs/core_ep240_seed{seed}/best.pt'
 saved=torch.load(ck,map_location='cpu',weights_only=True)
 assert saved['config']['spec']['recipe']==release['recipe']
 model=ResidualNet((48,96,192),3,'blur').to(memory_format=torch.channels_last).eval()
 model.load_state_dict(saved['model_state'],strict=True)
 prior=np.load(ROOT/f'delivery_reports/improvement_audit_20261106/seed{seed}_validation.npz')
 assert np.array_equal(prior['ids'],ids) and np.array_equal(prior['labels'],y)
 views={'raw':torch.from_numpy(prior['raw_logits']).softmax(1),'flip':torch.from_numpy(prior['flip_logits']).softmax(1)}
 for deg in plan['angles']:
  plain=[];flipped=[]
  with torch.inference_mode():
   for start in range(0,len(ids),128):
    x=torch.from_numpy(np.array(images[ids[start:start+128]])).permute(0,3,1,2).float()/255
    a=math.radians(deg);c=math.cos(a);s=math.sin(a)
    theta=x.new_tensor([[c,-s,0],[s,c,0]]).unsqueeze(0).expand(len(x),-1,-1)
    z=F.grid_sample(x,F.affine_grid(theta,x.shape,align_corners=False),mode='bilinear',padding_mode='reflection',align_corners=False)
    z=((z-mean)/std).contiguous(memory_format=torch.channels_last)
    plain.append(model(z).softmax(1));flipped.append(model(z.flip(-1)).softmax(1))
  views[str(deg)]=torch.cat(plain);views[str(deg)+'f']=torch.cat(flipped)
  print(json.dumps({'seed':seed,'angle_completed':deg}),flush=True)
 policies={'raw':views['raw'],'flip':(views['raw']+views['flip'])/2}
 for deg in [5,10]:
  total=views['raw']+views[str(deg)]+views[str(-deg)]
  policies[f'rotation{deg}']=total/3
  policies[f'flip_rotation{deg}']=(total+views['flip']+views[str(deg)+'f']+views[str(-deg)+'f'])/6
 np.savez_compressed(out/f'seed{seed}_probabilities.npz',ids=ids,labels=y,**{k:v.numpy() for k,v in policies.items()})
 for name,prob in policies.items():
  pred=prob.argmax(1).numpy();base=policies['flip'].argmax(1).numpy();ok=pred==y;bok=base==y
  row={'seed':seed,'policy':name,'correct':int(ok.sum()),'accuracy_percent':float(ok.mean()*100),'delta_vs_flip_pp':float((ok.mean()-bok.mean())*100),'corrected_vs_flip':int((ok&~bok).sum()),'regressed_vs_flip':int((~ok&bok).sum()),'nll':float(-prob[torch.arange(len(y)),y].log().mean())}
  results.append(row);print(json.dumps(row),flush=True)
 (out/'run_results.json').write_text(json.dumps(results,indent=2))
summary=[]
for name in plan['policies']:
 runs=[r for r in results if r['policy']==name];d=np.array([r['delta_vs_flip_pp'] for r in runs]);h=4.302652729911275*d.std(ddof=1)/np.sqrt(3)
 summary.append({'policy':name,'n_seeds':3,'mean_accuracy_percent':float(np.mean([r['accuracy_percent'] for r in runs])),'mean_delta_vs_flip_pp':float(d.mean()),'paired_t95_ci_pp':[float(d.mean()-h),float(d.mean()+h)]})
report={'plan':plan,'results':results,'summary':summary,'limitations':'Same reused validation set, exploratory six-policy comparison; unadjusted intervals; no hidden-test or local_test evaluation, no training, no submission replacement','finished_at_utc':datetime.now(timezone.utc).isoformat()}
(out/'results.json').write_text(json.dumps(report,indent=2))
print(json.dumps(summary),flush=True)
