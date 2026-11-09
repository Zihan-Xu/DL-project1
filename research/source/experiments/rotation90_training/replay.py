import csv,json,time,sys,hashlib
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from experiments.research_sweep.architecture.models import ResidualNet
from experiments.research_sweep import runtime
torch.set_num_threads(2)
rows=[r for r in csv.DictReader((ROOT/'baseline/splits.csv').open()) if r['split']=='validation']
ids=np.array([int(r['id']) for r in rows]);labels=np.array([int(r['label']) for r in rows])
raw=np.load(ROOT/'.cache/baseline/train.npy',mmap_mode='r')
audit=json.loads((ROOT/'baseline/data_audit.json').read_text())
mean=torch.tensor(audit['train_rgb_mean']).reshape(1,3,1,1);std=torch.tensor(audit['train_rgb_std']).reshape(1,3,1,1)
results=[]
for seed in [42,43,44]:
 folder=HERE/'runs'/f'flip_rot90_seed{seed}'
 while not (folder/'metrics.json').exists():
  runtime.check_control('rotation90_training');time.sleep(5)
 metrics=json.loads((folder/'metrics.json').read_text())
 ck=folder/'best.pt'
 with ck.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==metrics['artifact_sha256']['best.pt']
 saved=torch.load(ck,map_location='cpu',weights_only=True)
 net=ResidualNet((48,96,192),3,'blur').to(memory_format=torch.channels_last).eval()
 net.load_state_dict(saved['model_state'],strict=True)
 batches=[]
 with torch.inference_mode():
  for start in range(0,len(ids),128):
   x=torch.from_numpy(np.array(raw[ids[start:start+128]])).permute(0,3,1,2).float()/255
   x=((x-mean)/std).contiguous(memory_format=torch.channels_last)
   batches.append(net(x))
 logits=torch.cat(batches).numpy();pred=logits.argmax(1)
 expected=list(csv.DictReader((folder/'validation_predictions.csv').open()))
 assert np.array_equal(ids,[int(r['id']) for r in expected])
 assert np.array_equal(labels,[int(r['label']) for r in expected])
 expected_pred=np.array([int(r['prediction']) for r in expected])
 result={'seed':seed,'correct':int((pred==labels).sum()),'reported_correct':metrics['validation_correct'],'prediction_mismatches':int((pred!=expected_pred).sum()),'max_abs_logit_difference':float(np.abs(logits-np.load(folder/'validation_logits.npy')).max()),'checkpoint_hash_verified':True}
 result['passed']=result['prediction_mismatches']==0 and result['correct']==result['reported_correct']
 results.append(result)
 runtime.atomic_json(HERE/'cpu_replay.json',{'completed':len(results),'total':3,'all_passed':all(r['passed'] for r in results),'runs':results})
 print(json.dumps(result),flush=True)
print('All CPU replays completed',flush=True)
