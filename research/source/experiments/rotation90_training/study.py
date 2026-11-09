import os,sys,json,time,subprocess,hashlib,csv,types,inspect
from contextlib import nullcontext
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from experiments.research_sweep import runtime
from experiments.research_sweep.combinations import runner as runner,methods as original
from experiments.research_sweep.combinations.register import BASE
CATEGORY='rotation90_training'

def write(path,obj):runtime.atomic_json(path,obj)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def augment(images,ids,norm,flip,angle,cut,cfg):
 x=original.augment(images,ids,norm,flip,angle,cut,dict(cfg,rotation=0,cutout=0))
 if cfg['rot90']:
  x=torch.where((angle>0)[:,None,None,None],torch.rot90(x,1,(-2,-1)),torch.where((angle<0)[:,None,None,None],torch.rot90(x,-1,(-2,-1)),x))
 if cfg['cutout']:
  axis=torch.arange(32,device=x.device);h=cfg['cutout']//2
  rows=(axis[None,:]>=cut[:,1,None]-h)&(axis[None,:]<cut[:,1,None]+h)
  cols=(axis[None,:]>=cut[:,2,None]-h)&(axis[None,:]<cut[:,2,None]+h)
  mask=cut[:,0,None,None].bool()&rows[:,:,None]&cols[:,None,:]
  x=x.masked_fill(mask[:,None],0)
 return x.contiguous(memory_format=torch.channels_last)

def setup():
 torch.set_num_threads(2)
 runner.HERE=HERE;runner.CATEGORY=CATEGORY
 runner.methods=types.SimpleNamespace(**{n:getattr(original,n) for n in ['build','make_scheduler','sampler','augmentation_plan','objective','predict']},augment=augment)

def prepare():
 if (HERE/'manifest.json').exists():return
 setup();cfg=dict(BASE,blur=True,cutout=12,weight_decay=.0005,rot90=False)
 # Pixel preflight: control parity and exact 90 permutations without cutout.
 x=torch.arange(3*32*32*4).remainder(256).reshape(4,32,32,3).to(torch.uint8);ids=torch.arange(4)
 norm=[torch.zeros(1,3,1,1),torch.ones(1,3,1,1)];flip=torch.tensor([False,True,False,True]);ang=torch.tensor([1.,-1.,0.,1.]);cut=torch.tensor([[1,16,16]]*4)
 assert torch.equal(augment(x,ids,norm,flip,ang,cut,cfg),original.augment(x,ids,norm,flip,ang,cut,cfg))
 z=augment(x,ids,norm,flip,ang,cut,dict(cfg,rot90=True,cutout=0));base=original.augment(x,ids,norm,flip,ang,cut,dict(cfg,cutout=0))
 for i,k in enumerate([1,-1,0,1]):assert torch.equal(z[i],torch.rot90(base[i],k,(-2,-1)))
 paths=[Path(__file__),HERE/'PLAN.txt',ROOT/'experiments/research_sweep/combinations/runner.py',ROOT/'experiments/research_sweep/combinations/methods.py',ROOT/'experiments/research_sweep/combinations/register.py',ROOT/'experiments/research_sweep/architecture/runner.py',ROOT/'experiments/research_sweep/architecture/models.py',ROOT/'experiments/research_sweep/training/methods.py',ROOT/'baseline/splits.csv',ROOT/'baseline/data_audit.json',ROOT/'.cache/baseline/train.npy']
 for seed in [42,43,44]:
  folder=ROOT/f'experiments/research_sweep/combinations/runs/core_111_seed{seed}'
  config=read(folder/'config.json')
  assert config['spec']['recipe']=={k:v for k,v in cfg.items() if k!='rot90'}
  runner.verify_completed(folder,config)
  paths.extend([folder/name for name in ['config.json','metrics.json','history.csv','best.pt','validation_predictions.csv']])
 hashes={str(p.relative_to(ROOT)):sha(p) for p in paths};cost=original.cost(cfg)
 runs=[]
 for seed in [42,43,44]:
  for arm in ['flip_rot90']:
   spec=dict(name=arm,recipe=dict(cfg,rot90=arm=='flip_rot90'),teacher_parameters=0,**cost)
   run=dict(run_id=f'{arm}_seed{seed}',seed=seed,spec=spec,input_source_hashes=hashes)
   run['config_sha256']=runner.digest(run);runs.append(run)
 write(HERE/'manifest.json',dict(runs=runs,hashes=hashes,total_runs=3,imported_controls=3,preflight_passed=True,created_at=runtime.utc_now()))

def worker(run):
 setup()
 for p,h in run['input_source_hashes'].items():assert sha(ROOT/p)==h,p
 def status(phase,**fields):write(HERE/'runs'/run['run_id']/'status.json',dict(phase=phase,pid=os.getpid(),updated_at=runtime.utc_now(),**fields))
 runner.status=status
 data,audit=runner.load_data()
 # Local function clone; imported historical runner is never edited on disk.
 source=inspect.getsource(runner.train)
 old="            if not torch.isfinite(loss).item():raise FloatingPointError('Nonfinite training loss')"
 assert source.count(old)==1
 source=source.replace(old,"            # Epoch totals below retain the finite-loss check before checkpoint commit.")
 namespace=dict(runner.__dict__);exec(compile(source,str(HERE/'study.py')+':fast_train','exec'),namespace)
 fast_train=namespace['train']
 augmented_source=source.replace("        net.train();totals=", "        assert cfg['views']==1\n        epoch_augmented=torch.empty((40000,3,32,32),device='cuda',dtype=torch.float32,memory_format=torch.channels_last)\n        for chunk in torch.arange(40000,device='cuda').split(2048):\n            epoch_augmented[chunk]=methods.augment(images,chunk,norm,fp[0,chunk],ang[0,chunk],cu[0,chunk],cfg)\n        net.train();totals=")
 old_x="            x=torch.cat([methods.augment(images,ids,norm,fp[v,ids],ang[v,ids],cu[v,ids],cfg) for v in range(cfg['views'])])"
 assert augmented_source.count(old_x)==1
 augmented_source=augmented_source.replace(old_x,"            x=epoch_augmented[ids].contiguous(memory_format=torch.channels_last)")
 augmented_source=augmented_source.replace("        val_logits=methods.predict", "        del epoch_augmented\n        val_logits=methods.predict")
 aug_namespace=dict(runner.__dict__);exec(compile(augmented_source,str(HERE/'study.py')+':cached_train','exec'),aug_namespace)
 cached_train=aug_namespace['train']
 if not run['run_id'].startswith('benchmark'):
  decision_path=HERE/'sync_optimization.json'
  if not decision_path.exists():
   old_cp=torch.load(HERE/'runs/benchmark_0/last.pt',map_location='cpu',weights_only=False)
   baseline_seconds=float(old_cp['history'][-1]['compute_seconds']);candidates=[]
   for engine,fn in [('deferred_sync',fast_train),('cached_augmentation',cached_train)]:
    probe=json.loads(json.dumps(run));probe['run_id']='benchmark_'+engine;probe['seed']=9242
    probe['spec']['recipe']['epochs']=2;probe['spec']['recipe']['rot90']=False
    probe['config_sha256']=runner.digest({k:v for k,v in probe.items() if k!='config_sha256'})
    fn(probe,data,audit)
    new_cp=torch.load(HERE/'runs'/probe['run_id']/'last.pt',map_location='cpu',weights_only=False)
    identical=all(torch.equal(v,new_cp['model_state'][k]) for k,v in old_cp['model_state'].items())
    optimizer_identical=all(torch.equal(v,new_cp['optimizer_state']['state'][k][name]) for k,state in old_cp['optimizer_state']['state'].items() for name,v in state.items())
    optimized_seconds=float(new_cp['history'][-1]['compute_seconds'])
    candidates.append(dict(engine=engine,model_tensors_bit_identical=identical,optimizer_tensors_bit_identical=optimizer_identical,optimized_epoch_seconds=optimized_seconds,speedup=baseline_seconds/optimized_seconds))
    del new_cp
   eligible=[c for c in candidates if c['model_tensors_bit_identical'] and c['optimizer_tensors_bit_identical'] and c['speedup']>1.02]
   chosen=min(eligible,key=lambda c:c['optimized_epoch_seconds'])['engine'] if eligible else 'original'
   write(decision_path,dict(engine=chosen,baseline_epoch_seconds=baseline_seconds,candidates=candidates,limitation='Two-epoch probes, second epoch timing; timing varies; control trajectory parity checked',source='User requested faster GPU computation without altering experimental recipe'))
   del old_cp
  choice=read(decision_path)
  write(HERE/'runs'/run['run_id']/'execution_engine.json',choice)
  selected={'original':runner.train,'deferred_sync':fast_train,'cached_augmentation':cached_train}[choice['engine']]
  selected(run,data,audit)
 else:runner.train(run,data,audit)

def launch_batch(runs,label,lease_held=False):
 with (nullcontext() if lease_held else runtime.gpu_lock(CATEGORY,label)):
  procs=[]
  try:
   for run in runs:
    folder=HERE/'runs'/run['run_id'];folder.mkdir(parents=True,exist_ok=True)
    write(folder/'launch.json',run)
    log=(folder/'worker.log').open('a',encoding='utf-8')
    p=subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).resolve()),'worker',str(folder/'launch.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    procs.append((p,log))
   while any(p.poll() is None for p,_ in procs):time.sleep(2)
   assert all(p.returncode==0 for p,_ in procs),[(p.pid,p.returncode) for p,_ in procs]
  finally:
   for p,log in procs:
    if p.poll() is None:p.terminate();p.wait()
    log.close()

def final_report(man):
 rows=[];histories={}
 for run in man['runs']:
  folder=HERE/'runs'/run['run_id'];m=runner.verify_completed(folder,run)
  pred=list(csv.DictReader((folder/'validation_predictions.csv').open()))
  assert len(pred)==5000 and sum(r['label']==r['prediction'] for r in pred)==m['validation_correct']
  histories[run['run_id']]=list(csv.DictReader((folder/'history.csv').open()))
  assert len(histories[run['run_id']])==80
  rows.append(m)
 for seed in [42,43,44]:
  folder=ROOT/f'experiments/research_sweep/combinations/runs/core_111_seed{seed}'
  config=read(folder/'config.json');m=runner.verify_completed(folder,config)
  m=dict(m,configuration='flip_only',source='historical audited core_111',source_path=str(folder))
  pred=list(csv.DictReader((folder/'validation_predictions.csv').open()))
  assert len(pred)==5000 and sum(r['label']==r['prediction'] for r in pred)==m['validation_correct']
  histories[m['run_id']]=list(csv.DictReader((folder/'history.csv').open()))
  assert len(histories[m['run_id']])==80
  rows.append(m)
 diffs=[]
 for seed in [42,43,44]:
  a,b=[next(x for x in rows if x['seed']==seed and x['configuration']==arm) for arm in ['flip_only','flip_rot90']]
  for x,y in zip(histories[a['run_id']],histories[b['run_id']]):
   assert x['order_sha256']==y['order_sha256'] and x['view0_sha256']==y['view0_sha256']
  diffs.append(100*(b['validation_accuracy']-a['validation_accuracy']))
 d=np.array(diffs);h=4.302652729911275*d.std(ddof=1)/np.sqrt(3)
 summary={arm:float(np.mean([100*m['validation_accuracy'] for m in rows if m['configuration']==arm])) for arm in ['flip_only','flip_rot90']}
 report=dict(passed=True,summary_percent=summary,paired_delta_pp=diffs,mean_delta_pp=float(d.mean()),paired_t95_ci_pp=[float(d.mean()-h),float(d.mean()+h)],runs=rows,limitations='Fixed reused validation set, best-epoch selection, three seeds, 80-epoch recipe; no hidden-test claims',completed_at=runtime.utc_now())
 write(HERE/'results.json',report)
 with (HERE/'comparison.csv').open('w',newline='') as f:
  w=csv.writer(f);w.writerow(['seed','flip_only_percent','flip_rot90_percent','delta_pp'])
  for seed,diff in zip([42,43,44],diffs):w.writerow([seed,*[100*next(m['validation_accuracy'] for m in rows if m['seed']==seed and m['configuration']==arm) for arm in ['flip_only','flip_rot90']],diff])
 write(HERE/'status.json',dict(phase='completed',completed_runs=3,total_runs=3,imported_controls=3,summary_percent=summary,updated_at=runtime.utc_now()))
 print(json.dumps({k:v for k,v in report.items() if k!='runs'}),flush=True)

def coordinate():
 prepare();setup();man=read(HERE/'manifest.json')
 if not (HERE/'concurrency.json').exists():
  bench=[]
  for i in range(3):
   run=json.loads(json.dumps(man['runs'][0]));run['run_id']=f'benchmark_{i}';run['spec']['recipe']['epochs']=2;run['spec']['recipe']['rot90']=False;run['seed']=9242+i;run['config_sha256']=runner.digest({k:v for k,v in run.items() if k!='config_sha256'});bench.append(run)
  write(HERE/'status.json',dict(phase='benchmarking',completed_runs=0,total_runs=3,updated_at=runtime.utc_now()))
  with runtime.gpu_lock(CATEGORY,'benchmark_comparison'):
   launch_batch(bench[:1],'benchmark_serial',True)
   launch_batch(bench[1:],'benchmark_parallel',True)
  times=[float(list(csv.DictReader((HERE/'runs'/r['run_id']/'history.csv').open()))[-1]['compute_seconds']) for r in bench]
  speedup=2*times[0]/max(times[1:]);concurrency=2 if speedup>=1.05 else 1
  write(HERE/'concurrency.json',dict(second_epoch_seconds=times,estimated_aggregate_speedup=speedup,workers=concurrency,criterion='two only if >=1.05x aggregate epoch throughput',threads_per_worker=2))
 width=read(HERE/'concurrency.json')['workers']
 pending=[r for r in man['runs'] if not (HERE/'runs'/r['run_id']/'metrics.json').exists()]
 for i in range(0,len(pending),width):
  runtime.check_control(CATEGORY)
  batch=pending[i:i+width]
  write(HERE/'status.json',dict(phase='queued',active_runs=[r['run_id'] for r in batch],workers=width,completed_runs=3-len(pending)+i,total_runs=3,imported_controls=3,updated_at=runtime.utc_now()))
  launch_batch(batch,'batch_'+str(i))
 final_report(man)

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='worker':worker(read(sys.argv[2]))
 else:coordinate()
