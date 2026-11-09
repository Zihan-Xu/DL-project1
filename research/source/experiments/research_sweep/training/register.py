"""Finite, outcome-independent E21--E34 registration. Never overwrite a frozen plan."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = dict(width=3, epochs=80, batch_size=128, lr=.1, momentum=.9, weight_decay=1e-4,
            optimizer='sgd', scheduler='cosine', warmup=0, exclude_bn_bias=False,
            nesterov=False, smoothing=0., dropout=0., drop_path=0., sam_rho=0.,
            views=1, supcon=0., temperature=.1, kd_temperature=0., kd_alpha=.5,
            flip=.5, rotation_degrees=0., rotation_probability=0., crop=0,
            initialization='project_Kaiming_fan_out', train_precision='BF16', validation_precision='FP32')

def build():
    configs = []
    def add(name, direction, *, control='base', seeds=3, output='online', trajectory=None, **changes):
        configs.append(dict(id=name, direction=direction, control=control, seeds=list(range(42,42+seeds)),
                            output=output, trajectory=trajectory or name, recipe=dict(BASE,**changes)))
    add('base','E21,E22,E23,E24,E25,E26,E27,E28,E29,E30,E32',control=None,seeds=5)
    add('epochs160','E21',epochs=160)
    add('epochs240','E21',epochs=240)
    add('lr005','E22',lr=.05)
    add('lr02','E22',lr=.2)
    add('warmup5','E22',warmup=5)
    add('multistep','E23',scheduler='multistep')
    add('sgdr','E23',scheduler='sgdr')
    add('wd0003','E24',weight_decay=.0003)
    add('wd0005','E24',seeds=5,weight_decay=.0005)
    add('wd_exclude_bn_bias','E24',exclude_bn_bias=True)
    add('nesterov','E25',nesterov=True)
    add('adamw0003','E25',optimizer='adamw',lr=.0003,weight_decay=.01)
    add('adamw001','E25',optimizer='adamw',lr=.001,weight_decay=.01)
    add('batch64','E26',batch_size=64)
    add('batch256','E26',batch_size=256)
    add('ls005','E27',smoothing=.05)
    add('ls01','E27',seeds=5,smoothing=.1)
    add('dropout01','E28',dropout=.1)
    add('dropout02','E28',dropout=.2)
    add('depth01','E29',drop_path=.1)
    add('depth02','E29',drop_path=.2)
    add('sam005','E30',seeds=5,sam_rho=.05)
    add('sam01','E30',sam_rho=.1)
    add('swa_noavg','E31',seeds=5,scheduler='swa',output='swa_control')
    add('swa','E31',control='swa_noavg',seeds=5,scheduler='swa',output='swa',trajectory='swa_noavg')
    add('ema099','E32',output='ema099',trajectory='base')
    add('ema0999','E32',seeds=5,output='ema0999',trajectory='base')
    dual=dict(views=2,rotation_degrees=5.,rotation_probability=.5)
    add('dual_ce','E33',**dual)
    add('supcon005','E33',control='dual_ce',supcon=.05,**dual)
    add('supcon01','E33',control='dual_ce',supcon=.1,**dual)
    add('student_ce','E34',control=None,seeds=5,width=2)
    add('kd_t2','E34',control='student_ce',width=2,kd_temperature=2.)
    add('kd_t4','E34',control='student_ce',seeds=5,width=2,kd_temperature=4.)
    for c in configs:
        c['parameters'] = (2436346 if c['recipe']['width']==3 else 1084586) + (41216 if c['recipe']['supcon'] else 0)
        c['teacher_parameters'] = 2436346 if c['recipe']['kd_temperature'] else 0
        c['steps_per_epoch'] = (40000+c['recipe']['batch_size']-1)//c['recipe']['batch_size']
        c['training_presentations'] = 40000*c['recipe']['epochs']*c['recipe']['views']
    runs=[dict(config=c['id'],seed=s,run_id=f"{c['id']}/seed_{s}") for c in configs for s in c['seeds']]
    trajectories=[]
    # Outcome-independent order: paired controls first within each seed; 45/46 only registered rows.
    for seed in range(42,47):
        for c in configs:
            if c['trajectory']==c['id'] and seed in c['seeds']:
                outputs=[d['id'] for d in configs if d['trajectory']==c['id'] and seed in d['seeds']]
                trajectories.append(dict(config=c['id'],seed=seed,outputs=outputs))
    return dict(version=1,category='training',date='2026-11-04',configs=configs,runs=runs,
                trajectories=trajectories,total_runs=len(runs),total_trajectories=len(trajectories),
                statistics='paired seed difference in percentage points; sample SD; two-sided 95% t CI; Holm over all registered primary comparisons',
                selection='maximum validation correct; ties earlier; SWA and matched control eligible epochs 61..80',
                forbidden=['random crop','Mixup','external training data','external pretrained weights','local_test selection'])

if __name__=='__main__':
    HERE.mkdir(parents=True,exist_ok=True)
    path=HERE/'manifest.json'
    value=build()
    if path.exists():
        assert json.loads(path.read_text())==value, 'Frozen manifest differs'
    else:
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(value['total_runs'],value['total_trajectories'],hashlib.sha256(path.read_bytes()).hexdigest())
