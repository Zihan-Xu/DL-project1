"""Complete registered contrasts, including paired factorial interactions."""
import math
from pathlib import Path
import numpy as np
from scipy import stats
from experiments.research_sweep.confirmation.runner import HERE, read, write_csv, atomic


def report(final=False):
    manifest=read(HERE/'manifest.json')
    rows=[];by={}
    for run in manifest['runs']:
        path=HERE/'runs'/run['run_id']/'metrics.json'
        if not path.exists():continue
        m=read(path);by[(m['configuration'],m['seed'])]=m
        rows.append({k:m[k] for k in ['run_id','configuration','seed','validation_correct','validation_n','validation_accuracy',
            'best_epoch','epochs_completed','macro_f1','worst_class_recall','validation_nll','parameters','macs',
            'teacher_parameters','gpu_compute_seconds','lock_seconds','peak_allocated_bytes','optimizer_updates','training_views']})
    if final:assert len(rows)==manifest['total_runs']
    inference_rows=[]
    for run in manifest['runs']:
        key=(run['spec']['name'],run['seed'])
        if key not in by:continue
        raw=by[key]
        inference_rows.append(dict(configuration=key[0],seed=key[1],policy='raw',validation_correct=raw['validation_correct'],
            validation_accuracy=raw['validation_accuracy'],validation_n=5000,macro_f1=raw['macro_f1'],
            worst_class_recall=raw['worst_class_recall'],validation_nll=raw['validation_nll'],
            parameters=raw['parameters'],inference_macs=raw['macs']))
        path=HERE/'runs'/run['run_id']/'tta_metrics.json'
        if path.exists():
            tta=read(path);by[(key[0]+'__tta',key[1])]=tta
            inference_rows.append(dict(configuration=key[0],seed=key[1],policy='tta',validation_correct=tta['validation_correct'],
                validation_accuracy=tta['validation_accuracy'],validation_n=5000,macro_f1=tta['macro_f1'],
                worst_class_recall=tta['worst_class_recall'],validation_nll=tta['validation_nll'],
                parameters=tta['parameters'],inference_macs=tta['inference_macs']))
        elif final:raise AssertionError('Missing fixed inference evaluation: '+run['run_id'])
    write_csv(HERE/'inference_run_metrics.csv',inference_rows)
    write_csv(HERE/'run_metrics.csv',rows)
    summary=[]
    for spec in manifest['configs']:
        ms=[by[(spec['name'],s)] for s in spec['seeds'] if (spec['name'],s) in by]
        if not ms:continue
        values=np.array([m['validation_correct']*.02 for m in ms])
        summary.append(dict(configuration=spec['name'],n=len(ms),planned_n=len(spec['seeds']),
            mean_accuracy_percent=float(values.mean()),sd_accuracy_pp=float(values.std(ddof=1)) if len(values)>1 else '',
            mean_macro_f1=float(np.mean([m['macro_f1'] for m in ms])),
            mean_worst_class_recall=float(np.mean([m['worst_class_recall'] for m in ms])),
            mean_nll=float(np.mean([m['validation_nll'] for m in ms])),
            parameters=spec['parameters'],macs=spec['macs'],teacher_parameters=spec['teacher_parameters'],
            mean_gpu_compute_seconds=float(np.mean([m['gpu_compute_seconds'] for m in ms])),
            mean_lock_seconds=float(np.mean([m['lock_seconds'] for m in ms])),
            mean_peak_allocated_bytes=float(np.mean([m['peak_allocated_bytes'] for m in ms]))))
    write_csv(HERE/'summary.csv',summary)
    configmap={c['name']:c for c in manifest['configs']}
    inference_summary=[]
    for spec in manifest['configs']:
        for policy in ('raw','tta'):
            ms=[r for r in inference_rows if r['configuration']==spec['name'] and r['policy']==policy]
            if not ms:continue
            values=[m['validation_correct']*.02 for m in ms]
            inference_summary.append(dict(configuration=spec['name'],policy=policy,n=len(ms),planned_n=3,
                mean_accuracy_percent=float(np.mean(values)),sd_accuracy_pp=float(np.std(values,ddof=1)) if len(values)>1 else '',
                total_correct=sum(m['validation_correct'] for m in ms),mean_macro_f1=float(np.mean([m['macro_f1'] for m in ms])),
                mean_worst_class_recall=float(np.mean([m['worst_class_recall'] for m in ms])),
                mean_nll=float(np.mean([m['validation_nll'] for m in ms])),parameters=spec['parameters'],
                inference_macs=spec['macs']*(2 if policy=='tta' else 1),epochs=spec['recipe']['epochs']))
    write_csv(HERE/'inference_summary.csv',inference_summary)
    all_comparisons=manifest['comparisons']+manifest['inference_comparisons']
    comparisons=[];paired=[]
    for contrast in all_comparisons:
        weights=contrast['weights']
        planned=sorted(set.intersection(*(set(configmap[k.removesuffix('__tta')]['seeds']) for k in weights)))
        seeds=[s for s in planned if all((k,s) in by for k in weights)]
        d=np.array([sum(w*by[k,s]['validation_correct']*.02 for k,w in weights.items()) for s in seeds])
        for s,value in zip(seeds,d):paired.append(dict(comparison=contrast['name'],seed=s,difference_pp=float(value)))
        if len(d)<2:continue
        mean=float(d.mean());sd=float(d.std(ddof=1));se=sd/math.sqrt(len(d))
        half=float(stats.t.ppf(.975,len(d)-1))*se
        p=float(2*stats.t.sf(abs(mean/se),len(d)-1)) if se else (1. if mean==0 else 0.)
        comparisons.append(dict(comparison=contrast['name'],kind=contrast['kind'],n=len(d),planned_n=len(planned),
            seeds=','.join(map(str,seeds)),mean_pp=mean,sd_pp=sd,ci_low=mean-half,ci_high=mean+half,
            positive_seeds=int((d>0).sum()),p_uncorrected=p,p_holm='',complete=len(d)==len(planned)))
    # No partial-family corrected significance claims while training remains incomplete.
    if final:
        assert len(comparisons)==len(all_comparisons) and all(x['complete'] for x in comparisons)
        previous=0.
        for rank,row in enumerate(sorted(comparisons,key=lambda x:x['p_uncorrected'])):
            previous=max(previous,min(1.,row['p_uncorrected']*(len(comparisons)-rank)))
            row['p_holm']=previous
    write_csv(HERE/'paired_seed_differences.csv',paired)
    write_csv(HERE/'comparisons.csv',comparisons)
    heading='全部30项训练及TTA已完成，待独立验收' if final else f'阶段结果：{len(rows)}/{manifest["total_runs"]}'
    lines=['# 新种子组合复验报告','',heading+'。','',
        '本轮固定BlurPool+Cutout12+WD5e-4，confirm_RDS依次表示旋转10度、Dropout.1、随机深度.1；long为240轮，其余80轮。新seed142–144，各配置3次，固定40000训练/5000验证。原图最佳轮次固定后才比较TTA，后者是原图和水平翻转的等权softmax概率平均。','',
        '|配置|seed完成/计划|原图均值%|SD/pp|参数|MAC/图|平均训练+验证秒|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        sd=f'{r["sd_accuracy_pp"]:.4f}' if r['sd_accuracy_pp']!='' else '—'
        lines.append(f'|{r["configuration"]}|{r["n"]}/{r["planned_n"]}|{r["mean_accuracy_percent"]:.4f}|{sd}|{r["parameters"]}|{r["macs"]}|{r["mean_gpu_compute_seconds"]:.1f}|')
    lines+=['','## 两种推理方案','',
        '|配置|推理|n|均值%|SD/pp|macro-F1|最差类召回|NLL|推理MAC/图|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in inference_summary:
        sd=f'{r["sd_accuracy_pp"]:.4f}' if r['sd_accuracy_pp']!='' else '—'
        lines.append(f'|{r["configuration"]}|{r["policy"]}|{r["n"]}|{r["mean_accuracy_percent"]:.4f}|{sd}|{r["mean_macro_f1"]:.5f}|{r["mean_worst_class_recall"]:.5f}|{r["mean_nll"]:.5f}|{r["inference_macs"]}|')
    lines+=['','## 全部34项预注册配对比较','',
        '差值为百分点。使用共同新seed，95%t区间未校正，Holm p对24项训练比较与10项TTA比较统一校正。只有完整后才计算Holm；小n、固定验证集、选轮和候选筛选限制外推。','',
        '|比较|n|差值pp|95% t CI|Holm p|','|---|---:|---:|---|---:|']
    for r in comparisons:
        ph=f'{r["p_holm"]:.5g}' if r['p_holm']!='' else '待完整'
        lines.append(f'|{r["comparison"]}|{r["n"]}|{r["mean_pp"]:+.4f}|[{r["ci_low"]:+.4f}, {r["ci_high"]:+.4f}]|{ph}|')
    if final:
        ranked=sorted(inference_summary,key=lambda r:(-r['total_correct'],r['policy']!='raw',r['epochs'],r['configuration']))
        winner=ranked[0]
        selection=dict(status='pending_independent_acceptance',rule='max sum validation correct over seeds142-144; exact tie: raw, fewer epochs, configuration name',
            selected_configuration=winner['configuration'],selected_policy=winner['policy'],seeds=[142,143,144],
            deployment_seed=42,mean_accuracy_percent=winner['mean_accuracy_percent'],ranking=ranked)
        from experiments.research_sweep.confirmation.runner import write_json
        write_json(HERE/'SELECTION.json',selection)
        lines+=['','## 预定规则选出的候选','',
            f'暂定（待验收）：{winner["configuration"]} / {winner["policy"]}，均值{winner["mean_accuracy_percent"]:.4f}%。此为已测候选均值排名，不宣称总体精度确定最优；部署固定seed42，不挑最好seed。']
    lines+=['','## 范围和产物','',
        '未读取local_test或比赛样本进行方案选择。种子重复不创造新的独立图像；CI跨0不是等效。没有按结果增减seed。TTA无新增参数，但推理前向与MAC为2倍。MAC计Conv/Linear和固定Blur，排除BN/激活/加法/池化。训练时间含每轮验证，持锁时间另含I/O。','',
        'PLAN.md、manifest.json、run_metrics.csv、inference_run_metrics.csv、summary.csv、inference_summary.csv、comparisons.csv、paired_seed_differences.csv；各run保留best/last/previous、原始和翻转logits、TTA概率和逐图预测。全部产物与报告审计通过后才改为completed。']
    atomic(HERE/'REPORT.md',lambda f:f.write('\n'.join(lines)+'\n'))


if __name__=='__main__':report(final=True)
