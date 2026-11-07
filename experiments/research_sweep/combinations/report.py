"""Complete registered contrasts, including paired factorial interactions."""
import math
from pathlib import Path
import numpy as np
from scipy import stats
from experiments.research_sweep.combinations.runner import HERE, read, write_csv, atomic


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
    comparisons=[];paired=[]
    for contrast in manifest['comparisons']:
        weights=contrast['weights']
        planned=sorted(set.intersection(*(set(configmap[k]['seeds']) for k in weights)))
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
        assert len(comparisons)==len(manifest['comparisons']) and all(x['complete'] for x in comparisons)
        previous=0.
        for rank,row in enumerate(sorted(comparisons,key=lambda x:x['p_uncorrected'])):
            previous=max(previous,min(1.,row['p_uncorrected']*(len(comparisons)-rank)))
            row['p_holm']=previous
    write_csv(HERE/'paired_seed_differences.csv',paired)
    write_csv(HERE/'comparisons.csv',comparisons)
    heading='全部训练已完成，待独立验收' if final else f'阶段结果：{len(rows)}/{manifest["total_runs"]}'
    lines=['# 绑定实验报告', '',heading+'。', '',
        '这是旧233份单因素研究之后的新组合实验。核心三因素为BlurPool、Cutout12与WD5e-4；扩展参照在训练前固定为core111。core_BCW的三位依次表示上述开关。', '',
        '同一固定5000张验证图，seed42–46（核心与蒸馏）或42–44（扩展）；每run按原始验证正确数选最佳轮次。以下均值为当前已完成seed；未齐备时不作最终胜出判断。', '',
        '|配置|完成/计划seed|验证Top-1均值%|SD/pp|参数|MAC/图|平均训练+验证秒|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        sd=f'{r["sd_accuracy_pp"]:.4f}' if r['sd_accuracy_pp']!='' else '—'
        lines.append(f'|{r["configuration"]}|{r["n"]}/{r["planned_n"]}|{r["mean_accuracy_percent"]:.4f}|{sd}|{r["parameters"]}|{r["macs"]}|{r["mean_gpu_compute_seconds"]:.1f}|')
    lines+=['','## 预注册配对比较与相互作用','',
        '差值单位为百分点。二阶相互作用是在第三因素两个水平上平均的差中之差；三阶为二阶相互作用的变化。单因素结果不相加。原始t区间未做多重校正；完整后Holm对本轮全部注册比较统一校正。', '',
        '|比较|n|差值pp|95% t CI|Holm p|','|---|---:|---:|---|---:|']
    for r in comparisons:
        p=f'{r["p_holm"]:.5g}' if r['p_holm']!='' else '待完整'
        lines.append(f'|{r["comparison"]}|{r["n"]}|{r["mean_pp"]:+.4f}|[{r["ci_low"]:+.4f}, {r["ci_high"]:+.4f}]|{p}|')
    lines+=['','## 适用范围与产物','',
        'CI跨0是未见明确差异，不是等效；即使区间不跨0仍是固定划分、验证选轮与已有结果筛选下的探索性证据。种子重复仅刻画训练随机性，不创造新的独立验证图像。未读取local_test或比赛样本，不发布或提交模型。', '',
        'PreAct与不含Blur的core011比较；双视图直接对照是同旋转单视图core_rot5。240轮及双视图/batch64的额外算力必须计入。学生蒸馏使用项目原固定教师，KD因果比较只对同结构student_ce，教师不会进入学生推理。', '',
        'MAC计Conv/Linear乘加和固定Blur滤波，排除BN/激活/加法/池化和SE元素乘法；训练时间包含每轮验证，持锁时间还含检查点I/O。', '',
        '完整清单与筛选理由：PLAN.md、manifest.json；逐seed：run_metrics.csv；对比：comparisons.csv、paired_seed_differences.csv；各run有预测、完整FP32 logits、best/last/previous与源码/输入哈希。状态只有独立审计通过后才改为completed。']
    atomic(HERE/'REPORT.md',lambda f:f.write('\n'.join(lines)+'\n'))


if __name__=='__main__':report(final=True)
