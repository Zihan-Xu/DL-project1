"""Finite combination design, without outcome-adaptive additions."""
import itertools

BASE = dict(width=3, epochs=80, batch_size=128, lr=.1, momentum=.9,
            weight_decay=.0001, scheduler='cosine', warmup=0, blur=False,
            cutout=0, dropout=0., drop_path=0., rotation=0., views=1,
            preact=False, se=False, kd_temperature=0.)


def design():
    configs = []
    def add(name, control, n=3, base=None, **changes):
        configs.append(dict(name=name, control=control, seeds=list(range(42, 42+n)), recipe=dict(base or BASE, **changes)))
    for b, c, w in itertools.product(range(2), repeat=3):
        add(f'core_{b}{c}{w}', None if (b,c,w)==(0,0,0) else 'core_000', n=5,
            blur=bool(b), cutout=12*c, weight_decay=.0005 if w else .0001)
    core = configs[-1]['recipe']
    for name, changes in [
        ('core_wd3',dict(weight_decay=.0003)), ('core_cut8',dict(cutout=8)),
        ('core_rot5',dict(rotation=5.)), ('core_rot10',dict(rotation=10.)),
        ('core_drop1',dict(dropout=.1)), ('core_depth1',dict(drop_path=.1)),
        ('core_se8',dict(se=True)), ('core_ep240',dict(epochs=240)),
        ('core_batch64',dict(batch_size=64)), ('core_warmup5',dict(warmup=5)),
    ]:
        add(name, 'core_111', base=core, **changes)
    add('preact_wc', 'core_011', base=core, preact=True, blur=False)
    add('core_dual', 'core_rot5', base=core, views=2, rotation=5.)
    add('student_ce', None, n=5, base=core, width=2)
    add('student_kd4', 'student_ce', n=5, base=core, width=2, kd_temperature=4.)
    comparisons = []
    def contrast(name, weights, kind):
        comparisons.append(dict(name=name, weights=weights, kind=kind))
    for c in configs:
        if c['control']:
            contrast(c['name']+'__vs__'+c['control'], {c['name']:1.,c['control']:-1.}, 'paired')
    contrast('dual_recipe__vs__core111', {'core_dual':1.,'core_111':-1.}, 'recipe')
    for bit, factor in enumerate('BCW'):
        for other in itertools.product(range(2), repeat=2):
            zero = list(other); zero.insert(bit, 0)
            one = list(zero); one[bit] = 1
            a,b = ['core_'+''.join(map(str,x)) for x in (one,zero)]
            # One edge is already an overall comparison; avoid duplicate testing.
            if not any(q['weights']=={a:1.,b:-1.} for q in comparisons):
                contrast(f'{factor}_edge_{a}_{b}',{a:1.,b:-1.},'factor_edge')
    for subset in [(0,1),(0,2),(1,2),(0,1,2)]:
        weights={}
        for bits in itertools.product(range(2),repeat=3):
            weights['core_'+''.join(map(str,bits))] = ((-1.)**sum(1-bits[i] for i in subset))/(2**(3-len(subset)))
        contrast('interaction_'+''.join('BCW'[i] for i in subset),weights,'interaction')
    return configs, comparisons
