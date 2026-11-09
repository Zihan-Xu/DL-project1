"""Frozen fresh-seed design: three regularizers, length controls, and fixed flip TTA."""
import itertools
from experiments.research_sweep.combinations.register import BASE

SEEDS = [142, 143, 144]


def design():
    configs = []
    base = dict(BASE, blur=True, cutout=12, weight_decay=.0005)
    for r, d, s in itertools.product(range(2), repeat=3):
        name = f'confirm_{r}{d}{s}'
        configs.append(dict(name=name, control=None if name == 'confirm_000' else 'confirm_000',
                            seeds=list(SEEDS), recipe=dict(base, rotation=10. * r,
                                                           dropout=.1 * d, drop_path=.1 * s)))
    for bits in ('000', '111'):
        parent = next(c for c in configs if c['name'] == 'confirm_' + bits)
        configs.append(dict(name='long_' + bits, control=parent['name'], seeds=list(SEEDS),
                            recipe=dict(parent['recipe'], epochs=240)))
    comparisons = []

    def add(name, weights, kind):
        if not any(q['weights'] == weights for q in comparisons):
            comparisons.append(dict(name=name, weights=weights, kind=kind))

    for c in configs:
        if c['control']:
            add(c['name'] + '__vs__' + c['control'],
                {c['name']: 1., c['control']: -1.}, 'paired')
    for bit, factor in enumerate('RDS'):
        for other in itertools.product(range(2), repeat=2):
            zero = list(other); zero.insert(bit, 0)
            one = list(zero); one[bit] = 1
            a, b = ['confirm_' + ''.join(map(str, x)) for x in (one, zero)]
            add(f'{factor}_edge_{a}_{b}', {a: 1., b: -1.}, 'factor_edge')
    for subset in [(0, 1), (0, 2), (1, 2), (0, 1, 2)]:
        weights = {'confirm_' + ''.join(map(str, bits)):
                   (-1.) ** sum(1 - bits[i] for i in subset) / (2 ** (3 - len(subset)))
                   for bits in itertools.product(range(2), repeat=3)}
        add('interaction_' + ''.join('RDS'[i] for i in subset), weights, 'interaction')
    add('long_111__vs__long_000', {'long_111': 1., 'long_000': -1.}, 'paired')
    add('interaction_stack_length', {'long_111': 1., 'long_000': -1.,
                                     'confirm_111': -1., 'confirm_000': 1.}, 'interaction')
    assert len(configs) == 10 and len(comparisons) == 24
    return configs, comparisons


def inference_comparisons(configs):
    return [dict(name=c['name'] + '__tta_vs_raw', kind='inference',
                 weights={c['name'] + '__tta': 1., c['name']: -1.}) for c in configs]
