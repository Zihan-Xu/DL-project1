"""Independent implementations of the registered training variants; no historical edits."""
import copy
import hashlib
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from model import ResNet20, ResidualBlock


class RegularizedBlock(ResidualBlock):
    def forward(self, x):
        y = self.residual[2](self.residual[1](self.residual[0](x)))
        y = F.dropout(y, self.dropout_probability, self.training)
        y = self.residual[4](self.residual[3](y))
        if self.training and self.drop_probability:
            keep = 1 - self.drop_probability
            mask = (torch.rand((len(y),1,1,1), device=y.device) < keep).to(y.dtype)
            y = y * mask / keep
        return self.relu(y + self.shortcut(x))


class Classifier(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = ResNet20(cfg['width'])
        if cfg['dropout'] or cfg['drop_path']:
            for i, b in enumerate(m for m in self.net if isinstance(m, ResidualBlock)):
                b.__class__ = RegularizedBlock
                b.dropout_probability = cfg['dropout']
                b.drop_probability = cfg['drop_path'] * (i+1) / 9
        self.projector = (nn.Sequential(nn.Linear(192,128),nn.ReLU(),nn.Linear(128,128))
                          if cfg['supcon'] else None)

    def forward(self, x, project=False):
        for layer in list(self.net.children())[:-1]:
            x = layer(x)
        logits = self.net[-1](x)
        if project:
            return logits, F.normalize(self.projector(x).float(),dim=1)
        return logits


def sampler(labels, seed, epoch, batch_size):
    """Uniform shuffle followed by minimum class-coverage repair; no repeats/drops."""
    rng = np.random.default_rng([seed, epoch, 1001])
    order = rng.permutation(len(labels))
    bins = np.arange(len(labels)) // batch_size
    n_batches = int(bins[-1])+1
    counts = np.bincount(bins*10+labels[order],minlength=n_batches*10).reshape(-1,10)
    swaps = 0
    for b, cls in np.argwhere(counts==0):
        if counts[b,cls]:
            continue
        start, end = int(b)*batch_size,min((int(b)+1)*batch_size,len(order))
        candidates = np.flatnonzero(labels[order]==cls)
        donors = candidates[(counts[bins[candidates],cls]>1) & (bins[candidates]!=b)]
        if not len(donors):
            raise RuntimeError('Cannot repair class coverage')
        donor = int(donors[0]); d = int(bins[donor])
        receiver = next(j for j in range(start,end) if counts[b,labels[order[j]]]>1)
        displaced = int(labels[order[receiver]])
        order[receiver],order[donor] = order[donor],order[receiver]
        counts[b,cls]+=1; counts[d,cls]-=1
        counts[b,displaced]-=1; counts[d,displaced]+=1
        swaps+=1
    assert counts.min()>=1 and np.all(np.bincount(order,minlength=len(labels))==1)
    return order,counts,swaps


def augmentation_plan(seed, epoch, n, views, rotation_probability):
    # Separate random streams, indexed by source image rather than minibatch position.
    flips = np.random.default_rng([seed,epoch,2002]).random((views,n)) < .5
    rng = np.random.default_rng([seed,epoch,3003])
    angles = rng.uniform(-5,5,(views,n)).astype(np.float32)
    angles *= rng.random((views,n)) < rotation_probability
    return flips,angles


def normalized(images, mean, std):
    return (images.permute(0,3,1,2).float().div_(255)-mean)/std


def augment(images, index, mean, std, flip, angle=None):
    x = normalized(images[index],mean,std)
    x = torch.where(flip[:,None,None,None],x.flip(-1),x)
    if angle is not None:
        theta = torch.zeros(len(index),2,3,device=x.device)
        radians = angle * (math.pi/180)
        theta[:,0,0]=theta[:,1,1]=radians.cos()
        theta[:,0,1]=-radians.sin(); theta[:,1,0]=radians.sin()
        grid=F.affine_grid(theta,x.shape,align_corners=False)
        x=F.grid_sample(x,grid,mode='bilinear',padding_mode='reflection',align_corners=False)
    return x.contiguous(memory_format=torch.channels_last)


def supcon_loss(features, labels, temperature=.1):
    features=F.normalize(features.float(),dim=1)
    similarity=features@features.T/temperature
    eye=torch.eye(len(labels),dtype=torch.bool,device=labels.device)
    positive=labels[:,None].eq(labels[None,:]) & ~eye
    log_denominator=torch.logsumexp(similarity.masked_fill(eye,float('-inf')),dim=1)
    log_probability=similarity-log_denominator[:,None]
    return -(log_probability.masked_fill(~positive,0).sum(1)/positive.sum(1).clamp_min(1)).mean()


def make_optimizer(model,cfg):
    params=list(model.parameters())
    if cfg['exclude_bn_bias']:
        excluded=set()
        for module in model.modules():
            for name,p in module.named_parameters(recurse=False):
                if name=='bias' or isinstance(module,nn.modules.batchnorm._BatchNorm):
                    excluded.add(id(p))
        params=[dict(params=[p for p in params if id(p) not in excluded],weight_decay=cfg['weight_decay']),
                dict(params=[p for p in params if id(p) in excluded],weight_decay=0.)]
    if cfg['optimizer']=='adamw':
        return torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    return torch.optim.SGD(params,lr=cfg['lr'],momentum=cfg['momentum'],
                           weight_decay=cfg['weight_decay'],nesterov=cfg['nesterov'])


def make_scheduler(optimizer,cfg):
    name=cfg['scheduler']
    if name=='multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer,[40,60],gamma=.1)
    if name=='sgdr':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,20,T_mult=2)
    if cfg['warmup'] or name=='swa':
        def factor(completed):
            if name=='swa' and completed>=60:
                return .01/cfg['lr']
            if completed<cfg['warmup']:
                return (completed+1)/cfg['warmup']
            return .5*(1+math.cos(math.pi*(completed-cfg['warmup'])/(cfg['epochs']-cfg['warmup'])))
        return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,cfg['epochs'])


def objective(model,x,y,cfg,teacher=None):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if cfg['supcon']:
            logits,z=model(x,project=True)
        else:
            logits=model(x)
        ce=F.cross_entropy(logits,y,label_smoothing=cfg['smoothing'])
    auxiliary=ce.new_zeros(())
    if cfg['supcon']:
        auxiliary=supcon_loss(z,y,cfg['temperature'])
        loss=ce+cfg['supcon']*auxiliary
    elif teacher is not None:
        t=cfg['kd_temperature']
        with torch.no_grad():
            target=teacher(x).float()
        auxiliary=F.kl_div(F.log_softmax(logits.float()/t,dim=1),F.softmax(target/t,dim=1),reduction='batchmean')*(t*t)
        loss=(1-cfg['kd_alpha'])*ce+cfg['kd_alpha']*auxiliary
    else:
        loss=ce
    return loss,logits,ce.detach(),auxiliary.detach()


def train_step(model,optimizer,x,y,cfg,teacher=None):
    optimizer.zero_grad(set_to_none=True)
    loss,logits,ce,aux=objective(model,x,y,cfg,teacher)
    if not torch.isfinite(loss).item():
        raise FloatingPointError('Non-finite first loss')
    loss.backward()
    parameters=[p for p in model.parameters() if p.grad is not None]
    norm=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters]))
    if not torch.isfinite(norm).item():
        raise FloatingPointError('Non-finite gradient')
    if cfg['sam_rho']:
        with torch.no_grad():
            originals=[p.detach().clone() for p in parameters]
            perturbations=[p.grad*(cfg['sam_rho']/(norm+1e-12)) for p in parameters]
            torch._foreach_add_(parameters,perturbations)
        bn=[m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
        tracking=[m.track_running_stats for m in bn]
        try:
            for m in bn: m.track_running_stats=False
            optimizer.zero_grad(set_to_none=True)
            second,*_=objective(model,x,y,cfg,teacher)
            if not torch.isfinite(second).item():
                raise FloatingPointError('Non-finite SAM second loss')
            second.backward()
            norm2=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in parameters]))
            if not torch.isfinite(norm2).item():
                raise FloatingPointError('Non-finite SAM second gradient')
        finally:
            with torch.no_grad(): torch._foreach_copy_(parameters,originals)
            for m,old in zip(bn,tracking): m.track_running_stats=old
    optimizer.step()
    return loss.detach(),logits.detach(),ce,aux


@torch.no_grad()
def update_average(average, model, decay):
    torch._foreach_lerp_(list(average.parameters()),list(model.parameters()),1-decay)


@torch.no_grad()
def recompute_bn(model,images,mean,std,batch_size=128):
    model.eval()
    bn=[m for m in model.modules() if isinstance(m,nn.modules.batchnorm._BatchNorm)]
    momenta=[m.momentum for m in bn]
    for m in bn: m.reset_running_stats(); m.train()
    seen=0
    for part in images.split(batch_size):
        for m in bn: m.momentum=len(part)/(seen+len(part))
        model(normalized(part,mean,std))
        seen+=len(part)
    for m,momentum in zip(bn,momenta): m.momentum=momentum
    model.eval()


@torch.inference_mode()
def predict(model,images,mean,std,batch_size=128):
    model.eval()
    logits=torch.cat([model(normalized(part,mean,std)).float() for part in images.split(batch_size)])
    if not torch.isfinite(logits).all().item():
        raise FloatingPointError('Non-finite validation logits')
    return logits


def model_cost(cfg):
    total=0
    def count(layer,inputs,output):
        nonlocal total
        total+=output.numel()*layer.weight[0].numel()
    with torch.device('meta'):
        model=Classifier(cfg).eval()
        params=sum(p.numel() for p in model.parameters())
        for m in model.modules():
            if isinstance(m,(nn.Conv2d,nn.Linear)): m.register_forward_hook(count)
        model(torch.empty(1,3,32,32),project=bool(cfg['supcon']))
    project_macs=192*128+128*128 if cfg['supcon'] else 0
    return dict(trainable_parameters=params,inference_parameters=params-(41216 if cfg['supcon'] else 0),
                macs_per_image=total-project_macs,training_forward_macs=total,
                macs_convention='Conv/Linear multiply-accumulates only, one 32x32 RGB image; excludes BN, activation, loss, averaging')
