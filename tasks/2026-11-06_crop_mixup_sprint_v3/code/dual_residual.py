"""A fixed equal-probability combination of two from-scratch residual branches."""
import torch
from torch import nn
import operators

class DualResidual(nn.Module):
    def __init__(self,new_checkpoint,reference_checkpoint,new_policy):
        super().__init__()
        assert new_policy in ['raw','tta']
        self.new_policy=new_policy
        self.new=operators.build(new_checkpoint['recipe'])
        self.reference=operators.build(reference_checkpoint['recipe'])
        self.new.load_state_dict(new_checkpoint['model_state'],strict=True)
        self.reference.load_state_dict(reference_checkpoint['model_state'],strict=True)
        for name,cp in [('new',new_checkpoint),('reference',reference_checkpoint)]:
            for key in ['mean','std']:self.register_buffer(name+'_'+key,torch.tensor(cp[key],dtype=torch.float32).view(1,3,1,1))
        self.parameter_count=sum(p.numel() for p in self.parameters())
        assert self.parameter_count==4872692<=5000000
        assert all(p.requires_grad for p in self.parameters()),'Count both branches without hiding frozen parameters'

    def forward(self,pixels):
        x=pixels.permute(0,3,1,2).float()/255
        a=((x-self.new_mean)/self.new_std).contiguous(memory_format=torch.channels_last)
        b=((x-self.reference_mean)/self.reference_std).contiguous(memory_format=torch.channels_last)
        pa=self.new(a).softmax(1)
        if self.new_policy=='tta':pa=(pa+self.new(a.flip(-1)).softmax(1))/2
        pb=(self.reference(b).softmax(1)+self.reference(b.flip(-1)).softmax(1))/2
        return (pa+pb)/2
