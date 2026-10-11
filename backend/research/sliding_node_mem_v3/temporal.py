"""Optional full causal attention for short fixed windows, requiring retraining."""
import torch
from torch import nn
from torch.nn import functional as F


class CausalAttention(nn.Module):
    def __init__(self,dim=128,heads=4):
        super().__init__()
        self.heads=heads
        self.qkv=nn.Linear(dim,3*dim)
        self.output=nn.Linear(dim,dim)
        self.norm=nn.LayerNorm(dim)

    def forward(self,x,valid):
        b,l,d=x.shape
        q,k,v=self.qkv(x).reshape(b,l,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        causal=torch.ones(l,l,device=x.device,dtype=torch.bool).tril()
        allowed=causal[None,None]&valid[:,None,None,:]
        # Give padded queries a finite softmax row; their output is zeroed.
        allowed=allowed|(~valid[:,None,:,None]&torch.eye(l,device=x.device,dtype=torch.bool)[None,None])
        h=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed,dropout_p=0.)
        h=h.transpose(1,2).reshape(b,l,d)
        return self.norm(x+self.output(h))*valid[...,None]
