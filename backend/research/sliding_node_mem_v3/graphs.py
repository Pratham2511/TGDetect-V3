"""Bounded inference-only CUDA graph cache for a fixed temporal region."""
import torch


class TemporalGraphs:
    def __init__(self):
        self.entries={}
        self.signature=None

    def __call__(self,module,x,mask):
        if module.training or torch.is_grad_enabled() or x.device.type!='cuda':
            return module(x,mask)
        signature=tuple((p.data_ptr(),p._version,p.dtype,p.device) for p in module.parameters())
        if signature!=self.signature:
            self.entries.clear();self.signature=signature
        key=(tuple(x.shape),x.dtype,x.device)
        if key not in self.entries:
            # Avoid growing graph pools for arbitrary last-batch shapes.
            if len(self.entries)>=2:return module(x,mask)
            a=torch.empty_like(x);b=torch.empty_like(mask)
            a.copy_(x);b.copy_(mask)
            current=torch.cuda.current_stream(x.device)
            warm=torch.cuda.Stream(device=x.device);warm.wait_stream(current)
            with torch.cuda.stream(warm):
                for _ in range(3):module(a,b)
            current.wait_stream(warm)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=module(a,b)
            self.entries[key]=(graph,a,b,out)
        graph,a,b,out=self.entries[key]
        a.copy_(x);b.copy_(mask);graph.replay()
        # Two calls share this pool. Return independent storage to preserve the
        # first result when the second short-memory pass replays the graph.
        return out.clone()
