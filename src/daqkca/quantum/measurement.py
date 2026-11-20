"""Independent binomial shot sampling of ideal fidelity; no hardware gradients."""
import torch


def finite_shot_kernel(k, shots=1024, seed=0, symmetric=True):
    if shots < 1:
        raise ValueError('shots must be positive')
    rng = torch.Generator(device='cpu').manual_seed(seed)
    arr = k.detach().cpu().to(torch.float64).clamp(0,1)
    if symmetric and len(arr)==arr.shape[1]:
        # Unique upper triangle, mirrored to force symmetric measurement.
        ids=torch.triu_indices(len(arr),len(arr),0)
        sims = torch.binomial(torch.full_like(arr[ids[0],ids[1]],float(shots)),
                              arr[ids[0],ids[1]],generator=rng)/shots
        out = torch.zeros_like(arr)
        out[ids[0],ids[1]]=sims
        out[ids[1],ids[0]]=sims
        return out.to(k.device)
    return (torch.binomial(torch.full_like(arr,float(shots)),arr,generator=rng)/shots).to(k.device)
