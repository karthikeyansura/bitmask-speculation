"""Installed-package smoke check for a terminal or a Colab !python cell.

Exit 2 means no suitable CUDA runtime; exit 1 means a failed validation.
This runner neither installs dependencies nor changes Python's import path.
"""
import json


def main() -> int:
    import torch
    if not torch.cuda.is_available():
        print(json.dumps({'status':'unavailable','reason':'NVIDIA CUDA required'}))
        return 2
    from bitmask_speculation.kernels.triton import masked_reduce
    device=torch.device('cuda',torch.cuda.current_device())
    logits=torch.zeros(33,dtype=torch.float32,device=device)
    logits[31]=3
    mask=torch.tensor([-2147483648,-1],dtype=torch.int32,device=device)
    result=masked_reduce(logits,mask).check()
    assert result.token_id.item()==31
    assert result.valid_count.item()==2
    print(json.dumps({'status':'passed','gpu':torch.cuda.get_device_name(device),
                      'torch':torch.__version__,'cuda':torch.version.cuda,
                      'token_id':31,'valid_count':2}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
