"""Run inside the training container on an allocated GPU; no model weights needed."""
from unsloth import FastModel  # Validate the training stack before PyTorch imports.
import torch
from causal_conv1d import causal_conv1d_fn


def main():
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required')
    for dtype in (torch.float32,torch.bfloat16):
        torch.manual_seed(3407)
        x=torch.randn(2,128,256,device='cuda',dtype=dtype,requires_grad=True)
        weight=torch.randn(128,4,device='cuda',dtype=dtype,requires_grad=True)
        actual=causal_conv1d_fn(x,weight,activation='silu')
        expected=torch.nn.functional.silu(torch.nn.functional.conv1d(x,weight.unsqueeze(1),padding=3,groups=128)[...,:256])
        torch.testing.assert_close(actual,expected,rtol=0.03,atol=0.03)
        actual.float().square().mean().backward()
        assert torch.isfinite(x.grad).all() and torch.isfinite(weight.grad).all()
        print(f'{dtype}: optimized forward agrees with reference; backward gradients are finite',flush=True)

if __name__=='__main__':main()
