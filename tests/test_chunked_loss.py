import unittest
import torch
from common.training import causal_loss, chunked_causal_loss

class ChunkedLossTests(unittest.TestCase):
    def test_shift_mask_loss_and_gradient(self):
        torch.manual_seed(6304)
        for dtype in (torch.float32, torch.float16):
            for chunk in (1, 7, 32):
                with self.subTest(dtype=dtype, chunk=chunk):
                    a=torch.randn(2,19,97,dtype=dtype,requires_grad=True)
                    b=a.detach().clone().requires_grad_()
                    labels=torch.randint(0,97,(2,19)); labels[:,:5]=-100; labels[1,10:]=-100
                    x=causal_loss(a,labels); y=chunked_causal_loss(b,labels,chunk)
                    torch.testing.assert_close(x,y,atol=3e-5,rtol=2e-7)
                    (x/23).backward(); (y/23).backward()
                    torch.testing.assert_close(a.grad,b.grad,atol=2e-6 if dtype==torch.float16 else 1e-8,rtol=1e-4 if dtype==torch.float16 else 1e-6)
                    self.assertEqual(torch.count_nonzero(b.grad[:,-1]).item(),0)
    def test_all_ignored(self):
        a=torch.randn(2,5,13,requires_grad=True)
        labels=torch.full((2,5),-100)
        loss=chunked_causal_loss(a,labels); loss.backward()
        self.assertEqual(loss.item(),0); self.assertEqual(torch.count_nonzero(a.grad).item(),0)
