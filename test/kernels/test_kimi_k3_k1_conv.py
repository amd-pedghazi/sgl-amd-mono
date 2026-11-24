# SPDX-License-Identifier: Apache-2.0
"""K1's conv window staging and hand-back as Triton launches, against the torch
ops they replace (k1.prepare / k1.finish), at Kimi-K3's shapes. Copies only, so
bit for bit, every layer and with a where-mask of the engaged ones."""

import unittest

import torch
from sglang.test.test_utils import CustomTestCase

LAYERS, SLOTS, WIN, C, QLEN = 69, 5, 3, 3 * 1536, 6


@unittest.skipIf(not torch.cuda.is_available(), "needs a GPU")
class TestK1ConvTriton(CustomTestCase):
    def _case(self, where):
        from sgl_amd_mono.kimi_k3.gfx950 import k1

        dev = torch.device("cuda", 0)
        g = torch.Generator(device=dev).manual_seed(7)
        committed = torch.randn(LAYERS, SLOTS, WIN, C, generator=g, device=dev).to(
            torch.bfloat16
        )
        n = 50
        idx = torch.randperm(LAYERS, generator=g, device=dev)[:n].sort().values
        slot = torch.tensor([3], dtype=torch.int64, device=dev)
        row = torch.tensor([2], dtype=torch.int64, device=dev)
        conv = torch.randn(n, C, WIN + QLEN, generator=g, device=dev).to(torch.bfloat16)
        rows = 4
        view_ref = torch.zeros(
            LAYERS, rows, QLEN, WIN, C, dtype=torch.bfloat16, device=dev
        )
        view_tri = view_ref.clone()

        # staging: torch
        ref = conv.clone()
        window = committed.index_select(0, idx).index_select(1, slot).squeeze(1)
        ref[:, :, :WIN] = window.mT
        # staging: triton
        tri = conv.clone()
        k1._stage_window[(n, (C + k1._CONV_BLOCK - 1) // k1._CONV_BLOCK)](
            committed, *committed.stride(), idx, slot, tri, *tri.stride(), C,
            WIN=WIN, BLOCK=k1._CONV_BLOCK,
        )  # fmt: skip
        self.assertTrue(torch.equal(ref, tri), "staging differs")

        # the launch rolls its buffer; the hand-back reads conv[.., WIN:] after it
        after = torch.randn(n, C, WIN + QLEN, generator=g, device=dev).to(
            torch.bfloat16
        )
        sel = torch.arange(n, device=dev) if where is None else where
        lidx = idx[sel]
        seq = torch.cat([window[sel].mT, after[sel][:, :, WIN:]], dim=2)
        windows = seq.unfold(2, WIN, 1)[:, :, 1 : QLEN + 1]
        view_ref[lidx, row] = windows.permute(0, 2, 3, 1)
        k1._hand_back[(sel.shape[0], (C + k1._CONV_BLOCK - 1) // k1._CONV_BLOCK)](
            committed, *committed.stride(), after, *after.stride(), view_tri, *view_tri.stride(),
            lidx, sel, slot, row, C, WIN=WIN, QLEN=QLEN, BLOCK=k1._CONV_BLOCK,
        )  # fmt: skip
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(view_ref, view_tri), "hand-back differs")

    def test_every_layer(self):
        self._case(None)

    def test_engaged_subset(self):
        self._case(
            torch.tensor([0, 3, 4, 17, 40, 49], dtype=torch.int64, device="cuda")
        )


if __name__ == "__main__":
    unittest.main()
