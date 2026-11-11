"""Bitwise shape equivalence across token counts and tensor-parallel sizes."""

import unittest
import torch


class GDNShape(unittest.TestCase):
    def test_patched_expression_preserves_all_values(self):
        for tp in [1, 2, 4]:
            for tokens in [0, 1, 2, 17, 2048]:
                local_heads = 32 // tp
                z = torch.arange(tokens * local_heads * 128).reshape(
                    tokens, local_heads * 128
                )
                new = z.reshape(-1, local_heads, 128)
                if not tokens:
                    self.assertEqual(new.shape, (0, local_heads, 128))
                    continue
                old = z.reshape(z.size(0), -1, 128)
                self.assertEqual(old.shape, new.shape)
                self.assertTrue(torch.equal(old, new))

    def test_patched_expression_traces_with_dynamic_token_counts(self):
        def reshape(z):
            return z.reshape(-1, 16, 128)

        compiled = torch.compile(
            reshape, backend="aot_eager", dynamic=True, fullgraph=True
        )
        for tokens in [1, 17, 2048, 2]:
            z = torch.arange(tokens * 2048).reshape(tokens, 2048)
            self.assertTrue(torch.equal(compiled(z), reshape(z)))


if __name__ == "__main__":
    unittest.main()
