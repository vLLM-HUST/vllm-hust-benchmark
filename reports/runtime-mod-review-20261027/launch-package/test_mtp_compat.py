import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location(
    "vspec_compat", Path(__file__).with_name("vspec_compat.py")
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GroupHookTest(unittest.TestCase):
    def test_preserves_builder_identity_arguments_and_groups(self):
        calls = []
        groups = [SimpleNamespace(is_eagle_group=False)]

        def get_kv_cache_groups(vllm_config, kv_cache_spec):
            calls.append(("build", vllm_config, kv_cache_spec))
            return groups

        def annotate(config, spec, actual):
            self.assertIs(actual, groups)
            calls.append(("annotate", config, spec))
            actual[0].is_eagle_group = True

        host = SimpleNamespace(get_kv_cache_groups=get_kv_cache_groups)
        self.assertTrue(module.patch_mtp_group_hook(host, annotate))
        self.assertIs(host.get_kv_cache_groups("cfg", "spec"), groups)
        self.assertEqual(calls, [("build", "cfg", "spec"), ("annotate", "cfg", "spec")])
        self.assertTrue(groups[0].is_eagle_group)
        self.assertFalse(module.patch_mtp_group_hook(host, annotate))

    def test_unsupported_signature_rejected(self):
        host = SimpleNamespace(get_kv_cache_groups=lambda other: [])
        with self.assertRaises(AssertionError):
            module.patch_mtp_group_hook(host, lambda *args: None)


if __name__ == "__main__":
    unittest.main()
