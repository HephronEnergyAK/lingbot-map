from __future__ import annotations

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


def parse(path):
    return ast.parse((ROOT / path).read_text(encoding="utf-8"))


def function(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


class ResearchEntrypointRegressionTests(unittest.TestCase):
    def test_demo_retains_streaming_windowed_sdpa_and_compile_controls(self):
        source = (ROOT / "demo.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        self.assertIn('choices=["streaming", "windowed"]', source)
        self.assertIn('parser.add_argument("--use_sdpa"', source)
        self.assertIn('parser.add_argument("--compile"', source)
        self.assertIsNotNone(function(tree, "compile_model"))
        self.assertIn("model.inference_streaming(", source)
        self.assertIn("model.inference_windowed(", source)

    def test_benchmark_retains_documented_defaults_and_both_modes(self):
        tree = parse(Path("benchmark") / "methods" / "lingbot_map.py")
        source = (ROOT / "benchmark" / "methods" / "lingbot_map.py").read_text(
            encoding="utf-8"
        )
        init = function(tree, "__init__")
        positional = [argument.arg for argument in init.args.args]
        defaults = dict(
            zip(
                positional[-len(init.args.defaults) :],
                init.args.defaults,
            )
        )

        self.assertEqual(ast.literal_eval(defaults["mode"]), "streaming")
        self.assertFalse(ast.literal_eval(defaults["use_sdpa"]))
        self.assertEqual(ast.literal_eval(defaults["num_scale_frames"]), 8)
        self.assertIn("self.model.inference_streaming(", source)
        self.assertIn("self.model.inference_windowed(", source)
        self.assertIn("'rgb': rgb_list", source)
        self.assertIn("'depth': depth_list", source)
        self.assertIn("'pose': pose_list", source)
        self.assertIn("'intrinsics': intrinsics_list", source)


if __name__ == "__main__":
    unittest.main()
