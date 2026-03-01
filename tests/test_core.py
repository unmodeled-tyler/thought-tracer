from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from logitlens_tui.lens import sanitize_token_text
from logitlens_tui.modeling import ModelArtifactError, ensure_real_weights, is_lfs_pointer


class PointerDetectionTests(unittest.TestCase):
    def test_is_lfs_pointer_detects_pointer_header(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.safetensors"
            path.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:abc\n"
                "size 123\n",
                encoding="utf-8",
            )
            self.assertTrue(is_lfs_pointer(path))

    def test_ensure_real_weights_rejects_pointer_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_text(
                "version https://git-lfs.github.com/spec/v1\n",
                encoding="utf-8",
            )

            with self.assertRaises(ModelArtifactError):
                ensure_real_weights(model_dir)


class TokenSanitizationTests(unittest.TestCase):
    def test_sanitize_token_text_replaces_control_characters(self) -> None:
        self.assertEqual(sanitize_token_text("a\nb\t"), "a\\nb\\t")

    def test_sanitize_token_text_marks_empty_tokens(self) -> None:
        self.assertEqual(sanitize_token_text(""), "<empty>")


if __name__ == "__main__":
    unittest.main()
