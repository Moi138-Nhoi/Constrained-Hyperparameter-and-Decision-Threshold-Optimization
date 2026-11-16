import tempfile
import unittest
import hashlib
from pathlib import Path

from scripts.verify_results import BEGIN, END, replace_readme_block, verify_hashes, verify_relative_links
from src.experiment import save_json, save_rows


class ConsistencyTests(unittest.TestCase):
    def test_writers_use_lf_and_hash_check_detects_changed_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            save_json(root / "manifest.json", {"value": "máy", "rows": 1})
            save_rows(root / "data.csv", [{"sensor": 3, "label": 1}])
            hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
            for name in hashes:
                self.assertNotIn(b"\r", (root / name).read_bytes())
            verify_hashes(root, hashes)
            path = root / "data.csv"
            original = path.read_bytes()
            path.write_bytes(original.replace(b"\n", b"\r\n"))
            with self.assertRaisesRegex(AssertionError, "hash differs"):
                verify_hashes(root, hashes)
            path.write_bytes(original.replace(b"3", b"4"))
            with self.assertRaisesRegex(AssertionError, "hash differs"):
                verify_hashes(root, hashes)

    def test_missing_relative_document_link_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            readme = root / "README.md"
            readme.write_text("[report](report.pdf) [UCI](https://archive.ics.uci.edu/)", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "Broken README link"):
                verify_relative_links(readme)
            (root / "report.pdf").write_bytes(b"exists")
            self.assertEqual(verify_relative_links(readme), 1)

    def test_readme_mismatch_is_detected_and_can_be_regenerated(self):
        with tempfile.TemporaryDirectory() as folder:
            readme = Path(folder) / "README.md"
            readme.write_text(f"Title\n{BEGIN}\nold number\n{END}\n", encoding="utf-8")
            with self.assertRaises(AssertionError):
                replace_readme_block(readme, "new number", write=False)
            replace_readme_block(readme, "new number", write=True)
            replace_readme_block(readme, "new number", write=False)
            self.assertIn("new number", readme.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
