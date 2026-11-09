import tempfile
import unittest
from pathlib import Path

from scripts.verify_results import BEGIN, END, replace_readme_block


class ConsistencyTests(unittest.TestCase):
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
