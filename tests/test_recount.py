"""Offline regression tests for the published experiment's input integrity."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("recount", ROOT / "src" / "recount.py")
recount = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recount)


class RecountInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copytree(ROOT / "data", self.root / "data")
        (self.root / "results").mkdir()
        for name in recount.RESULT_FILES:
            shutil.copyfile(ROOT / "results" / name, self.root / "results" / name)

    def run_recount(self):
        output = io.StringIO()
        with patch.object(recount, "ROOT", self.root), contextlib.redirect_stdout(output):
            code = recount.main()
        return code, output.getvalue()

    def test_published_inputs_reproduce_tables(self):
        code, output = self.run_recount()
        self.assertEqual(code, 0)
        self.assertIn("gemini-3.1-flash-lite: n=93", output)
        self.assertIn("gemini-3.5-flash-lite: n=122", output)
        self.assertIn("gemma-4-31b-it: n=87", output)
        self.assertIn("1388        531          1        0", output)

    def test_missing_result_file_fails(self):
        (self.root / "results" / recount.RESULT_FILES[0]).unlink()
        code, output = self.run_recount()
        self.assertEqual(code, 1)
        self.assertIn("Missing required input files", output)

    def test_missing_dataset_fails(self):
        (self.root / "data" / "spendaudit.jsonl").unlink()
        self.assertEqual(self.run_recount()[0], 1)

    def test_empty_result_file_fails(self):
        (self.root / "results" / recount.RESULT_FILES[0]).write_text("")
        code, output = self.run_recount()
        self.assertEqual(code, 1)
        self.assertIn("640 missing", output)

    def test_missing_api_error_row_fails(self):
        path = self.root / "results" / recount.RESULT_FILES[0]
        rows = path.read_text().splitlines()
        pos = next(i for i, row in enumerate(rows) if "error" in json.loads(row))
        del rows[pos]
        path.write_text("\n".join(rows) + "\n")
        code, output = self.run_recount()
        self.assertEqual(code, 1)
        self.assertIn("1 missing", output)

    def test_duplicate_response_fails(self):
        path = self.root / "results" / recount.RESULT_FILES[0]
        with path.open("a") as out:
            out.write(path.read_text().splitlines()[0] + "\n")
        code, output = self.run_recount()
        self.assertEqual(code, 1)
        self.assertIn("1 duplicate", output)

    def test_unexpected_response_fails(self):
        path = self.root / "results" / recount.RESULT_FILES[0]
        row = json.loads(path.read_text().splitlines()[0])
        row["model"] = "unregistered-model"
        with path.open("a") as out:
            out.write(json.dumps(row) + "\n")
        code, output = self.run_recount()
        self.assertEqual(code, 1)
        self.assertIn("1 unexpected", output)

    def test_missing_dataset_pair_fails(self):
        path = self.root / "data" / "spendaudit.jsonl"
        path.write_text("\n".join(path.read_text().splitlines()[2:]) + "\n")
        self.assertEqual(self.run_recount()[0], 1)

    def test_duplicate_item_id_fails(self):
        path = self.root / "data" / "spendaudit.jsonl"
        rows = path.read_text().splitlines()
        rows[-1] = rows[0]
        path.write_text("\n".join(rows) + "\n")
        self.assertEqual(self.run_recount()[0], 1)


if __name__ == "__main__":
    unittest.main()
