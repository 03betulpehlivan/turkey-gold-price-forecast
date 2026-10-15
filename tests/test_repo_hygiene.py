"""
Depo hijyeni testleri: requirements.txt kodlamasi ve CI is akisi
Calistirma: python -m unittest tests.test_repo_hygiene -v
"""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
REQ = ROOT / "requirements.txt"
CI = ROOT / ".github" / "workflows" / "tests.yml"


class TestRequirements(unittest.TestCase):

    def test_plain_ascii_without_bom(self):
        data = REQ.read_bytes()
        self.assertFalse(data.startswith(b"\xef\xbb\xbf") or data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"),
                         "requirements.txt BOM'lu ya da UTF-16 (PowerShell '>' ile yazilmis olabilir)")
        data.decode("ascii")
        self.assertNotIn(b"\x00", data)

    def test_core_packages_pinned(self):
        lines = [l.split("#")[0].strip() for l in REQ.read_text(encoding="ascii").splitlines()]
        pins = dict(l.split("==") for l in lines if l)
        for pkg in ("pandas", "numpy", "scikit-learn", "xgboost", "scipy", "arch", "yfinance"):
            self.assertIn(pkg, pins)
            self.assertRegex(pins[pkg], r"^\d+(\.\d+)+$")


class TestWorkflow(unittest.TestCase):

    def setUp(self):
        self.ci = CI.read_text(encoding="utf-8")

    def test_runs_full_suite_on_project_python(self):
        self.assertIn('python-version: "3.11"', self.ci)
        self.assertIn("pip install -r requirements.txt", self.ci)
        self.assertIn("python -m unittest discover -s tests", self.ci)

    def test_read_only_and_no_secrets(self):
        self.assertRegex(self.ci, r"permissions:\s*\n\s*contents: read")
        for bad in ("secrets.", "EVDS_API_KEY", "contents: write", "git push"):
            self.assertNotIn(bad, self.ci)

    def test_daily_live_commits_do_not_trigger(self):
        self.assertIn('- "research/live/**"', self.ci)
        ignored = re.search(r"paths-ignore:\n((?:\s+- .*\n)+)", self.ci).group(1)
        self.assertNotIn("src/", ignored)
        self.assertNotIn("tests/", ignored)


if __name__ == "__main__":
    unittest.main(verbosity=2)
