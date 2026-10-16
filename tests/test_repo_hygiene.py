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
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


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


class TestWorkflowSecurity(unittest.TestCase):

    def test_all_workflows_hardened(self):
        self.assertGreaterEqual(len(WORKFLOWS), 3)
        for wf in WORKFLOWS:
            text = wf.read_text(encoding="utf-8")
            with self.subTest(wf.name):
                text.encode("ascii")
                self.assertRegex(text, r"permissions:\s*\n\s*contents: read")
                self.assertNotIn("pull_request_target", text)
                self.assertNotIn("contents: write", text)
                self.assertNotIn("secrets.", text)
                uses = re.findall(r"uses:\s*(\S+)", text)
                self.assertTrue(uses)
                for u in uses:
                    self.assertRegex(u, r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$", f"surum SHA ile sabitlenmemis: {u}")
                for block in re.findall(r"uses: actions/checkout@\S+.*?\n(?:\s+with:\n(?:\s{10}.*\n)+)?", text):
                    self.assertIn("persist-credentials: false", block)

    def test_integrity_runs_on_every_push_with_full_history(self):
        text = (ROOT / ".github" / "workflows" / "integrity.yml").read_text(encoding="utf-8")
        self.assertNotIn("paths-ignore", text)
        self.assertIn("fetch-depth: 0", text)
        self.assertIn("python -m src.audit", text)
        self.assertIn("schedule:", text)

    def test_dependabot_does_not_touch_python_pins(self):
        text = (ROOT / ".github" / "dependabot.yml").read_text(encoding="utf-8")
        self.assertIn('package-ecosystem: "github-actions"', text)
        self.assertNotIn('package-ecosystem: "pip"', text)

    def test_security_scan_reports_only(self):
        text = (ROOT / ".github" / "workflows" / "security.yml").read_text(encoding="utf-8")
        self.assertRegex(text, r"pip-audit==\d+\.\d+\.\d+")
        self.assertIn("pip-audit -r requirements.txt", text)
        self.assertNotIn("--fix", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
