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
BULUT = "sabah_bulut.yml"          # gizli anahtar kullanan ve depoya yazan tek gorev (bulut karne kaydi)


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
                self.assertNotIn("pull_request_target", text)
                if wf.name != BULUT:                                   # tek istisna; kendi testiyle denetlenir
                    self.assertRegex(text, r"permissions:\s*\n\s*contents: read")
                    self.assertNotIn("contents: write", text)
                    self.assertNotIn("secrets.", text)
                uses = re.findall(r"uses:\s*(\S+)", text)
                self.assertTrue(uses)
                for u in uses:
                    self.assertRegex(u, r"^[\w.-]+/[\w.-]+@[0-9a-f]{40}$", f"surum SHA ile sabitlenmemis: {u}")
                for block in re.findall(r"uses: actions/checkout@\S+.*?\n(?:\s+with:\n(?:\s{10}.*\n)+)?", text):
                    self.assertIn("persist-credentials: false", block)

    def test_cloud_morning_job_is_the_only_writer_and_tightly_scoped(self):
        """sabah_bulut.yml: gizli anahtar kullanan ve depoya yazan TEK gorev. Yalnizca zamanlayici/elle calisir,
        yalnizca EVDS anahtarini kullanir, yalnizca karne kaydini commit eder, kayit penceresinin icinde calisir."""
        text = (ROOT / ".github" / "workflows" / BULUT).read_text(encoding="utf-8")
        tetik = text[text.index("\non:"): text.index("\npermissions:")]
        self.assertIn("schedule:", tetik)
        self.assertIn("workflow_dispatch:", tetik)
        for bad in ("push", "pull_request", "workflow_run", "issue", "release"):
            self.assertNotIn(bad, tetik)
        self.assertEqual(set(re.findall(r"secrets\.(\w+)", text)), {"EVDS_API_KEY"})
        self.assertRegex(text, r"\npermissions:\n  contents: write\n\n")                 # baska yetki yok
        crons = re.findall(r'cron: "(\d+) (\d+) \* \* 1-5"', text)
        self.assertGreaterEqual(len(crons), 2)
        self.assertEqual(len(crons), text.count("cron:"))                                 # hepsi hafta ici
        for m, h in crons:
            ist = (int(h) + 3) * 60 + int(m)                                              # UTC -> Istanbul
            self.assertTrue(8 * 60 <= ist <= 13 * 60 + 15, f"kayit penceresi disinda: {h}:{m} UTC")
        self.assertIn("pip install -r requirements-chronos.txt", text)
        self.assertIn("python -m src.tool.bulut_veri", text)
        self.assertIn("src.tool.canli_karne kaydet --chronos-zorunlu --gec-uyari", text)    # yedek modelle kayit yok
        self.assertIn("f=research/tool/canli_aralik_log.csv", text)
        self.assertIn('git add -- "$f"', text)
        self.assertNotRegex(text, r"git add (-A|\.|--all)")
        self.assertNotIn("--force", text)
        self.assertIn("timeout-minutes:", text)
        bekci = text[text.index("- name: Bekci"):]
        self.assertIn("if: github.event.schedule == '5 10 * * 1-5'", bekci)      # yalnizca son denemede
        self.assertIn("python -m src.tool.canli_karne denetle", bekci)
        self.assertFalse((ROOT / ".github" / "workflows" / "watchdog.yml").exists())  # eski bilgisayar bekcisi yok
        self.assertLess(text.index("python -m src.tool.site"), text.index("- name: Bekci"))   # alarmda da site
        self.assertIn("path: _site", text)
        yayin = text[text.index("\n  yayin:"):]
        self.assertIn("if: always() && needs.karne.outputs.site == 'evet'", yayin)
        self.assertRegex(yayin, r"permissions:\n      pages: write\n      id-token: write\n    environment:")
        self.assertNotIn("secrets.", yayin)
        self.assertNotIn("contents:", yayin)
        self.assertIn("_site/", (ROOT / ".gitignore").read_text(encoding="utf-8"))
        self.assertRegex(text, r"concurrency:\n  group: sabah-bulut\n  cancel-in-progress: false")
        others = [w.name for w in WORKFLOWS if w.name != BULUT and
                  ("contents: write" in w.read_text(encoding="utf-8") or "secrets." in w.read_text(encoding="utf-8"))]
        self.assertEqual(others, [])

    def test_locked_protocol_code_unchanged(self):
        """On kayitli protokollerin kilitli kodu degismemeli (GitHub'daki integrity denetiminin ayni kontrolu).
        Kilitli dosyada degisiklik gerekiyorsa yeni bir modul yazilir (ornek: src/tool/bulut_veri.py)."""
        from src import audit
        sonuc = audit.check_locks(ROOT)
        self.assertTrue(sonuc)
        for durum, ad, mesaj in sonuc:
            self.assertEqual(durum, audit.OK, f"{ad}: {mesaj}")

    def test_chronos_requirements_pinned_and_log_merges_by_union(self):
        lines = [l.strip() for l in (ROOT / "requirements-chronos.txt").read_text(encoding="utf-8").splitlines()]
        pkgs = [l for l in lines if l and not l.startswith(("#", "--"))]
        self.assertEqual({l.split("==")[0] for l in pkgs},
                         {"torch", "chronos-forecasting", "transformers", "accelerate", "huggingface_hub"})
        for l in pkgs:
            self.assertRegex(l, r"^[\w.-]+==\d+\.\d+\.\d+$", f"surum sabit degil: {l}")
        attrs = (ROOT / ".gitattributes").read_text(encoding="utf-8")
        self.assertIn("research/tool/canli_aralik_log.csv merge=union", attrs)

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
