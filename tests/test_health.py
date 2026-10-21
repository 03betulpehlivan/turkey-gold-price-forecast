"""
Sistem sagligi (src/health.py), GitHub esitleme betigi ve sarmalayici testleri
Calistirma: python -m unittest tests.test_health -v
Gecici klasorler ve gecici git depolari. PowerShell CALISTIRILMAZ; betik metni sinanir.
"""

from pathlib import Path
from unittest import mock
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
import warnings

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore", message=".*httpx2.*")
from fastapi.testclient import TestClient  # noqa: E402

from src import health  # noqa: E402
from src.api import app as api  # noqa: E402

FRI_14 = "2026-10-16T11:00:00+00:00"        # cuma 14:00 Istanbul
OK_AUDIT = lambda root: [("TAMAM", "x", "y")] * 31


def rmtree(path):
    def fix(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onerror=fix)


def git(root, *args):
    r = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(root), *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout


def tr(s):
    return pd.Timestamp(s).tz_localize("Europe/Istanbul")


GOOD_LOG = "Baslangic\nYEDEK TAMAM\nESITLEME TAMAM\n"


class TestSchedule(unittest.TestCase):
    def test_expected_last(self):
        cases = {"2026-10-16 14:00": ("2026-10-16 13:15", "2026-10-15 23:15"),    # cuma ogleden sonra
                 "2026-10-16 23:20": ("2026-10-16 13:15", "2026-10-16 23:15"),
                 "2026-10-17 10:00": ("2026-10-16 13:15", "2026-10-16 23:15"),    # cumartesi
                 "2026-10-19 12:00": ("2026-10-16 13:15", "2026-10-16 23:15"),    # pazartesi sabah once
                 "2026-10-19 13:20": ("2026-10-19 13:15", "2026-10-16 23:15")}
        for now, (sabah, aksam) in cases.items():
            with self.subTest(now):
                self.assertEqual(health.expected_last("sabah", tr(now)), tr(sabah))
                self.assertEqual(health.expected_last("aksam", tr(now)), tr(aksam))

    def test_utc_input_converted(self):
        self.assertEqual(health.now_tr(FRI_14).hour, 14)


class TestCollect(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="hl_"))
        self.addCleanup(rmtree, self.root)
        live = self.root / "research" / "live"
        live.mkdir(parents=True)
        pd.DataFrame({"target_date": ["2026-10-15", "2026-10-16"]}).to_csv(live / "physical_log.csv", index=False)
        for n in ("forecast_log.csv", "forecast_v2_log.csv"):
            pd.DataFrame({"asof_date": ["2026-10-15"]}).to_csv(live / n, index=False)

    def run_(self, task="sabah", kod=0, log=GOOD_LOG, now=FRI_14, audit=OK_AUDIT):
        e = health.collect(task, kod, log, root=self.root, now=now, audit_fn=audit)
        return e["overall"], {c["name"]: c["status"] for c in e["checks"]}, e

    def test_all_good(self):
        overall, m, e = self.run_()
        self.assertEqual(overall, "TAMAM", e["checks"])
        self.assertEqual(set(m), {"gorev", "yedek", "github", "tahmin", "butunluk"})

    def test_exit_code_and_push_failure(self):
        self.assertEqual(self.run_(kod=1)[1]["gorev"], "HATA")
        log = "HATA: git push basarisiz; kayit yerelde commit edildi\n" + GOOD_LOG
        self.assertEqual(self.run_(kod=1, log=log)[1]["gorev"], "UYARI")              # esitleme gonderdi
        self.assertEqual(self.run_(kod=1, log=log.replace("ESITLEME TAMAM", "ESITLEME HATA: x"))[1]["gorev"], "HATA")

    def test_warnings_backup_sync_forecast_audit(self):
        o, m, e = self.run_(log=GOOD_LOG + "UYARI: LIVE_V2 adimi basarisiz (LIVE_V1 etkilenmedi).\n")
        self.assertEqual((o, m["gunluk"]), ("UYARI", "UYARI"))
        self.assertEqual(self.run_(log="ESITLEME TAMAM\n")[1]["yedek"], "UYARI")
        self.assertEqual(self.run_(log="YEDEK TAMAM\nESITLEME HATA: ayrinti gunlukte\n")[0], "HATA")
        o, m, e = self.run_(task="aksam", now="2026-10-16T19:40:00+00:00")                 # 22:40, V1/V2 15.10'da
        self.assertEqual((o, m["tahmin"]), ("UYARI", "UYARI"))
        self.assertNotIn("tahmin", self.run_(now="2026-10-17T09:00:00+00:00")[1])         # cumartesi
        self.assertEqual(self.run_(audit=lambda r: [("HATA", "a", "b")])[1]["butunluk"], "HATA")

        def boom(root):
            raise RuntimeError("x")
        self.assertEqual(self.run_(audit=boom)[1]["butunluk"], "HATA")

    def test_unpushed_commits_are_error(self):
        remote = Path(tempfile.mkdtemp(prefix="hl_r_"))
        self.addCleanup(rmtree, remote)
        git(remote, "init", "-q", "--bare")
        git(self.root, "init", "-q", "-b", "main")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "a")
        git(self.root, "remote", "add", "origin", str(remote))
        git(self.root, "push", "-q", "-u", "origin", "main")
        self.assertEqual(self.run_()[1]["github"], "TAMAM")
        (self.root / "x.txt").write_text("x", encoding="utf-8")
        git(self.root, "add", "-A")
        git(self.root, "commit", "-q", "-m", "b")
        o, m, e = self.run_()
        self.assertEqual((o, m["github"]), ("HATA", "HATA"))
        self.assertIn("1 commit", [c for c in e["checks"] if c["name"] == "github"][0]["detail"])


class TestStatus(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hs_"))
        self.addCleanup(rmtree, self.tmp)
        self.path = self.tmp / "status.json"

    def entry(self, task, at, overall="TAMAM"):
        return {"task": task, "at": tr(at).isoformat(), "kod": 0, "overall": overall,
                "checks": [{"status": overall, "name": "gorev", "detail": "d"}]}

    def test_missing_fresh_stale(self):
        s = health.system_status(FRI_14, self.path)
        self.assertEqual(s["overall"], "UYARI")
        health.write(self.entry("sabah", "2026-10-16 12:41"), self.path)
        health.write(self.entry("aksam", "2026-10-15 22:31"), self.path)
        s = health.system_status(FRI_14, self.path)
        self.assertEqual(s["overall"], "TAMAM", s)
        s = health.system_status("2026-10-19T10:30:00+00:00", self.path)              # pazartesi 13:30
        self.assertEqual(s["tasks"]["sabah"]["overall"], "HATA")
        self.assertEqual(s["tasks"]["sabah"]["checks"][0]["name"], "zaman")
        self.assertEqual(s["tasks"]["aksam"]["overall"], "HATA")                       # cuma aksami yok
        self.assertEqual(s["overall"], "HATA")

    def test_task_result_propagates_and_write_keeps_other_task(self):
        health.write(self.entry("sabah", "2026-10-16 12:41", "UYARI"), self.path)
        health.write(self.entry("aksam", "2026-10-15 22:31"), self.path)
        health.write(self.entry("sabah", "2026-10-16 13:31"), self.path)
        d = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(set(d), {"sabah", "aksam"})
        self.assertTrue(d["sabah"]["at"].startswith("2026-10-16T13:31"))
        self.assertEqual(health.system_status(FRI_14, self.path)["overall"], "TAMAM")

    def test_api_endpoint(self):
        health.write(self.entry("sabah", "2026-10-16 12:41"), self.path)
        with mock.patch.object(health, "STATUS_PATH", self.path):
            r = TestClient(api.app, base_url="http://127.0.0.1").get("/api/system")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(set(r.json()["tasks"]), {"sabah", "aksam"})

    def test_main_write_reads_log_and_durum(self):
        log = self.tmp / "sabah.log"
        log.write_text("HATA: git push basarisiz\n", encoding="utf-8")
        durum = self.tmp / "sabah.log.durum"
        durum.write_bytes("\ufeffYEDEK TAMAM\nESITLEME TAMAM\n".encode("utf-8"))           # PowerShell BOM'u
        with mock.patch.object(health, "STATUS_PATH", self.path), \
                mock.patch.object(health, "HEARTBEAT_PATH", self.tmp / "heartbeat.json"), \
                mock.patch.object(health, "_audit", OK_AUDIT), mock.patch.object(health, "ROOT", self.tmp), \
                mock.patch.object(health, "check_forecast", lambda root, task, today: []):    # haftanin gununden bagimsiz
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                health.main(["write", "--gorev", "sabah", "--kod", "1", "--log", str(log), "--durum", str(durum)])
        hb = json.loads((self.tmp / "heartbeat.json").read_text(encoding="utf-8"))
        self.assertEqual(set(hb), {"sabah"})
        self.assertEqual(hb["sabah"]["overall"], "UYARI")
        self.assertEqual(hb["sabah"]["problems"], ["gorev"])
        e = json.loads(self.path.read_text(encoding="utf-8"))["sabah"]
        m = {c["name"]: c["status"] for c in e["checks"]}
        self.assertEqual((m["gorev"], m["yedek"]), ("UYARI", "TAMAM"))


class TestHeartbeat(unittest.TestCase):
    def test_heartbeat_keeps_both_tasks_and_no_details(self):
        tmp = Path(tempfile.mkdtemp(prefix="hb_"))
        self.addCleanup(rmtree, tmp)
        p = tmp / "heartbeat.json"
        e = {"task": "sabah", "at": "2026-10-19T12:41:00+03:00", "kod": 0, "overall": "UYARI",
             "checks": [{"status": "UYARI", "name": "tahmin", "detail": "GIZLI_AYRINTI"},
                        {"status": "TAMAM", "name": "gorev", "detail": "x"}]}
        health.write_heartbeat(e, p)
        health.write_heartbeat(dict(e, task="aksam", overall="TAMAM", checks=[]), p)
        hb = json.loads(p.read_text(encoding="utf-8"))
        self.assertEqual(set(hb), {"sabah", "aksam"})
        self.assertEqual(hb["sabah"]["problems"], ["tahmin"])
        self.assertNotIn("GIZLI_AYRINTI", p.read_text(encoding="utf-8"))
        self.assertNotIn(b"\r\n", p.read_bytes())


class TestScripts(unittest.TestCase):
    def setUp(self):
        self.sync = (PROJECT_ROOT / "scripts" / "git_esitle.ps1").read_text(encoding="utf-8")
        self.wrap = (PROJECT_ROOT / "scripts" / "zamanli_calistir.ps1").read_text(encoding="utf-8")

    def code(self, text):
        return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))

    def test_sync_merges_never_rebases_or_forces(self):
        c = self.code(self.sync)
        c.encode("ascii")
        for bad in ("rebase", "--force", " -f ", "reset --hard", "push -f", "+HEAD"):
            self.assertNotIn(bad, c.replace("pull.rebase=false", ""))
        self.assertLess(c.index("git fetch"), c.index("merge --no-edit"))
        self.assertLess(c.index("merge --no-edit"), c.index("git push"))
        self.assertIn("git merge --abort", c)
        self.assertRegex(c, r"git merge --abort\s*\n\s*Write-Host [^\n]*\n\s*exit 1")
        self.assertIn("ESITLEME TAMAM", c)
        self.assertTrue(c.rstrip().endswith("exit 0"))

    def test_wrapper_order_and_exit_code(self):
        w = self.code(self.wrap)
        order = ['& ".\\$betik"', "teorik_v2.ps1", "src.db.store load", "yedekle.ps1", "git_esitle.ps1",
                 "src.health write", 'Write-Host "Bitis']
        pos = [w.index(s) for s in order]
        self.assertEqual(pos, sorted(pos))
        tail = w[w.index("$kod = $LASTEXITCODE") + 20:w.index('Write-Host "Bitis')]
        self.assertNotRegex(tail, r"\$kod\s*=")
        self.assertNotIn("*>&1", w)                                                 # PS 5.1 stderr tuzagi
        self.assertIn("--kod $kod --log $log --durum $durum", w)
        self.assertIn('Not "YEDEK TAMAM"', w)
        self.assertIn('Not "ESITLEME TAMAM"', w)

    def test_heartbeat_committed_alone_after_status(self):
        w = self.code(self.wrap)
        i = w.index('$kalp = "research/live/heartbeat.json"')
        self.assertLess(w.index("src.health write"), i)
        self.assertLess(i, w.index('Write-Host "Bitis'))
        block = w[i:w.index('Write-Host "Bitis')]
        self.assertIn('git commit -q -m "Kalp atisi ($Gorev)" -- $kalp', block)
        self.assertIn("git_esitle.ps1", block)
        self.assertNotRegex(block, r"git add (-A|\.|--all)")
        self.assertNotRegex(block, r"\$kod\s*=")
        self.assertIn("catch {", block)

    def test_dashboard_shows_system_status(self):
        html = (PROJECT_ROOT / "src" / "web" / "detay.html").read_text(encoding="utf-8")
        js = (PROJECT_ROOT / "src" / "web" / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="sistem"', html)
        self.assertIn('al("/api/system")', js)
        self.assertNotIn("innerHTML", js)


_REAL = [PROJECT_ROOT / "research", PROJECT_ROOT / "data"]
_BEFORE = {}


def _h(folder):
    return {str(p): hashlib.md5(p.read_bytes()).hexdigest() for p in folder.rglob("*") if p.is_file()} if folder.exists() else {}


def setUpModule():
    for d in _REAL:
        _BEFORE[d] = _h(d)


def tearDownModule():
    for d in _REAL:
        if _h(d) != _BEFORE[d]:
            raise AssertionError(f"Testler gercek klasoru degistirdi: {d}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
