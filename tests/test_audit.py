"""
Butunluk denetimi testleri: src/audit.py
Calistirma: python -m unittest tests.test_audit -v
Gecici git deposu kurulur (kilitli dosyalarin kopyalari, sentetik kayitlar). Gercek depo degistirilmez.
"""

from pathlib import Path
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import audit  # noqa: E402
from src.live import forecast as lf  # noqa: E402
from src.live import forecast_v2 as v2  # noqa: E402
from src.live import physical_forecast as lp  # noqa: E402
from src.live import physical_intervals as pi  # noqa: E402

LOCKED = sorted({f for fs in audit.LOCKS.values() for f in fs})


def rmtree(path):
    """Windows'ta git nesneleri salt okunurdur; yazilabilir yapip siler."""
    def fix(func, p, _exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onerror=fix)


def git(root, *args):
    r = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "core.autocrlf=false",
                        "-C", str(root), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout


def commit(root, msg):
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", msg)


def _row(cols, text, hash_fn, values):
    r = {c: ("x" if c in text else 1.0) for c in cols}
    r.update(values)
    r["row_hash"] = hash_fn(pd.Series(r))
    return r


class Repo(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="audit_"))
        self.addCleanup(rmtree, self.root)
        git(self.root, "init", "-q")
        for f in LOCKED:
            (self.root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(PROJECT_ROOT / f, self.root / f)
        read = lambda f: (self.root / f).read_bytes()
        self.exp = {pid: audit.lock_value(pid, read) for pid in audit.LOCKS}
        pre = self.root / "research" / "preregistration"
        pre.mkdir(parents=True)
        docs = {"live_phys_v1": {"code_md5": self.exp["LIVE_PHYS_V1"]},
                "live_phys_int_v1": {"code_md5": self.exp["LIVE_PHYS_INT_V1"]},
                "live_v1": {"forecast_code_md5": self.exp["LIVE_V1"][0], "eval_code_md5": self.exp["LIVE_V1"][1]},
                "live_v2": {"code_md5_combined": self.exp["LIVE_V2"]}}
        reg = []
        for i, (name, d) in enumerate(docs.items()):
            text = json.dumps(d, indent=2)
            (pre / f"{name}.json").write_text(text, encoding="utf-8", newline="\n")
            reg.append({"protocol_id": name.upper(), "event": "prepared", "at_utc": f"2026-10-15T0{i}:00:00+00:00",
                        "prereg_md5": hashlib.md5(text.encode()).hexdigest()})
        self.write_registry(reg)
        commit(self.root, "on kayitlar")
        self.write_logs(2)
        commit(self.root, "ilk tahminler")

    def write_registry(self, events):
        (self.root / "research" / "registry.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events),
                                                                encoding="utf-8", newline="\n")

    def write_logs(self, n, tamper_first=False):
        live = self.root / "research" / "live"
        live.mkdir(parents=True, exist_ok=True)
        days = ["2026-10-15", "2026-10-16", "2026-10-19", "2026-10-20"][:n]
        phys = [_row(lp.FORECAST_COLS, lp.TEXT, lp.row_hash, {"forecast_id": f"LIVE_PHYS_V1-{d}", "target_date": d,
                                                               "code_md5": self.exp["LIVE_PHYS_V1"]}) for d in days]
        ivl = [_row(pi.FORECAST_COLS, pi.TEXT, pi.row_hash, {"forecast_id": f"LIVE_PHYS_INT_V1-{d}", "target_date": d,
                                                              "code_md5": self.exp["LIVE_PHYS_INT_V1"]}) for d in days]
        th1 = [_row(lf.FORECAST_COLS, lf.TEXT_COLS, lf.row_hash, {"system_version": "LIVE_V1", "asof_date": d}) for d in days]
        th2 = [_row(v2.FORECAST_COLS, v2.TEXT_COLS, v2.row_hash, {"system_version": "LIVE_V2", "asof_date": d,
                                                                   "code_md5": self.exp["LIVE_V2"]}) for d in days]
        if tamper_first:
            phys[0]["pred_Bosluk"] = 999.0                                      # ozet guncellenmeden
        pd.DataFrame(phys).reindex(columns=lp.ALL_COLS).to_csv(live / "physical_log.csv", index=False)
        pd.DataFrame(ivl).reindex(columns=pi.ALL_COLS).to_csv(live / "physical_intervals_log.csv", index=False)
        pd.DataFrame(th1).reindex(columns=lf.ALL_COLS).to_csv(live / "forecast_log.csv", index=False)
        pd.DataFrame(th2).reindex(columns=v2.ALL_COLS).to_csv(live / "forecast_v2_log.csv", index=False)

    def fails(self):
        return [(n, d) for s, n, d in audit.run_all(self.root) if s == audit.FAIL]


class TestClean(Repo):
    def test_clean_repo_passes(self):
        self.write_logs(3)
        commit(self.root, "ucuncu gun")
        res = audit.run_all(self.root)
        self.assertEqual(self.fails(), [])
        names = [n for _, n, _ in res]
        self.assertIn("kayit defteri gecmisi", names)
        self.assertIn("LIVE_V2 kilit gecmisi", names)
        self.assertGreaterEqual(len(res), 25)

    def test_no_git_history_is_warning_not_error(self):
        rmtree(self.root / ".git")
        res = audit.run_all(self.root)
        self.assertIn((audit.WARN, "git gecmisi", "tam gecmis yok (sig klon); gecmis denetimi atlandi"), res)


class TestLocks(Repo):
    def test_edit_detected_and_change_back_still_detected(self):
        f = self.root / "src" / "models" / "train.py"
        orig = f.read_bytes()
        f.write_bytes(orig + b"\n# degisti\n")
        self.assertIn(("LIVE_V1 kod kilidi", "KILITLI KOD DEGISMIS"), self.fails())
        commit(self.root, "kilitli dosya degisti")
        f.write_bytes(orig)
        commit(self.root, "geri alindi")
        fails = dict(self.fails())
        self.assertNotIn("LIVE_V1 kod kilidi", fails)                         # bugun ayni
        self.assertIn("LIVE_V1 kilit gecmisi", fails)                         # ama gecmiste degismis
        self.assertIn("LIVE_V2 kilit gecmisi", fails)                         # train.py V2'de de kilitli
        self.assertNotIn("LIVE_PHYS_V1 kilit gecmisi", fails)

    def test_line_endings_do_not_break_locks(self):
        f = self.root / "src" / "live" / "physical_forecast.py"
        f.write_bytes(f.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        commit(self.root, "crlf")
        self.assertEqual(self.fails(), [])


class TestLogsAndRegistry(Repo):
    def test_tampered_row_detected(self):
        self.write_logs(2, tamper_first=True)
        self.assertTrue(any(n == "research/live/physical_log.csv" for n, _ in self.fails()))

    def test_row_made_by_other_code_detected(self):
        for rel, mod in (("physical_log.csv", lp), ("forecast_v2_log.csv", v2)):
            with self.subTest(rel):
                p = self.root / "research" / "live" / rel
                orig = p.read_bytes()
                log = mod.read_log(p)
                log.loc[0, "code_md5"] = "baska_kod"
                log.loc[0, "row_hash"] = mod.row_hash(log.loc[0])                 # ozet gecerli, kod farkli
                log.to_csv(p, index=False)
                self.assertTrue(any(n == f"research/live/{rel}" and "kilitli koddan farkli" in d for n, d in self.fails()))
                p.write_bytes(orig)

    def test_deleted_past_forecast_detected_in_history(self):
        p = self.root / "research" / "live" / "forecast_v2_log.csv"
        log = pd.read_csv(p)
        log.iloc[1:].to_csv(p, index=False)                                   # gecmis satir silindi
        commit(self.root, "satir silindi")
        self.assertIn("research/live/forecast_v2_log.csv gecmisi", dict(self.fails()))

    def test_scoring_update_is_allowed(self):
        p = self.root / "research" / "live" / "forecast_log.csv"
        log = pd.read_csv(p)
        log.loc[0, "realized_price"] = 6600.0                                 # gerceklesen eklenir, ozet ayni
        log.to_csv(p, index=False)
        commit(self.root, "puanlama")
        self.assertEqual(self.fails(), [])

    def test_registry_rewrite_and_time_order(self):
        reg = self.root / "research" / "registry.jsonl"
        lines = reg.read_text(encoding="utf-8").splitlines()
        e = json.loads(lines[0])
        e["at_utc"] = "2026-10-15T00:30:00+00:00"
        reg.write_text("\n".join([json.dumps(e)] + lines[1:]) + "\n", encoding="utf-8", newline="\n")
        commit(self.root, "eski olay degisti")
        self.assertIn("kayit defteri gecmisi", dict(self.fails()))
        reg.write_text(reg.read_text(encoding="utf-8") + json.dumps(
            {"protocol_id": "X", "event": "y", "at_utc": "2020-01-01T00:00:00+00:00"}) + "\n", encoding="utf-8", newline="\n")
        self.assertIn(("kayit defteri", "satir 5 zaman damgasi geriye gidiyor"), self.fails())

    def test_prereg_md5_accepts_crlf_rejects_change(self):
        p = self.root / "research" / "preregistration" / "live_v2.json"
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))   # her platformda tam CRLF
        self.assertNotIn("LIVE_V2 on kayit (prepared)", dict(self.fails()))
        p.write_text(p.read_text(encoding="utf-8") + " ", encoding="utf-8", newline="\n")
        self.assertIn("LIVE_V2 on kayit (prepared)", dict(self.fails()))


class TestHygiene(Repo):
    def test_tracked_data_env_db_and_secrets(self):
        (self.root / "data").mkdir()
        (self.root / "data" / "x.csv").write_text("a\n", encoding="utf-8", newline="\n")
        (self.root / ".env").write_text("A=1\n", encoding="utf-8", newline="\n")
        (self.root / "altin.sqlite").write_bytes(b"x")
        (self.root / "src" / "config.py").write_text("EVDS_API_" + 'KEY = "abcdEFGH1234ijkl"\n', encoding="utf-8", newline="\n")
        commit(self.root, "kotu dosyalar")
        details = " ".join(d for n, d in self.fails() if n == "depo hijyeni")
        for want in ("data/ altinda", ".env", "veritabani dosyasi", "acik anahtar"):
            self.assertIn(want, details)

    def test_main_exit_codes(self):
        import contextlib
        import io
        from unittest import mock
        with mock.patch.object(audit, "ROOT", self.root), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main([]), 0)
            (self.root / "src" / "models" / "adaptive.py").write_text("# bozuk\n", encoding="utf-8", newline="\n")
            self.assertEqual(audit.main([]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
