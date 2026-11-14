"""
src/tool/kapanis.py ve src/audit.check_closures testleri. Gecici klasorde calisir; gercek kayit defteri degismez.
"""
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil
import sys
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import audit  # noqa: E402
from src.tool import kapanis as kp  # noqa: E402

NOW = datetime(2026, 11, 14, 20, 0, tzinfo=timezone.utc)


class TestKapanis(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="kp_"))
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        (self.root / "research" / "live").mkdir(parents=True)
        for rel, _ in kp.DENEYLER.values():
            (self.root / rel).write_text("issued_at_utc,x\r\n2026-11-12T10:00:00+00:00,1\r\n2026-11-13T10:00:00+00:00,2\r\n",
                                         encoding="utf-8", newline="")
        self.reg = self.root / "research" / "registry.jsonl"
        self.reg.write_text(json.dumps({"protocol_id": "LIVE_V1", "event": "prepared", "at_utc": "2026-10-10T00:00:00+00:00"}) + "\n",
                            encoding="utf-8")

    def kapat(self):
        ev = kp.plan(self.root, NOW)
        kp.yaz(ev, self.root)
        return ev

    def test_plan_and_write_appends_only(self):
        once = self.reg.read_bytes()
        ev = self.kapat()
        self.assertEqual([e["protocol_id"] for e in ev], list(kp.DENEYLER))
        self.assertTrue(all(e["event"] == "closed" and e["evaluated"] is False and e["n_records"] == 2 for e in ev))
        self.assertEqual(ev[0]["last_issued_utc"], "2026-11-13T10:00:00+00:00")
        self.assertTrue(self.reg.read_bytes().startswith(once))                       # yalnizca sona eklenir
        self.assertEqual(kp.plan(self.root, NOW), [])                                  # ikinci kez kapanis yok

    def test_audit_detects_any_change_after_closure(self):
        self.kapat()
        sonuc = {n: s for s, n, _ in audit.check_closures(self.root)}
        self.assertEqual(set(sonuc.values()), {audit.OK})
        rel = kp.DENEYLER["LIVE_PHYS_V1"][0]
        with open(self.root / rel, "a", encoding="utf-8", newline="") as f:
            f.write("2026-11-16T10:00:00+00:00,3\r\n")                                 # kapanistan sonra yeni kayit
        sonuc = {n: (s, d) for s, n, d in audit.check_closures(self.root)}
        self.assertEqual(sonuc["LIVE_PHYS_V1 kapanis"][0], audit.FAIL)
        self.assertEqual(sonuc["LIVE_V1 kapanis"][0], audit.OK)

    def test_line_endings_do_not_matter(self):
        self.kapat()
        rel = kp.DENEYLER["LIVE_V2"][0]
        p = self.root / rel
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))                        # Windows/Linux farki
        self.assertEqual({s for s, _, _ in audit.check_closures(self.root)}, {audit.OK})

    def test_event_after_closure_detected(self):
        self.kapat()
        with open(self.reg, "a", encoding="utf-8") as f:
            f.write(json.dumps({"protocol_id": "LIVE_V2", "event": "evaluated", "at_utc": "2026-11-20T00:00:00+00:00"}) + "\n")
        sonuc = {n: (s, d) for s, n, d in audit.check_closures(self.root)}
        self.assertEqual(sonuc["LIVE_V2 kapanis"][0], audit.FAIL)


if __name__ == "__main__":
    unittest.main()
