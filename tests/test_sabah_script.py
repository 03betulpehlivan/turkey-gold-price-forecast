"""
Sabah fiziki betigi yapisal testleri: scripts/sabah_fiziki.ps1
Calistirma: python -m unittest tests.test_sabah_script -v
PowerShell CALISTIRILMAZ; betigin calistirilabilir satirlari sirayla okunur (test_gunluk_script ile ayni yontem).
"""

from pathlib import Path
import re
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "sabah_fiziki.ps1"


def lines():
    return [l.strip() for l in SCRIPT.read_text(encoding="utf-8").splitlines() if l.strip() and not l.strip().startswith("#")]


def steps():
    out = []
    for l in lines():
        m = re.search(r'Adim "[^"]+" @\(([^)]*)\)(?: @\(([^)]*)\))?', l)
        if m and not l.startswith("function"):
            argv = [a or f"${v}" for a, v in re.findall(r'"([^"]*)"|\$(\w+)', m.group(1))]
            out.append((argv[1], argv[2:] , m.group(2)))
    return out


class TestSabah(unittest.TestCase):

    def test_order(self):
        mods = [(m, a[0] if a else None) for m, a, _ in steps()]
        self.assertEqual(mods, [("src.data.collect", None), ("src.data.build_panel", "--write"),
                                ("src.data.collect_intraday", None), ("src.live.physical_forecast", "refresh"),
                                ("src.live.physical_forecast", "score"), ("src.live.physical_forecast", "issue"),
                                ("src.live.physical_forecast", "card"), ("src.live.physical_intervals", "score"),
                                ("src.live.physical_intervals", "issue"), ("src.live.physical_intervals", "card")])

    def test_refusal_is_not_failure_but_errors_are(self):
        issues = [s for s in steps() if s[1][:1] == ["issue"]]
        self.assertEqual(len(issues), 2)
        for issue in issues:
            self.assertEqual(issue[2].replace(" ", ""), "0,3")
        text = SCRIPT.read_text(encoding="utf-8")
        body = text[text.index("function Adim"):text.index("# 1) VERI")]
        self.assertIn("$kabul -notcontains $kod", body)
        self.assertIn("exit 1", body)

    def test_key_required_and_never_written(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("if (-not $env:EVDS_API_KEY)", text)
        self.assertLess(text.index("if (-not $env:EVDS_API_KEY)"), text.index('Adim "Gunluk seriler'))
        self.assertNotRegex("\n".join(lines()), r"EVDS_API_KEY\s*=")

    def test_only_log_committed_and_main_files_untouched(self):
        gits = [l for l in lines() if l.startswith("git ") and l != "git push"]
        self.assertTrue(gits and all("research/live/physical_log.csv" in g and "research/live/physical_intervals_log.csv" in g
                                     for g in gits))
        panel = [a for m, a, _ in steps() if m == "src.data.build_panel"][0]
        self.assertEqual(panel[panel.index("--processed-dir") + 1], "$canli")
        for l in lines():
            if l.startswith("Remove-Item"):
                self.assertIn("$canli", l)
            self.assertNotIn("physical\\physical_gold_daily.csv", l.replace("Write-Host", ""))

    def test_push_after_commit_and_checked(self):
        ls = lines()
        c = next(k for k, l in enumerate(ls) if l.startswith("git commit"))
        p = ls.index("git push")
        self.assertGreater(p, c)
        self.assertIn("$LASTEXITCODE -ne 0", ls[p + 1])
        self.assertIn("exit 1", ls[p + 1])

    def test_commit_failure_reported(self):
        ls = lines()
        i = next(k for k, l in enumerate(ls) if l.startswith("git commit"))
        self.assertIn("$LASTEXITCODE -ne 0", ls[i + 1])
        self.assertIn("exit 1", ls[i + 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
