"""
Gunluk betik yapisal testleri: scripts/gunluk_tahmin.ps1
Calistirma: python -m unittest tests.test_gunluk_script -v

PowerShell bu testlerde CALISTIRILMAZ. Betigin calistirilabilir satirlari sirayla okunur ve
iki modda (normal / -TahminYok) hangi adimlarin hangi sirayla calisacagi cikarilir:
  Adim "<ad>" @("-m", "<modul>", ...)   -> python adimi
  git ...                               -> git komutu
  if ($TahminYok) { ... exit N }        -> -TahminYok modunda erken cikis
  Remove-Item ...                       -> silme
Bu bir yapisal dogrulamadir; gercek PowerShell davranisinin yerini tutmaz.
"""

from pathlib import Path
import re
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "gunluk_tahmin.ps1"


def code_lines() -> list:
    out = []
    for raw in SCRIPT.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def simulate(tahmin_yok: bool) -> dict:
    """Ust duzey akisi izler: fonksiyon govdesi atlanir, if ($TahminYok) blogu moda gore islenir."""
    lines = code_lines()
    actions, i, exit_code = [], 0, None
    while i < len(lines):
        line = lines[i]
        if line.startswith("function "):                       # fonksiyon govdesini atla
            depth = line.count("{") - line.count("}")
            i += 1
            while depth > 0:
                depth += lines[i].count("{") - lines[i].count("}")
                i += 1
            continue
        if line.startswith("if ($TahminYok)"):
            block, depth = [], line.count("{") - line.count("}")
            i += 1
            while depth > 0:
                depth += lines[i].count("{") - lines[i].count("}")
                if depth > 0:
                    block.append(lines[i])
                i += 1
            if tahmin_yok:
                for b in block:
                    if b.startswith("git ") or "Adim " in b:
                        actions.append(("YASAK_BLOK_ICI", b))
                    m = re.match(r"exit (\d+)", b)
                    if m:
                        exit_code = int(m.group(1))
                        return {"actions": actions, "exit": exit_code}
            continue
        m = re.search(r'Adim "([^"]+)" @\(([^)]*)\)', line)
        if m and not line.startswith("function"):
            args = re.findall(r'"([^"]*)"|\$(\w+)', m.group(2))
            argv = [a or f"${v}" for a, v in args]
            actions.append(("py", argv[1] if len(argv) > 1 else "", argv))
        elif line.startswith("git "):
            actions.append(("git", line))
        elif line.startswith("Remove-Item"):
            actions.append(("rm", line))
        i += 1
    return {"actions": actions, "exit": exit_code}


def modules(sim) -> list:
    return [a[1] for a in sim["actions"] if a[0] == "py"]


class TestGunlukScript(unittest.TestCase):

    def test_parameter_declared(self):
        self.assertTrue(any(re.search(r"param\(.*\[switch\]\$TahminYok", l) for l in code_lines()))

    def test_tahmin_yok_runs_only_archive_and_series(self):
        sim = simulate(True)
        self.assertEqual(modules(sim), ["src.data.collect", "src.data.collect_intraday",
                                        "src.data.build_panel", "src.data.sync_dataset"])
        self.assertEqual(sim["exit"], 0)
        self.assertFalse([a for a in sim["actions"] if a[0] in ("git", "YASAK_BLOK_ICI")])
        self.assertFalse([a for a in sim["actions"] if a[0] == "py" and a[1] == "src.live.forecast"])

    def test_normal_mode_unchanged_order(self):
        sim = simulate(False)
        self.assertEqual(modules(sim), ["src.data.collect", "src.data.collect_intraday", "src.data.build_panel",
                                        "src.data.sync_dataset", "src.live.forecast", "src.live.forecast"])
        live = [a[2][2] for a in sim["actions"] if a[0] == "py" and a[1] == "src.live.forecast"]
        self.assertEqual(live, ["score", "issue"])
        gits = [a[1] for a in sim["actions"] if a[0] == "git"]
        self.assertTrue(gits and all("research/live/forecast_log.csv" in g for g in gits))
        self.assertTrue(any(g.startswith("git commit") for g in gits))

    def test_no_step_touches_main_panel_or_exam_results(self):
        for mode in (True, False):
            sim = simulate(mode)
            bp = [a[2] for a in sim["actions"] if a[0] == "py" and a[1] == "src.data.build_panel"][0]
            self.assertIn("--processed-dir", bp)
            self.assertEqual(bp[bp.index("--processed-dir") + 1], "$canli")
            for a in sim["actions"]:
                text = " ".join(map(str, a))
                self.assertNotIn("final_exam", text)
                if a[0] == "rm":
                    self.assertIn("$canli", text)
        self.assertTrue(any(l == '$canli = "data\\processed\\live"' for l in code_lines()))
        executable = [l for l in code_lines() if not l.startswith("Write-Host")]
        self.assertFalse([l for l in executable if "processed\\market_panel.csv" in l
                          and "$canli" not in l])

    def test_every_python_step_stops_on_failure(self):
        text = SCRIPT.read_text(encoding="utf-8")
        body = text[text.index("function Adim"):text.index("# 1) ARSIV")]
        self.assertIn("$kabul -notcontains $kod", body)
        self.assertIn("exit 1", body)
        bare = [l for l in code_lines() if "& $py" in l and "@argv" not in l]
        self.assertEqual(bare, [])                                         # python yalnizca Adim icinden
        self.assertIn("$ErrorActionPreference = \"Stop\"", text)

    def test_commit_failure_is_not_reported_as_success(self):
        lines = code_lines()
        i = next(k for k, l in enumerate(lines) if l.startswith("git commit"))
        self.assertIn("$LASTEXITCODE -ne 0", lines[i + 1])
        self.assertIn("exit 1", lines[i + 1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
