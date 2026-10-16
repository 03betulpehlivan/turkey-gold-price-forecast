"""
Veri yedegi yapisal testleri: scripts/yedekle.ps1 ve scripts/zamanli_calistir.ps1 icindeki cagri
Calistirma: python -m unittest tests.test_yedekle -v
PowerShell CALISTIRILMAZ; betik metni sinanir.
"""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
BACKUP = (ROOT / "scripts" / "yedekle.ps1").read_text(encoding="utf-8")
WRAP = (ROOT / "scripts" / "zamanli_calistir.ps1").read_text(encoding="utf-8")


def code_lines(text):
    return [l for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]


class TestBackupScript(unittest.TestCase):

    def test_ascii(self):
        BACKUP.encode("ascii")

    def test_never_deletes_or_moves(self):
        code = "\n".join(code_lines(BACKUP)).upper()
        for bad in ("/MIR", "/PURGE", "/MOV", "/MOVE", "REMOVE-ITEM", " DEL ", "RMDIR"):
            self.assertNotIn(bad, code)
        robo = [l for l in code_lines(BACKUP) if l.strip().startswith("robocopy")]
        self.assertEqual(len(robo), 1)
        self.assertIn("/E", robo[0])
        self.assertIn("/XO", robo[0])

    def test_copies_irreplaceable_folders(self):
        folders = re.search(r'foreach \(\$k in @\(([^)]*)\)\)', BACKUP).group(1)
        self.assertEqual(re.findall(r'"([^"]+)"', folders), ["data\\raw", "data\\processed", "logs"])

    def test_robocopy_exit_code_threshold(self):
        self.assertIn("$rc = $LASTEXITCODE", BACKUP)
        self.assertRegex(BACKUP, r"if \(\$rc -ge 8\)")
        self.assertTrue(BACKUP.rstrip().endswith("exit $hata"))

    def test_target_required_and_outside_repo(self):
        self.assertIn("param([string]$Hedef = $env:ALTIN_YEDEK)", BACKUP)
        self.assertLess(BACKUP.index("if (-not $Hedef)"), BACKUP.index("    robocopy $k"))
        self.assertLess(BACKUP.index("StartsWith($kokTam"), BACKUP.index("    robocopy $k"))
        self.assertIn("OrdinalIgnoreCase", BACKUP)


class TestWrapperCall(unittest.TestCase):

    def test_exit_code_captured_before_backup(self):
        self.assertLess(WRAP.index("$kod = $LASTEXITCODE"), WRAP.index("yedekle.ps1"))

    def test_backup_cannot_change_exit_code(self):
        start = WRAP.index("yedekle.ps1")
        end = WRAP.index("Write-Host \"Bitis")
        block = WRAP[start:end]
        self.assertNotRegex(block, r"\$kod\s*=")             # okunabilir (sistem durumu), atanamaz
        self.assertNotIn("exit", block)
        self.assertIn("catch {", block)                       # yedek istisnasi dis catch'e ulasmaz

    def test_backup_optional_with_warning(self):
        self.assertIn("if ($env:ALTIN_YEDEK)", WRAP)
        self.assertIn("ALTIN_YEDEK tanimli degil", WRAP)


if __name__ == "__main__":
    unittest.main(verbosity=2)
