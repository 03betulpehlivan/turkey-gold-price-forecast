"""
Butunluk Denetimi (salt okunur)
===============================
Kullanim (proje kokunden):  python -m src.audit
GitHub Actions her push'ta calistirir (.github/workflows/integrity.yml). Hicbir dosyayi degistirmez.

Denetlenenler:
  1. Kayit defteri (research/registry.jsonl): her satir gecerli JSON; zaman damgalari azalmiyor; kayitli
     on kayit ve sonuc ozetleri dosyalarla ayni (satir sonundan bagimsiz: LF veya CRLF).
  2. Etkin canli protokollerin kod kilitleri: LIVE_PHYS_V1, LIVE_PHYS_INT_V1, LIVE_V1, LIVE_V2.
  3. Canli kayitlar: her satirin ozeti gecerli; satirdaki kod ozeti (varsa) on kayittakiyle ayni.
  4. Git gecmisi (tam gecmis varsa): kayit defteri yalnizca SONUNA eklenmis; canli kayitlarda gecmiste
     yazilmis hicbir tahmin satiri silinmemis, sirasi veya ozeti degismemis; kilitli kod dosyalari on kayit
     commit'inden sonra hic degismemis.
  5. Depo hijyeni: data/ altinda izlenen dosya yok; .env, veritabani dosyasi, 5 MB ustu dosya yok;
     kodda acik anahtar atamasi yok.
Cikis: 0 = hata yok (uyari olabilir), 1 = en az bir HATA.
"""

from pathlib import Path
import argparse
import hashlib
import io
import json
import re
import subprocess
import sys

import pandas as pd

from src.live import forecast as lf
from src.live import forecast_eval as le
from src.live import forecast_v2 as v2
from src.live import physical_forecast as lp
from src.live import physical_intervals as pi

ROOT = Path(__file__).resolve().parents[1]
OK, WARN, FAIL = "TAMAM", "UYARI", "HATA"
LOGS = {  # yol: (anahtar sutun, okuyucu, dogrulayici)
    "research/live/physical_log.csv": ("forecast_id", lp.read_log, lp.verify),
    "research/live/physical_intervals_log.csv": ("forecast_id", pi.read_log, pi.verify),
    "research/live/forecast_log.csv": ("asof_date", lf.read_log, lf.verify_log),
    "research/live/forecast_v2_log.csv": ("asof_date", v2.read_log, v2.verify_log),
}
LOCKS = {
    "LIVE_PHYS_V1": ["src/live/physical_forecast.py"],
    "LIVE_PHYS_INT_V1": ["src/live/physical_intervals.py", "src/live/physical_forecast.py", "src/models/interval_study.py"],
    "LIVE_V1": le.FORECAST_FILES + le.EVAL_FILES,
    "LIVE_V2": v2.CODE_FILES,
}
MAX_FILE_MB = 5
SECRET_RE = re.compile(r"""(?i)(api[_-]?key|secret|password|parola|token)\s*[:=]\s*['"][A-Za-z0-9_\-]{12,}['"]""")


def _md5_variants(data: bytes) -> set:
    lf_ = data.replace(b"\r\n", b"\n")
    return {hashlib.md5(lf_).hexdigest(), hashlib.md5(lf_.replace(b"\n", b"\r\n")).hexdigest()}


def _prereg_path(root: Path, pid: str) -> Path:
    return root / "research" / "preregistration" / f"{pid.lower()}.json"


def _git(root: Path, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True)


def git_history_available(root: Path) -> bool:
    r = _git(root, "rev-parse", "--is-shallow-repository")
    return r.returncode == 0 and r.stdout.strip() == b"false"


# ---------------------------------------------------------------------------
def check_registry(root: Path) -> list:
    out, path = [], root / "research" / "registry.jsonl"
    if not path.exists():
        return [(FAIL, "kayit defteri", "research/registry.jsonl yok")]
    rows, last = [], ""
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            out.append((FAIL, "kayit defteri", f"satir {i} gecerli JSON degil"))
            continue
        if not all(k in e for k in ("protocol_id", "event", "at_utc")):
            out.append((FAIL, "kayit defteri", f"satir {i} eksik alan"))
            continue
        if e["at_utc"] < last:
            out.append((FAIL, "kayit defteri", f"satir {i} zaman damgasi geriye gidiyor"))
        last = max(last, e["at_utc"])
        rows.append(e)
    for e in rows:
        pid = e["protocol_id"]
        if "prereg_md5" in e:
            p = _prereg_path(root, pid)
            ok = p.exists() and e["prereg_md5"] in _md5_variants(p.read_bytes())
            out.append((OK if ok else FAIL, f"{pid} on kayit ({e['event']})", "ozet ayni" if ok else "ozet FARKLI veya dosya yok"))
        if "result_md5" in e:
            cand = [root / "research" / pid.lower() / "result.json", root / "research" / "live" / f"{pid.lower()}_result.json"]
            p = next((c for c in cand if c.exists()), None)
            ok = p is not None and e["result_md5"] in _md5_variants(p.read_bytes())
            out.append((OK if ok else FAIL, f"{pid} sonuc", "ozet ayni" if ok else "ozet FARKLI veya dosya yok"))
    if not out:
        out.append((OK, "kayit defteri", f"{len(rows)} olay"))
    return out


def _norm(b: bytes) -> bytes:
    return b.replace(b"\r\n", b"\n")


def lock_value(pid: str, read):
    """Protokolun kilit ozeti; read(dosya) -> bayt veya None. Dosya yoksa None."""
    files = LOCKS[pid]
    data = {f: read(f) for f in files}
    if any(v is None for v in data.values()):
        return None
    if pid == "LIVE_PHYS_V1":
        return hashlib.md5(_norm(data[files[0]])).hexdigest()
    if pid == "LIVE_PHYS_INT_V1":
        h = hashlib.md5()
        for f in files:
            h.update(_norm(data[f]))
        return h.hexdigest()
    md5s = {f: hashlib.md5(_norm(b)).hexdigest() for f, b in data.items()}
    if pid == "LIVE_V1":
        return ({f: md5s[f] for f in le.FORECAST_FILES}, {f: md5s[f] for f in le.EVAL_FILES})
    return v2.combined_md5(md5s)


def lock_expected(pid: str, prereg: dict):
    if pid == "LIVE_V1":
        return (prereg["forecast_code_md5"], prereg["eval_code_md5"])
    return prereg["code_md5_combined"] if pid == "LIVE_V2" else prereg["code_md5"]


def _read_prereg(root: Path, pid: str):
    p = _prereg_path(root, pid)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def check_locks(root: Path) -> list:
    out = []
    for pid in LOCKS:
        d = _read_prereg(root, pid)
        if d is None:
            out.append((WARN, f"{pid} kod kilidi", "on kayit yok"))
            continue
        ok = lock_value(pid, lambda f: (root / f).read_bytes() if (root / f).exists() else None) == lock_expected(pid, d)
        out.append((OK if ok else FAIL, f"{pid} kod kilidi", "kilitli dosyalar ayni" if ok else "KILITLI KOD DEGISMIS"))
    return out


def check_logs(root: Path) -> list:
    out = []
    expected = {}
    for pid, key in (("LIVE_PHYS_V1", "code_md5"), ("LIVE_PHYS_INT_V1", "code_md5"), ("LIVE_V2", "code_md5_combined")):
        p = _prereg_path(root, pid)
        if p.exists():
            expected[pid] = json.loads(p.read_text(encoding="utf-8"))[key]
    for rel, (keycol, read, verify) in LOGS.items():
        p = root / rel
        if not p.exists():
            out.append((WARN, rel, "kayit yok"))
            continue
        log = read(p)
        try:
            verify(log)
        except Exception as e:  # LogTampered turleri
            out.append((FAIL, rel, f"satir ozeti gecersiz: {e}"))
            continue
        bad = []
        if "code_md5" in log.columns and len(log):
            pid = "LIVE_V2" if "forecast_v2" in rel else ("LIVE_PHYS_INT_V1" if "intervals" in rel else "LIVE_PHYS_V1")
            want = expected.get(pid)
            if want:
                bad = [str(k) for k, c in zip(log[keycol], log["code_md5"]) if str(c) != want]
        if log[keycol].astype(str).duplicated().any():
            out.append((FAIL, rel, "ayni gun icin birden fazla tahmin"))
        elif bad:
            out.append((FAIL, rel, f"kilitli koddan farkli kodla uretilmis satir: {bad[:3]}"))
        else:
            out.append((OK, rel, f"{len(log)} satir, ozetler gecerli"))
    return out


def _versions(root: Path, rel: str) -> list:
    r = _git(root, "log", "--format=%H", "--reverse", "--", rel)
    return [h for h in r.stdout.decode().split() if h] if r.returncode == 0 else []


def _show(root: Path, commit: str, rel: str):
    r = _git(root, "show", f"{commit}:{rel}")
    return r.stdout if r.returncode == 0 else None


def check_history(root: Path) -> list:
    if not git_history_available(root):
        return [(WARN, "git gecmisi", "tam gecmis yok (sig klon); gecmis denetimi atlandi")]
    out = []
    reg = "research/registry.jsonl"
    prev, broken = b"", None
    for h in _versions(root, reg):
        cur = _show(root, h, reg) or b""
        if not cur.replace(b"\r\n", b"\n").startswith(prev.replace(b"\r\n", b"\n")):
            broken = h[:7]
        prev = cur
    out.append((FAIL, "kayit defteri gecmisi", f"{broken} commit'inde eski satir degismis") if broken
               else (OK, "kayit defteri gecmisi", "yalnizca sonuna eklenmis"))
    for rel, (keycol, _, _) in LOGS.items():
        prev, broken = [], None
        for h in _versions(root, rel):
            data = _show(root, h, rel)
            if data is None:
                continue
            df = pd.read_csv(io.BytesIO(data), dtype=str)
            cur = list(zip(df[keycol], df["row_hash"])) if {keycol, "row_hash"} <= set(df.columns) else []
            if cur[:len(prev)] != prev:
                broken = h[:7]
            prev = cur
        out.append((FAIL, f"{rel} gecmisi", f"{broken} commit'inde gecmis tahmin satiri degismis") if broken
                   else (OK, f"{rel} gecmisi", "gecmis tahminler degismemis"))
    for pid, files in LOCKS.items():
        d = _read_prereg(root, pid)
        p = f"research/preregistration/{pid.lower()}.json"
        added = _git(root, "log", "--diff-filter=A", "--format=%H", "--", p).stdout.decode().split()
        if d is None or not added:
            out.append((WARN, f"{pid} kilit gecmisi", "on kayit commit'i bulunamadi"))
            continue
        want = lock_expected(pid, d)
        later = _git(root, "log", "--format=%H", "--reverse", f"{added[-1]}..HEAD", "--", *files).stdout.decode().split()
        diff = [h[:7] for h in later if lock_value(pid, lambda f, h=h: _show(root, h, f)) != want]
        if diff:
            out.append((FAIL, f"{pid} kilit gecmisi", f"on kayittan sonra kilitli kod farkli: {diff[:3]}"))
        else:
            note = f"; {len(later)} commit dosyalara dokundu, icerik on kayitla ayni" if later else ""
            out.append((OK, f"{pid} kilit gecmisi", "on kayittan beri degismemis" + note))
    return out


def check_hygiene(root: Path) -> list:
    r = _git(root, "ls-files", "-z")
    if r.returncode != 0:
        return [(WARN, "depo hijyeni", "git yok; atlandi")]
    files = [f for f in r.stdout.decode("utf-8", "replace").split("\0") if f]
    out = []
    data = [f for f in files if f.startswith("data/")]
    env = [f for f in files if Path(f).name.startswith(".env")]
    dbs = [f for f in files if f.endswith((".sqlite", ".db", ".mdf", ".ldf"))]
    big = [f for f in files if (root / f).exists() and (root / f).stat().st_size > MAX_FILE_MB * 1024 * 1024]
    secret = []
    for f in files:
        if f.endswith((".py", ".ps1", ".yml", ".yaml", ".json", ".md", ".txt", ".html", ".js")) and (root / f).exists():
            if SECRET_RE.search((root / f).read_text(encoding="utf-8", errors="replace")):
                secret.append(f)
    for name, lst in (("data/ altinda izlenen dosya", data), (".env dosyasi", env), ("veritabani dosyasi", dbs),
                      (f"{MAX_FILE_MB} MB ustu dosya", big), ("acik anahtar atamasi", secret)):
        out.append((FAIL, "depo hijyeni", f"{name}: {lst[:3]}") if lst else (OK, "depo hijyeni", f"{name} yok"))
    return out


def run_all(root: Path = None) -> list:
    root = Path(root or ROOT)
    res = []
    for fn in (check_registry, check_locks, check_logs, check_history, check_hygiene):
        try:
            res += fn(root)
        except Exception as e:  # denetimin kendisi coktuyse bu bir HATA'dir
            res.append((FAIL, fn.__name__, f"denetim calismadi: {type(e).__name__}: {e}"))
    return res


def main(argv=None) -> int:
    argparse.ArgumentParser(description="Butunluk denetimi (salt okunur).").parse_args([] if argv is None else argv)
    res = run_all()
    print("=" * 78 + "\nBUTUNLUK DENETIMI\n" + "=" * 78)
    for s, name, detail in res:
        print(f"  [{s:5s}] {name}: {detail}")
    n_fail = sum(s == FAIL for s, _, _ in res)
    n_warn = sum(s == WARN for s, _, _ in res)
    print(f"\n  Sonuc: {len(res)} denetim, {n_fail} hata, {n_warn} uyari")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
