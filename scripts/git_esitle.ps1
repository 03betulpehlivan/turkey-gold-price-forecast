# Yerel commit'leri GitHub ile guvenli esitler: fetch -> gerekirse BIRLESTIR (merge) -> push.
# Rebase KULLANILMAZ: LIVE_V1 kayit satirlari uretildikleri commit'in kimligini saklar; rebase bu kimlikleri
# degistirir ve o tahminleri degerlendirme disina dusururdu. Zorla push (force) KULLANILMAZ.
# Birlestirme cakisirsa geri alinir (merge --abort) ve HATA bildirilir; yerel commit'ler kaybolmaz.
# Kullanim (proje kokunden): powershell -ExecutionPolicy Bypass -File scripts\git_esitle.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

git fetch --quiet origin main
if ($LASTEXITCODE -ne 0) { Write-Host "ESITLEME HATA: GitHub'a ulasilamadi (fetch)."; exit 1 }
$geride = [int](git rev-list --count HEAD..origin/main)
if ($geride -gt 0) {
    git -c pull.rebase=false merge --no-edit --no-ff origin/main
    if ($LASTEXITCODE -ne 0) {
        git merge --abort
        Write-Host "ESITLEME HATA: GitHub'daki $geride commit yerel commit'lerle cakisti; birlestirme geri alindi."
        exit 1
    }
}
$ileride = [int](git rev-list --count origin/main..HEAD)
if ($ileride -gt 0) {
    git push --quiet origin HEAD:main
    if ($LASTEXITCODE -ne 0) { Write-Host "ESITLEME HATA: push basarisiz ($ileride commit yerelde)."; exit 1 }
}
Write-Host "ESITLEME TAMAM: GitHub'dan $geride commit alindi, $ileride commit gonderildi."
exit 0
