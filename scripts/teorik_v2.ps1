# Canli tahmin, GLD tabanli (LIVE_V2). Aksam gorevinde gunluk_tahmin.ps1'den SONRA calisir:
#   esanli seri (data/processed/sync/sync_daily.csv) ve canli panel o betikte guncellenir.
# Kullanim (proje kokunden): powershell -ExecutionPolicy Bypass -File scripts\teorik_v2.ps1
# Git'e GIREN: yalnizca research/live/forecast_v2_log.csv (commit; push sabah gorevinde yapilir).
# LIVE_V1 kaydina ve kilitli dosyalara dokunulmaz.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
$kayit = "research/live/forecast_v2_log.csv"

function Adim([string]$ad, [string[]]$argv, [int[]]$kabul = @(0)) {
    Write-Host "`n>>> $ad" -ForegroundColor Cyan
    & $py @argv | Out-Host
    $kod = $LASTEXITCODE
    if ($kabul -notcontains $kod) {
        Write-Host "HATA: '$ad' basarisiz (cikis $kod). Sonraki adimlar calistirilmadi." -ForegroundColor Red
        exit 1
    }
    return $kod
}

Adim "GLD saatlik" @("-m", "src.live.forecast_v2", "fetch") | Out-Null
Adim "V2 puanlama" @("-m", "src.live.forecast_v2", "score") | Out-Null
$t = Adim "V2 tahmin" @("-m", "src.live.forecast_v2", "issue") @(0, 3, 4)
$durum = switch ($t) { 0 { "yeni tahmin kaydedildi" } 3 { "bu gun icin tahmin zaten vardi" } 4 { "tahmin zamani disinda; tahmin YOK" } }

git add $kayit
if ($LASTEXITCODE -ne 0) { Write-Host "HATA: git add basarisiz." -ForegroundColor Red; exit 1 }
git diff --cached --quiet -- $kayit
if ($LASTEXITCODE -eq 0) { Write-Host "`nSonuc (V2): $durum; commit edilecek degisiklik yok." -ForegroundColor Yellow; exit 0 }
git commit -m "Canli tahmin V2 $(Get-Date -Format 'yyyy-MM-dd'): $durum" -- $kayit
if ($LASTEXITCODE -ne 0) { Write-Host "HATA: git commit basarisiz." -ForegroundColor Red; exit 1 }
Write-Host "`nSonuc (V2): $durum; kayit commit edildi." -ForegroundColor Green
