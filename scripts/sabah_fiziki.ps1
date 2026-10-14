# Canli sabah fiziki altin tahmini (LIVE_PHYS_V1)
# ZAMAN: hedef gunun sabahi, 08:00-14:00 TSI (05:00-11:00 UTC). Disinda tahmin REDDEDILIR.
# On kosullar (bir kez):
#   1) python -m src.live.physical_forecast prepare  ve on kaydi commit edin
#   2) EVDS anahtarini KULLANICI ortam degiskeni yapin (Git'e girmez):
#        setx EVDS_API_KEY "anahtariniz"     (yeni PowerShell penceresinde gecerli olur)
# Kullanim (proje kokunden):
#   powershell -ExecutionPolicy Bypass -File scripts\sabah_fiziki.ps1
# Git'e GIREN: yalnizca research/live/physical_log.csv
# Ana data\processed\physical\physical_gold_daily.csv ve market_panel.csv DEGISTIRILMEZ.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"

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

if (-not $env:EVDS_API_KEY) {
    Write-Host "HATA: EVDS_API_KEY tanimli degil (setx EVDS_API_KEY ... ve yeni pencere)." -ForegroundColor Red
    exit 1
}

# 1) VERI (internet)
Adim "Gunluk seriler (ons)" @("-m", "src.data.collect") | Out-Null
$canli = "data\processed\live"
Remove-Item "$canli\market_panel.csv", "$canli\market_panel_report.json" -ErrorAction SilentlyContinue
Adim "Canli panel" @("-m", "src.data.build_panel", "--write", "--processed-dir", $canli) | Out-Null
Adim "Saatlik kur" @("-m", "src.data.collect_intraday") | Out-Null
Adim "EVDS fiziki seri (canli kopya)" @("-m", "src.live.physical_forecast", "refresh") | Out-Null

# 2) PUANLAMA, 3) TAHMIN
Adim "Puanlama" @("-m", "src.live.physical_forecast", "score") | Out-Null
$t = Adim "Tahmin" @("-m", "src.live.physical_forecast", "issue") @(0, 3)
Adim "Karne" @("-m", "src.live.physical_forecast", "card") | Out-Null
$durum = switch ($t) { 0 { "yeni tahmin kaydedildi" } 3 { "tahmin REDDEDILDI (kural geregi; ustteki nedene bakin)" } }

# 4) COMMIT (yalnizca kayit)
git add research/live/physical_log.csv
if ($LASTEXITCODE -ne 0) { Write-Host "HATA: git add basarisiz." -ForegroundColor Red; exit 1 }
git diff --cached --quiet -- research/live/physical_log.csv
if ($LASTEXITCODE -eq 0) { Write-Host "`nSonuc: $durum; kayitta commit edilecek degisiklik yok." -ForegroundColor Yellow; exit 0 }
git commit -m "Sabah fiziki tahmin $(Get-Date -Format 'yyyy-MM-dd'): $durum" -- research/live/physical_log.csv
if ($LASTEXITCODE -ne 0) { Write-Host "HATA: git commit basarisiz; kayit commit EDILMEDI." -ForegroundColor Red; exit 1 }
Write-Host "`nSonuc: $durum; kayit commit edildi." -ForegroundColor Green
