# Gunluk canli tahmin (LIVE_V1)
# ZAMAN: New York 14:00 cubugu kapandiktan SONRA ve AYNI New York gunu icinde calistirin.
#   Turkiye saatiyle: ABD yaz saatinde (Mart-Kasim) 21:00 sonrasi, kis saatinde 22:00 sonrasi.
#   Erken veya ertesi NY gunu calisirsa tahmin REDDEDILIR (puanlama yine yapilir).
# Kullanim (proje kokunden):
#   powershell -ExecutionPolicy Bypass -File scripts\gunluk_tahmin.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\gunluk_tahmin.ps1 -YedekKlasoru "D:\altin_yedek"
#   powershell -ExecutionPolicy Bypass -File scripts\gunluk_tahmin.ps1 -TahminYok
#
# -TahminYok: yalnizca 1) arsiv ve 2) seriler calisir (saatlik arsiv, canli panel, esanli seri).
#   Puanlama, tahmin ve git commit ATLANIR; research/live/forecast_log.csv'ye dokunulmaz.
#   Verilmezse davranis aynidir (tum adimlar).
#
# Git'e GIREN: yalnizca research/live/forecast_log.csv (kucuk, degistirilemezlik kaniti).
# Git'e GIRMEYEN (data/ .gitignore): ham anlik goruntuler ve saatlik arsiv. Yahoo saatlik veriyi
#   ~730 gun tuttugu icin bunlarin YEDEGI onemlidir: -YedekKlasoru verilirse data\raw altindaki
#   anlik goruntu klasorleri oraya SILMEDEN kopyalanir (robocopy /E, /MIR KULLANILMAZ).
# Ana data\processed\market_panel.csv ve FINAL_EXAM_V1 dosyalari DEGISTIRILMEZ.
param([string]$YedekKlasoru = "", [switch]$TahminYok)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"

function Adim([string]$ad, [string[]]$argv, [int[]]$kabul = @(0)) {
    Write-Host "`n>>> $ad" -ForegroundColor Cyan
    & $py @argv | Out-Host          # cikti ekrana; fonksiyonun donus degeri yalnizca cikis kodu
    $kod = $LASTEXITCODE
    if ($kabul -notcontains $kod) {
        Write-Host "HATA: '$ad' basarisiz (cikis $kod). Sonraki adimlar calistirilmadi." -ForegroundColor Red
        exit 1
    }
    return $kod
}

# 1) ARSIV (internet)
Adim "Gunluk seriler (GVZ dahil)" @("-m", "src.data.collect") | Out-Null
Adim "Saatlik anlik goruntu" @("-m", "src.data.collect_intraday") | Out-Null
if ($YedekKlasoru) {
    foreach ($k in @("snapshots_intraday", "snapshots")) {
        robocopy "data\raw\$k" (Join-Path $YedekKlasoru $k) /E /XO /NFL /NDL /NJH /NJS /NP | Out-Null
        if ($LASTEXITCODE -ge 8) { Write-Host "HATA: yedekleme basarisiz ($k, robocopy $LASTEXITCODE)." -ForegroundColor Red; exit 1 }
    }
    Write-Host "Yedek: $YedekKlasoru" -ForegroundColor Green
}

# 2) SERILER (canli panel ayri klasorde; ana panel degismez)
$canli = "data\processed\live"
Remove-Item "$canli\market_panel.csv", "$canli\market_panel_report.json" -ErrorAction SilentlyContinue
Adim "Canli panel (GVZ)" @("-m", "src.data.build_panel", "--write", "--processed-dir", $canli) | Out-Null
Adim "Esanli gunluk seri" @("-m", "src.data.sync_dataset") | Out-Null

if ($TahminYok) {
    Write-Host "`nSonuc: -TahminYok: arsiv ve seriler guncellendi; puanlama, tahmin ve commit ATLANDI (kayit degismedi)." -ForegroundColor Yellow
    exit 0
}

# 3) PUANLAMA, 4) TAHMIN
Adim "Puanlama" @("-m", "src.live.forecast", "score") | Out-Null
$t = Adim "Tahmin" @("-m", "src.live.forecast", "issue") @(0, 3, 4)
$durum = switch ($t) { 0 { "yeni tahmin kaydedildi" } 3 { "bu gun icin tahmin zaten vardi" } 4 { "tahmin zamani disinda; tahmin YOK" } }

# 5) COMMIT (yalnizca tahmin kaydi)
git add research/live/forecast_log.csv
if ($LASTEXITCODE -ne 0) { Write-Host "HATA: git add basarisiz." -ForegroundColor Red; exit 1 }
git diff --cached --quiet -- research/live/forecast_log.csv
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSonuc: $durum; kayitta commit edilecek degisiklik yok." -ForegroundColor Yellow
    exit 0
}
git commit -m "Canli tahmin $(Get-Date -Format 'yyyy-MM-dd'): $durum" -- research/live/forecast_log.csv
if ($LASTEXITCODE -ne 0) { Write-Host "HATA: git commit basarisiz; kayit commit EDILMEDI." -ForegroundColor Red; exit 1 }
Write-Host "`nSonuc: $durum; kayit commit edildi." -ForegroundColor Green
