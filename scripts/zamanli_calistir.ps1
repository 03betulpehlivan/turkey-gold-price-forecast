# Zamanlanmis calistirma sarmalayicisi (Gorev Zamanlayici bunu cagirir)
# Kullanim: powershell -NoProfile -ExecutionPolicy Bypass -File scripts\zamanli_calistir.ps1 -Gorev sabah|aksam
# Tum cikti logs\<gorev>_<tarih_saat>.log dosyasina yazilir (logs\ Git'e girmez).
param([Parameter(Mandatory = $true)][ValidateSet("sabah", "aksam")][string]$Gorev)
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root
$betik = @{ "sabah" = "scripts\sabah_fiziki.ps1"; "aksam" = "scripts\gunluk_tahmin.ps1" }[$Gorev]
New-Item -ItemType Directory -Force -Path "logs" | Out-Null
$log = "logs\{0}_{1}.log" -f $Gorev, (Get-Date -Format "yyyyMMdd_HHmm")
Start-Transcript -Path $log -Append | Out-Null
$kod = 1
try {
    Write-Host "Baslangic: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | gorev: $Gorev | betik: $betik"
    if (-not $env:EVDS_API_KEY -and $Gorev -eq "sabah") { Write-Host "UYARI: EVDS_API_KEY tanimli degil." }
    & ".\$betik"
    $kod = $LASTEXITCODE
    # Sonraki adimlarin SONUCLARI (cikis kodlarina gore) ayrica $durum dosyasina yazilir; sistem durumu
    # bunu okur. Cikti yakalanmaz (*>&1 kullanilmaz: PowerShell 5.1'de stderr'i hataya cevirebilir).
    # Bu adimlarin HICBIRI cikis kodunu degistirmez.
    $durum = "$log.durum"
    function Not([string]$m) { Write-Host $m; Add-Content -Path $durum -Value $m -Encoding UTF8 }
    # LIVE_V2 (GLD tabanli teorik): yalnizca aksam; LIVE_V1 etkilenmez.
    if ($Gorev -eq "aksam") {
        try {
            & ".\scripts\teorik_v2.ps1"
            if ($LASTEXITCODE -ne 0) { Not "UYARI: LIVE_V2 adimi basarisiz (LIVE_V1 etkilenmedi)." }
        }
        catch { Not "UYARI: LIVE_V2 hatasi (LIVE_V1 etkilenmedi): $_" }
    }
    # Ileriye donuk karne: 1/3/5 gun araliklari, fiziki kapanistan ONCE kaydedilir (yalnizca sabah).
    if ($Gorev -eq "sabah") {
        try {
            & ".\.venv\Scripts\python.exe" -m src.tool.canli_karne kaydet
            if ($LASTEXITCODE -ne 0) { Not "UYARI: ileriye donuk karne kaydi yapilamadi (diger kayitlar etkilenmedi)." }
            $ck = "research/tool/canli_aralik_log.csv"
            if (Test-Path $ck) {
                git add $ck
                git diff --cached --quiet -- $ck
                if ($LASTEXITCODE -ne 0) { git commit -q -m "Ileriye donuk karne: 1/3/5 gun araliklari" -- $ck }
            }
        }
        catch { Not "UYARI: ileriye donuk karne hatasi: $_" }
    }
    # Veritabani aynasi (SQL Server / SQLite): kayitlari okur.
    try {
        & ".\.venv\Scripts\python.exe" -m src.db.store load
        if ($LASTEXITCODE -ne 0) { Not "UYARI: veritabani yuklemesi basarisiz (kayitlar etkilenmedi)." }
    }
    catch { Not "UYARI: veritabani yukleme hatasi (kayitlar etkilenmedi): $_" }
    # Veri yedegi
    if ($env:ALTIN_YEDEK) {
        try {
            & ".\scripts\yedekle.ps1" -Hedef $env:ALTIN_YEDEK
            if ($LASTEXITCODE -eq 0) { Not "YEDEK TAMAM" } else { Not "UYARI: yedekleme basarisiz (tahmin kaydi etkilenmedi)." }
        }
        catch { Not "UYARI: yedekleme hatasi (tahmin kaydi etkilenmedi): $_" }
    }
    else { Not "UYARI: ALTIN_YEDEK tanimli degil; veri yedegi alinmadi." }
    # GitHub esitlemesi (merge, rebase YOK; ayrinti gunlukte)
    try {
        & ".\scripts\git_esitle.ps1"
        if ($LASTEXITCODE -eq 0) { Not "ESITLEME TAMAM" } else { Not "ESITLEME HATA: ayrinti gunlukte" }
    }
    catch { Not "ESITLEME HATA: $_" }
    # Sistem durumu (data/processed/health/status.json; pano gosterir)
    try {
        & ".\.venv\Scripts\python.exe" -m src.health write --gorev $Gorev --kod $kod --log $log --durum $durum
        if ($LASTEXITCODE -ne 0) { Write-Host "UYARI: sistem durumu yazilamadi." }
    }
    catch { Write-Host "UYARI: sistem durumu hatasi: $_" }
    # Kalp atisi: yalnizca research/live/heartbeat.json commit edilir ve gonderilir (GitHub'daki bekci okur).
    try {
        $kalp = "research/live/heartbeat.json"
        if (Test-Path $kalp) {
            git add $kalp
            git diff --cached --quiet -- $kalp
            if ($LASTEXITCODE -ne 0) {
                git commit -q -m "Kalp atisi ($Gorev)" -- $kalp
                & ".\scripts\git_esitle.ps1"
                if ($LASTEXITCODE -ne 0) { Write-Host "UYARI: kalp atisi GitHub'a gonderilemedi." }
            }
        }
    }
    catch { Write-Host "UYARI: kalp atisi hatasi: $_" }
    Write-Host "Bitis: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | cikis kodu: $kod"
}
catch {
    Write-Host "HATA (beklenmeyen): $_"
    $kod = 1
}
finally {
    Stop-Transcript | Out-Null
}
exit $kod
