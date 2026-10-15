# Windows Gorev Zamanlayici kurulumu (bir kez; tekrar calistirilabilir, mevcut gorevlerin uzerine yazar)
# Kullanim (proje kokunden, normal PowerShell): powershell -ExecutionPolicy Bypass -File scripts\gorev_kur.ps1
# Gorevler YALNIZCA bu kullanici oturumu acikken calisir (parola saklanmaz). Bilgisayar uykudaysa
# kacirilan gorev uyaninca calisir (StartWhenAvailable); zaman disiysa betik tahmini kural geregi reddeder.
#   AltinTahmin-Sabah : hafta ici 12:30 ve 13:30 (yedek deneme; ilk basariliysa ikinci "zaten kayitli" der)
#   AltinTahmin-Aksam : hafta ici 22:30 (NY 14:00 cubugu yaz saatinde 21:00, kis saatinde 22:00 TSI'de kapanir)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$sarmalayici = Join-Path $root "scripts\zamanli_calistir.ps1"
$gunler = @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
$ayar = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

function Kur([string]$ad, [string]$gorev, [string[]]$saatler, [string]$aciklama) {
    $arg = "-NoProfile -ExecutionPolicy Bypass -File `"$sarmalayici`" -Gorev $gorev"
    $eylem = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arg -WorkingDirectory $root
    $tetik = @($saatler | ForEach-Object { New-ScheduledTaskTrigger -Weekly -DaysOfWeek $gunler -At $_ })
    Register-ScheduledTask -TaskName $ad -Action $eylem -Trigger $tetik -Settings $ayar -Description $aciklama -Force | Out-Null
    Write-Host "Kuruldu: $ad ($($saatler -join ', '), hafta ici)" -ForegroundColor Green
}

Kur "AltinTahmin-Sabah" "sabah" @("12:30", "13:30") "LIVE_PHYS_V1 + LIVE_PHYS_INT_V1 sabah fiziki altin tahmini"
Kur "AltinTahmin-Aksam" "aksam" @("22:30") "LIVE_V1 aksam esanli teorik gram tahmini"
Get-ScheduledTask -TaskName "AltinTahmin-*" | Select-Object TaskName, State | Format-Table -AutoSize
Write-Host "Kayitlar: $root\logs\ | Elle deneme: Start-ScheduledTask -TaskName AltinTahmin-Sabah"
