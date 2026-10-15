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
