# Veri yedegi (Git'e girmeyen, yeniden uretilemeyen dosyalar)
# Kullanim (proje kokunden):
#   powershell -ExecutionPolicy Bypass -File scripts\yedekle.ps1 -Hedef "D:\altin_yedek"
#   Hedef verilmezse ALTIN_YEDEK kullanici ortam degiskeni kullanilir (setx ALTIN_YEDEK "...").
# Kopyalanan: data\raw (Yahoo gunluk/saatlik ve EVDS anlik goruntuleri), data\processed (sinav md5'lerine
#   bagli ana paneller dahil), logs. Yalnizca YENI veya DEGISMIS dosyalar kopyalanir (robocopy /E /XO).
# Hedefte HICBIR SEY SILINMEZ (/MIR, /PURGE, /MOV kullanilmaz). Kaynak dosyalar degistirilmez.
# Cikis: 0 basarili; 1 hedef yok/gecersiz veya kopyalama hatasi (robocopy kodu >= 8).
param([string]$Hedef = $env:ALTIN_YEDEK)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

if (-not $Hedef) {
    Write-Host "HATA: yedek hedefi yok (-Hedef verin ya da setx ALTIN_YEDEK ...)." -ForegroundColor Red
    exit 1
}
$hedefTam = [System.IO.Path]::GetFullPath($Hedef).TrimEnd("\")
$kokTam = [System.IO.Path]::GetFullPath($root).TrimEnd("\")
if ($hedefTam -eq $kokTam -or $hedefTam.StartsWith($kokTam + "\", [System.StringComparison]::OrdinalIgnoreCase)) {
    Write-Host "HATA: yedek hedefi proje klasorunun icinde olamaz: $hedefTam" -ForegroundColor Red
    exit 1
}

$hata = 0
foreach ($k in @("data\raw", "data\processed", "logs")) {
    if (-not (Test-Path $k)) { Write-Host "Atlandi (yok): $k"; continue }
    robocopy $k (Join-Path $hedefTam $k) /E /XO /R:1 /W:2 /NFL /NDL /NJH /NJS /NP | Out-Null
    $rc = $LASTEXITCODE
    if ($rc -ge 8) { Write-Host "HATA: $k yedeklenemedi (robocopy $rc)." -ForegroundColor Red; $hata = 1 }
    else { Write-Host "Yedek tamam: $k (robocopy $rc)" }
}
exit $hata
