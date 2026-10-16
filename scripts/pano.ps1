# Web panosu ve API (salt okunur). Kullanim (proje kokunden):
#   powershell -ExecutionPolicy Bypass -File scripts\pano.ps1            # once veritabanini gunceller
#   powershell -ExecutionPolicy Bypass -File scripts\pano.ps1 -Yuklemeden
# Sunucu yalnizca bu bilgisayardan erisilir (127.0.0.1). Kapatmak icin Ctrl+C.
param([switch]$Yuklemeden, [int]$Port = 8000)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$py = ".\.venv\Scripts\python.exe"
if (-not $Yuklemeden) {
    & $py -m src.db.store load
    if ($LASTEXITCODE -ne 0) { Write-Host "UYARI: veritabani yuklemesi basarisiz; pano son yuklemeyi gosterir." -ForegroundColor Yellow }
}
Start-Process "http://127.0.0.1:$Port"
& $py -m uvicorn src.api.app:app --host 127.0.0.1 --port $Port --no-server-header --no-proxy-headers
