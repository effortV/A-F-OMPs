$ErrorActionPreference = "Continue"
Write-Host "=== Docker ===" -ForegroundColor Cyan
docker --version
docker compose version
docker compose ps

Write-Host "=== API ===" -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/api/health" -TimeoutSec 10
    $health | ConvertTo-Json -Depth 5
} catch {
    Write-Host "API is not ready: $($_.Exception.Message)" -ForegroundColor Yellow
}

Write-Host "=== Ports ===" -ForegroundColor Cyan
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
    Where-Object { $_.LocalPort -in 8000, 8501, 7474, 7687, 9000, 9001 } |
    Select-Object LocalAddress, LocalPort, OwningProcess

