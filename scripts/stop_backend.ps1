$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location -LiteralPath $projectRoot

if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker compose down
    if ($LASTEXITCODE -ne 0) { throw "docker compose down failed with exit code $LASTEXITCODE" }
}
else {
    $distribution = "Ubuntu-24.04"
    $keepAliveMarker = "-d $distribution -- sleep infinity"
    $wslProjectRoot = (& wsl.exe -d $distribution -- wslpath -a $projectRoot).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $wslProjectRoot) {
        throw "Could not map project path into WSL $distribution."
    }
    $safeWslPath = $wslProjectRoot.Replace("'", "'`"'`"'")
    $composeCommand = "cd '$safeWslPath' && docker compose down"
    & wsl.exe -d $distribution -- bash -lc $composeCommand
    if ($LASTEXITCODE -ne 0) { throw "WSL docker compose down failed with exit code $LASTEXITCODE" }

    Get-CimInstance Win32_Process -Filter "Name = 'wsl.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine.Contains($keepAliveMarker) } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

Write-Host "Backend stopped. PostgreSQL, Neo4j, Chroma and object-storage volumes were preserved." -ForegroundColor Green
