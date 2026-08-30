$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath (Join-Path $projectRoot ".env"))) {
    throw "Missing .env. Run: powershell -ExecutionPolicy Bypass -File scripts\configure_env.ps1"
}

if (Get-Command docker -ErrorAction SilentlyContinue) {
    docker compose up --build -d
    if ($LASTEXITCODE -ne 0) { throw "docker compose failed with exit code $LASTEXITCODE" }
    docker compose ps
}
else {
    $distribution = "Ubuntu-24.04"
    $keepAliveMarker = "-d $distribution -- sleep infinity"
    if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
        throw "Windows Docker and WSL were both unavailable."
    }
    $wslProjectRoot = (& wsl.exe -d $distribution -- wslpath -a $projectRoot).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $wslProjectRoot) {
        throw "Could not map project path into WSL $distribution."
    }
    # WSL can stop the entire distribution shortly after the last Windows-side
    # wsl.exe process exits, even though Docker services are running in systemd.
    # Keep one hidden process alive so API/worker containers remain reachable.
    $existingKeeper = Get-CimInstance Win32_Process -Filter "Name = 'wsl.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine.Contains($keepAliveMarker) } |
        Select-Object -First 1
    if (-not $existingKeeper) {
        Start-Process -FilePath "wsl.exe" -ArgumentList @("-d", $distribution, "--", "sleep", "infinity") -WindowStyle Hidden | Out-Null
        Start-Sleep -Seconds 2
    }
    $safeWslPath = $wslProjectRoot.Replace("'", "'`"'`"'")
    $composeCommand = "cd '$safeWslPath' && docker compose up --build -d && docker compose ps"
    & wsl.exe -d $distribution -- bash -lc $composeCommand
    if ($LASTEXITCODE -ne 0) { throw "WSL docker compose failed with exit code $LASTEXITCODE" }
}
Write-Host "Backend started. API docs: http://localhost:8000/docs" -ForegroundColor Green
Write-Host "Now start Streamlit from Anaconda Prompt: scripts\start_streamlit_anaconda.cmd" -ForegroundColor Cyan
