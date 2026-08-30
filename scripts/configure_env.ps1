param(
    [switch]$KeepExisting
)

$ErrorActionPreference = "Stop"
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$templatePath = Join-Path $projectRoot ".env.example"
$envPath = Join-Path $projectRoot ".env"

if (-not (Test-Path -LiteralPath $templatePath)) {
    throw "Missing template: $templatePath"
}

if ((Test-Path -LiteralPath $envPath) -and $KeepExisting) {
    $lines = [System.Collections.Generic.List[string]](Get-Content -LiteralPath $envPath)
} else {
    $lines = [System.Collections.Generic.List[string]](Get-Content -LiteralPath $templatePath)
}

function Read-SecretValue([string]$prompt) {
    $secureValue = Read-Host $prompt -AsSecureString
    if ($secureValue.Length -eq 0) { return "" }
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureValue)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

function Set-EnvValue([string]$name, [string]$value) {
    if ([string]::IsNullOrWhiteSpace($value)) { return }
    for ($index = 0; $index -lt $lines.Count; $index++) {
        if ($lines[$index] -match "^$([regex]::Escape($name))=") {
            $lines[$index] = "$name=$value"
            return
        }
    }
    $lines.Add("$name=$value")
}

Write-Host "NF-Atlas environment setup" -ForegroundColor Cyan
Write-Host "Input is saved only to .env (ignored by Git). Secret values are not echoed."

$contactEmail = Read-Host "Contact email for OpenAlex and Unpaywall (recommended)"
Set-EnvValue "OPENALEX_EMAIL" $contactEmail
Set-EnvValue "UNPAYWALL_EMAIL" $contactEmail
Set-EnvValue "SILICONFLOW_API_KEY" (Read-SecretValue "SiliconFlow API key")
Set-EnvValue "OPENALEX_API_KEY" (Read-SecretValue "OpenAlex API key (free, required for OpenAlex)")
Set-EnvValue "ELSEVIER_API_KEY" (Read-SecretValue "Elsevier API key (optional; press Enter to skip)")
Set-EnvValue "ELSEVIER_INSTTOKEN" (Read-SecretValue "Elsevier institutional token (optional; press Enter to skip)")
Set-EnvValue "SEMANTIC_SCHOLAR_API_KEY" (Read-SecretValue "Semantic Scholar API key (optional; press Enter to skip)")

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($envPath, $lines, $utf8NoBom)
Write-Host "Saved: $envPath" -ForegroundColor Green

