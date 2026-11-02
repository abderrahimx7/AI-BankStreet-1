$ErrorActionPreference = "Stop"

$projectFolder = $PSScriptRoot
$pythonPath = Join-Path $projectFolder ".venv-check\Scripts\python.exe"

if (-not (Test-Path $pythonPath)) {
    throw "Project Python environment not found."
}

Push-Location $projectFolder

try {
    & $pythonPath -m streamlit run app.py

    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}