$ErrorActionPreference = "Stop"

$projectFolder = $PSScriptRoot
$pythonPath = Join-Path $projectFolder ".venv-check\Scripts\python.exe"

$env:HADOOP_HOME = "C:\Users\abder\Desktop\hadoop"
$env:Path = "$env:HADOOP_HOME\bin;$env:Path"

if (-not (Test-Path $pythonPath)) {
    throw "Project Python environment not found."
}

if (-not (Test-Path "$env:HADOOP_HOME\bin\winutils.exe")) {
    throw "winutils.exe not found."
}

$steps = @(
    "lesson_14.py"
    "lesson_15.py"
    "lesson_16.py"
    "lesson_17.py"
    "check_replay_risk.py"
    "lesson_18.py"
    "lesson_19.py"
)

Push-Location $projectFolder

try {
    foreach ($step in $steps) {
        Write-Host "Running $step..."

        & $pythonPath (Join-Path $projectFolder $step)

        if ($LASTEXITCODE -ne 0) {
            throw "$step failed with exit code $LASTEXITCODE."
        }
    }

    Write-Host "BankStreet replay pipeline completed."
}
finally {
    Pop-Location
}