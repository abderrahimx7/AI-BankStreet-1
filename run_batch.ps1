$ErrorActionPreference = "Stop"

$env:HADOOP_HOME = Join-Path $env:USERPROFILE "Desktop\hadoop"
$env:Path = "$env:HADOOP_HOME\bin;$env:Path"

$pythonExe = Join-Path $PSScriptRoot ".venv-check\Scripts\python.exe"

if (-not (Test-Path "$env:HADOOP_HOME\bin\winutils.exe")) {
    throw "winutils.exe not found in $env:HADOOP_HOME\bin"
}

if (-not (Test-Path "$env:HADOOP_HOME\bin\hadoop.dll")) {
    throw "hadoop.dll not found in $env:HADOOP_HOME\bin"
}

& $pythonExe (Join-Path $PSScriptRoot "lesson_11.py")

if ($LASTEXITCODE -ne 0) {
    throw "Price and feature processing failed."
}

& $pythonExe (Join-Path $PSScriptRoot "lesson_12.py")

if ($LASTEXITCODE -ne 0) {
    throw "Annual report processing failed."
}

Write-Host "BankStreet batch completed."