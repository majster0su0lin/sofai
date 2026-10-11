$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Project virtual environment not found: $python"
}

& $python -m PyInstaller --noconfirm --clean --onedir --console `
    --name sofai `
    --distpath (Join-Path $PSScriptRoot "dist") `
    --collect-binaries llama_cpp `
    (Join-Path $PSScriptRoot "core.py")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed to build the sofai chat executable."
}

& $python -m PyInstaller --noconfirm --clean --onedir --console `
    --name sofai-trainer `
    --distpath (Join-Path $PSScriptRoot "dist\sofai") `
    --collect-binaries llama_cpp `
    (Join-Path $PSScriptRoot "extr.py")
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed to build the sofai trainer executable."
}

Write-Output "Build complete: $(Join-Path $PSScriptRoot 'dist\sofai')"
