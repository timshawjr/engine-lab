@echo off
REM ============================================================================
REM Engine Lab - fetch the model files.
REM
REM The models are ~2.3 GB and are deliberately NOT in this package. This
REM downloads them once from Hugging Face; after that the demo runs offline.
REM
REM HF_HUB_DISABLE_XET is set because the large openvino_model.bin files fail
REM to download with Xet enabled, leaving 0-byte .incomplete files that surface
REM later as a confusing "tokenizer was not provided" error.
REM
REM Does NOT require administrator rights.
REM ============================================================================
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: run setup-booth-machine.bat first.
    pause
    exit /b 1
)

set HF_HUB_DISABLE_XET=1

echo.
echo === Downloading Engine Lab models (about 2.3 GB, needs network) ===
echo.
.venv\Scripts\python.exe tools\download_models.py
if errorlevel 1 (
    echo.
    echo ERROR: the download did not finish.
    echo Check the messages above, then run this again - it resumes.
    pause
    exit /b 1
)

echo.
echo === Models present ===
.venv\Scripts\python.exe -c "import json,pathlib; c=json.load(open('config/models.json',encoding='utf-8')); print(len(c.get('models',[])),'vision models')" 2>nul
if exist "models\Qwen3-Embedding-0.6B-int8-ov\openvino_model.bin" (
    echo   Qwen3-Embedding-0.6B-int8-ov  OK
) else (
    echo   Qwen3-Embedding-0.6B-int8-ov  MISSING
)
if exist "models\Qwen3-1.7B-int4-ov\openvino_model.bin" (
    echo   Qwen3-1.7B-int4-ov             OK
) else (
    echo   Qwen3-1.7B-int4-ov             MISSING
)
if exist "models\rag\corpus.json" (
    echo   rag corpus + index             OK
) else (
    echo   rag corpus + index             MISSING - ask for the package that includes models\rag
)

echo.
echo Next: run verify-booth.bat
pause