@echo off
REM Creates the SIH26174 repo scaffold on Windows.
REM Run once from inside an empty repo folder:   bootstrap.bat

setlocal

echo Creating folders...
mkdir configs\protocol 2>nul
mkdir configs\model 2>nul
for %%d in (capture perception features models decoding runtime api eval) do (
  mkdir src\har\%%d 2>nul
)
mkdir scripts ui notebooks tests 2>nul
mkdir assets\voice assets\reference 2>nul
mkdir data\raw data\features data\splits data\runs 2>nul

echo Creating package markers...
type nul > src\har\__init__.py
for %%d in (capture perception features models decoding runtime api eval) do (
  type nul > src\har\%%d\__init__.py
)

echo Creating module stubs...
for %%f in (protocol.py events.py) do type nul > src\har\%%f
for %%f in (source.py ringbuffer.py recorder.py streamer.py) do type nul > src\har\capture\%%f
for %%f in (pose2d.py lifter3d.py canonical.py objects.py tracker.py contact.py) do type nul > src\har\perception\%%f
for %%f in (extractor.py windows.py augment.py) do type nul > src\har\features\%%f
for %%f in (base.py mock.py tcn.py mstcn.py ssl.py uncertainty.py) do type nul > src\har\models\%%f
for %%f in (smoothing.py fsm.py viterbi.py) do type nul > src\har\decoding\%%f
for %%f in (engine.py validator.py voice.py logger.py) do type nul > src\har\runtime\%%f
for %%f in (server.py routes.py) do type nul > src\har\api\%%f
for %%f in (metrics.py splits.py report.py) do type nul > src\har\eval\%%f

for %%f in (extract_features.py train.py evaluate.py export_onnx.py benchmark_edge.py run_live.py) do type nul > scripts\%%f
for %%f in (capture.yaml features.yaml deploy.yaml) do type nul > configs\%%f
for %%f in (mgs01.yaml schema.json) do type nul > configs\protocol\%%f
for %%f in (tcn_baseline.yaml mstcn_causal.yaml lifter.yaml) do type nul > configs\model\%%f
for %%f in (index.html app.js style.css) do type nul > ui\%%f
for %%f in (test_protocol.py test_viterbi.py test_events.py) do type nul > tests\%%f

echo Writing pyproject.toml...
(
echo [project]
echo name = "har"
echo version = "0.1.0"
echo requires-python = "^>=3.10"
echo.
echo [build-system]
echo requires = ["setuptools^>=68"]
echo build-backend = "setuptools.build_meta"
echo.
echo [tool.setuptools.packages.find]
echo where = ["src"]
) > pyproject.toml

echo Writing .gitignore...
(
echo data/
echo *.mp4
echo *.npz
echo *.npy
echo *.pt
echo *.pth
echo *.onnx
echo checkpoints/
echo __pycache__/
echo *.pyc
echo .venv/
echo .ipynb_checkpoints/
echo Thumbs.db
) > .gitignore

echo.
echo Scaffold created.
echo.
echo Next:
echo   1. Copy preflight.py, record_take.py, verify_take.py into scripts\
echo   2. Copy shoot.bat into this folder
echo   3. python -m venv .venv
echo   4. .venv\Scripts\activate
echo   5. pip install -e .
echo.
endlocal