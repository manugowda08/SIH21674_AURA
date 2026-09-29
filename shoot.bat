@echo off
if "%~3"=="" ( echo Usage: shoot ^<actor^> ^<session^> ^<seqtype^> & exit /b 1 )
python scripts\record_take.py --actor %1 --session %2 --seqtype %3 --outdir data\raw --camera 0 --width 1280 --height 720 --fps 20 --lighting overhead+lamp --camera-pos pos1