@echo off
setlocal
cd /d C:\Users\lenovo\Desktop\fixed_income_plus_allocation
set PYTHONUTF8=1
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
C:\Users\lenovo\AppData\Local\Programs\Python\Python312\pythonw.exe scripts\run_v43_clean_calendar_full_pipeline.py --from-stage train >> output\reports\v43_clean_calendar_long_training.log 2>&1
endlocal
