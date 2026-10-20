@echo off
setlocal
cd /d C:\Users\lenovo\Desktop\fixed_income_plus_allocation
set PYTHONUTF8=1
set FOR_DISABLE_CONSOLE_CTRL_HANDLER=1
C:\Users\lenovo\AppData\Local\Programs\Python\Python312\pythonw.exe scripts\136_run_hierarchical_asset_itransformer_ppo_v7.py --config config\hierarchical_asset_itransformer_v43_clean_calendar.yaml --profiles low_vol,balanced,high_vol >> output\reports\v43_clean_calendar_portfolio.log 2>&1
endlocal
