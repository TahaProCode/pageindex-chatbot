@echo off
setlocal enabledelayedexpansion
cd /d "C:\Users\Dell\Documents\Custom Office Templates\pageindex-chatbot"
if exist all_code.txt del all_code.txt

for /r %%f in (*.js *.jsx *.ts *.tsx *.py *.css *.html *.json *.yml *.yaml) do (
    set "fpath=%%f"
    echo !fpath! | findstr /i "node_modules \.git __pycache__ venv \.venv dist build .next package-lock.json" >nul
    if errorlevel 1 (
        echo === FILE: %%f === >> all_code.txt
        type "%%f" >> all_code.txt
        echo. >> all_code.txt
    )
)

REM Dockerfile has no extension, grab it separately
for /r %%f in (Dockerfile Dockerfile.*) do (
    if exist "%%f" (
        echo === FILE: %%f === >> all_code.txt
        type "%%f" >> all_code.txt
        echo. >> all_code.txt
    )
)

echo Done! Created all_code.txt