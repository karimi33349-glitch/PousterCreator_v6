$ErrorActionPreference = "Stop"

python tools/download_models.py
flet build apk --python-version 3.12 --split-per-abi --yes --skip-flutter-doctor
Write-Host "ساخت APK پایان یافت. فایل‌ها در build\apk قرار دارند."
