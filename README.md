# PousterCreator Professional — Android Background Removal

این نسخه برای پروژهٔ Flet 1.0.0 طراحی شده و حذف زمینه را با ONNX Runtime + مدل سبک U²-NetP انجام می‌دهد.

## نکته مهم درباره Android
برای این معماری **ساختن پوشهٔ `android/` در ریشه پروژه لازم نیست**. Flet خودش پروژهٔ Android/Flutter را هنگام `flet build apk` ایجاد می‌کند.

## فایل‌های لازم

```text
assets/
├── template1.jpg
├── template2.jpg
├── calibrib.ttf
└── models/
    └── u2netp.onnx
```

دو زمینه باید همان فایل‌های واقعی پروژه شما و هر دو دقیقاً 840×1260 باشند.

## دریافت مدل

```powershell
python tools/download_models.py
```

این اسکریپت SHA-256 مدل را بررسی می‌کند و فایل ناقص/خراب را قبول نمی‌کند.

## اجرای ویندوز

```powershell
pip install -r requirements.txt
python tools/download_models.py
python main.py
```

## ساخت APK

```powershell
pip install flet==1.0.0
python tools/download_models.py
flet build apk --python-version 3.12 --split-per-abi --yes --skip-flutter-doctor
```

یا در PowerShell:

```powershell
.\build_android.ps1
```

## رفتار حذف زمینه

حذف زمینه فقط با فشردن دکمه «حذف هوشمند زمینه» اجرا می‌شود و در حین drag/zoom دوباره مدل اجرا نمی‌شود. بنابراین حرکت کراپ سبک باقی می‌ماند.

مدل U²-NetP حدود 4.57MB است؛ این انتخاب یک موازنه بین حجم APK و کیفیت است. مدل‌های انسانی سنگین‌تر دقت بالقوه بالاتری دارند اما حجم بسیار بیشتری وارد برنامه می‌کنند.

## بررسی فنی

- Flet: 1.0.0
- Python Android: 3.12
- Pillow: 12.2.0
- NumPy: 2.4.6
- ONNX Runtime: 1.27.0
- حداقل Android API: 24
- خروجی مدل: PNG با آلفای شفاف
