# استخدام صورة Python الرسمية كقاعدة
FROM python:3.9-slim

# تعيين دليل العمل
WORKDIR /app

# نسخ ملفات المشروع
COPY . .

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

# تنزيل بيانات NLTK المطلوبة
RUN python -m nltk.downloader punkt

# تعيين المنفذ الذي سيعمل عليه التطبيق
EXPOSE 8501

# الأمر الذي سيتم تنفيذه عند تشغيل الحاوية
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
