Tạo và Kích hoạt Môi trường ảo:
python -m venv.venv

Terminal 1: backend
kích hoạt môi trường ảo: .venv\Scripts\Activate
chuyển đến thư mục app
chạy: powershell uvicorn api:app --reload 

Terminal 2: frontend

kích hoạt môi trường ảo: .venv\Scripts\Activate
chuyển đến thư mục app
chạy: powershell streamlit run dashboard.py



