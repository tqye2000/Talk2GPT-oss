@echo off
echo Starting GPT-OSS-20B Streamlit Chatbot...
echo.
echo Make sure you have installed the requirements:
echo pip install -r requirements.txt
echo.
echo Starting Streamlit server...
"C:\Users\yetid8\AppData\Local\Programs\Python\Python312\python.exe" -m streamlit run app.py --server.port 8501 --server.headless false
pause
