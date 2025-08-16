# A simple streamlit application for interacting with locally hosted GPT-OSS model

## Download the model
git clone https://huggingface.co/openai/gpt-oss-120b
git clone https://huggingface.co/openai/gpt-oss-20b

## Install Requirements
pip install -r requirements.txt

## Run
- For a simple console mode:
>python test_gpt-oss-20B.py 

- For run the streamlit based chatbot
>streamlit run app.py --server.port 8501 