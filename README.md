# LMKR RAG Chatbot (Mistral 7B Instruct + LangChain + FAISS)

A conversational AI assistant built using:
- **Mistral-7B-Instruct**
- **LangChain RAG pipeline**
- **FAISS vector store**
- **HuggingFace Token**
- **Streamlit UI**

The chatbot answers questions related to LMKR using only factual data from local `.txt` files.  

## How to run:
- Download the file
- Open the .py scripts in VS code
- Open the command prompt terminal in vs code.
- Navigate to folder and create a virtual environment: 'python -m .venv venv'
- Install all dependencies (these are all running perfectly on python 3.11.9): 'pip install -r requirements.txt'
- Set your huggingface api token in environment: 'set HF_API_TOKEN=YOUR_TOKEN_HERE'
- Check if your HF token is set and returns: 'echo %HF_API_TOKEN'
- Run 'utils.py', then 'lmkr_core.py'
- Run the application: 'streamlit run app.py'


