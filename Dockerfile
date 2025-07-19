# Base image
FROM python:3.10-slim

# Diretório da aplicação
WORKDIR /app

# Copiar arquivos
COPY . .

# Instalar dependências
RUN pip install --no-cache-dir -r requirements.txt

# Comando padrão: roda o Streamlit
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
