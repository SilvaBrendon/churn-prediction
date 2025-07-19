# Previsão de Cancelamento de Clientes (Churn Prediction)

Este projeto tem como objetivo prever o **cancelamento de clientes (churn)** com base em dados de uma operadora de telecomunicações. Ele cobre **todo o pipeline de ciência de dados**, do EDA ao deploy com Streamlit e FastAPI, com organização modular e foco em boas práticas.

---

## O que este projeto faz?

- Explora e analisa dados reais de clientes
- Cria um pipeline de pré-processamento profissional com Scikit-Learn
- Compara modelos supervisionados como Logistic Regression, Random Forest e XGBoost
- Escolhe o melhor modelo com base em métricas reais de negócio
- Salva modelo e pipeline para produção
- Entrega uma interface visual com Streamlit
- Expõe uma API REST com FastAPI
- Disponível para deploy local ou via Docker

---

## Estrutura do Projeto

```text
projeto_churn/
├── app/                        # Interface e API
│   ├── model.pkl               # Modelo final treinado
│   ├── preprocessor.pkl        # Pipeline de pré-processamento
│   ├── streamlit_app.py        # Interface visual para testar previsões
│   └── fastapi_app.py          # API REST para integração via JSON
│
├── data/
│   └── telco_churn.csv         # Dataset original (dados de clientes)
│
├── notebooks/
│   ├── eda_telco_churn.ipynb   # Análise exploratória de dados (EDA)
│   └── modelagem_telco_churn.ipynb # Treinamento, avaliação e salvamento do modelo
│
├── src/
│   ├── preprocessing.py        # Funções de tratamento e pipeline
│   └── modeling.py             # Funções de modelagem e avaliação
│
├── requirements.txt            # Dependências do projeto
├── Dockerfile                  # Script para criar imagem Docker
└── README.md                   # Este arquivo

---

## 📌 Dataset utilizado

- 📂 `data/telco_churn.csv`: Dados da [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Inclui: perfil do cliente, tipo de contrato, serviços contratados, valores pagos e se ele cancelou o serviço

---

## 🧾 Dicionário de Dados

| Variável         | Tipo       | Descrição |
|------------------|------------|-----------|
| `gender`         | Categórica | Gênero do cliente |
| `SeniorCitizen`  | Numérica   | 1 se é idoso |
| `tenure`         | Numérica   | Meses como cliente |
| `Contract`       | Categórica | Tipo de contrato |
| `MonthlyCharges` | Numérica   | Valor mensal cobrado |
| `TotalCharges`   | Numérica   | Valor total acumulado |
| `Churn`          | Alvo       | Cancelou o serviço? |

*(consulte `notebooks/eda_telco_churn.ipynb` para o dicionário completo)*

---

##  Como rodar este projeto

### 1. Clone o repositório

bash
git clone https://github.com/seu-usuario/projeto-churn.git
cd projeto-churn

### 2. Crie o ambiente virtual e instale as dependências

python -m venv venv
source venv/bin/activate no linux/mac os # ou .\venv\Scripts\activate no Windows
pip install -r requirements.txt

### 3. Executar localmente

streamlit run app/streamlit_app.py

Acesse http://localhost:8501

### 4. Api com FastAPI

uvicorn app.fastapi_app:app --reload

Acesse http://localhost:8000/docs para a documentação 

### 5. Executar com Docker (aplicação streamlit)

docker build -t churn-api .

### 6. Resultados

    Melhor modelo: Logistic Regression

    Acurácia: ~81%

    F1-score para churn: ~0.61

    ROC AUC: ~0.84

    O modelo foi salvo e integrado à interface e à API para uso imediato em produção.

### Tecnologias Utilizadas

Python 3.10

Pandas, NumPy, Scikit-Learn, XGBoost

Matplotlib, Seaborn (visualização)

Streamlit (frontend)

FastAPI + Uvicorn (backend)

Joblib (serialização)

Docker (deploy local)

### Autor

Brendon Lucas Pereira da Silva

Cientista de Dados com experiência em Big Data, Machine Learning, automação de pipelines, Construção de Dashboards.

Email: brendon.lucas@outlook.com.br
LinkedIn[https://www.linkedin.com/in/brendon-lucas-471b99145/] • GitHub[github.com/SilvaBrendon]
