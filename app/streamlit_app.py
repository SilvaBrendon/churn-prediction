import streamlit as st
import pandas as pd
import joblib
import os

# Carregar modelo e pipeline
model = joblib.load(os.path.join('app', 'model.pkl'))
preprocessor = joblib.load(os.path.join('app', 'preprocessor.pkl'))

st.set_page_config(page_title="Previsão de Churn", layout="centered")
st.title("🔍 Previsão de Cancelamento de Clientes (Churn)")

st.markdown("Preencha os dados do cliente:")

# Dicionários de tradução
map_sim_nao = {'Sim': 'Yes', 'Não': 'No'}
map_genero = {'Masculino': 'Male', 'Feminino': 'Female'}
map_idoso = {'Sim': 1, 'Não': 0}

# Entradas do usuário (em português → traduzidas para o modelo)
gender = map_genero[st.selectbox("Gênero", ['Feminino', 'Masculino'])]
SeniorCitizen = map_idoso[st.selectbox("É idoso?", ['Sim', 'Não'])]
Partner = map_sim_nao[st.selectbox("Possui parceiro(a)?", ['Sim', 'Não'])]
Dependents = map_sim_nao[st.selectbox("Possui dependentes?", ['Sim', 'Não'])]
tenure = st.slider("Meses como cliente", 0, 72, 12)
PhoneService = map_sim_nao[st.selectbox("Possui serviço telefônico?", ['Sim', 'Não'])]

MultipleLines = st.selectbox("Múltiplas linhas telefônicas?", [
    'Sim', 'Não', 'Sem serviço telefônico'])
MultipleLines = {
    'Sim': 'Yes', 'Não': 'No', 'Sem serviço telefônico': 'No phone service'
}[MultipleLines]

InternetService = st.selectbox("Tipo de internet", ['DSL', 'Fibra óptica', 'Sem internet'])
InternetService = {
    'DSL': 'DSL', 'Fibra óptica': 'Fiber optic', 'Sem internet': 'No'
}[InternetService]

def traduzir_servico(texto):
    return {
        'Sim': 'Yes', 'Não': 'No', 'Sem internet': 'No internet service'
    }[texto]

OnlineSecurity = traduzir_servico(st.selectbox("Segurança online", ['Sim', 'Não', 'Sem internet']))
OnlineBackup = traduzir_servico(st.selectbox("Backup online", ['Sim', 'Não', 'Sem internet']))
DeviceProtection = traduzir_servico(st.selectbox("Proteção de dispositivo", ['Sim', 'Não', 'Sem internet']))
TechSupport = traduzir_servico(st.selectbox("Suporte técnico", ['Sim', 'Não', 'Sem internet']))
StreamingTV = traduzir_servico(st.selectbox("Streaming TV", ['Sim', 'Não', 'Sem internet']))
StreamingMovies = traduzir_servico(st.selectbox("Streaming Filmes", ['Sim', 'Não', 'Sem internet']))

Contract = st.selectbox("Tipo de contrato", ['Mensal', '1 ano', '2 anos'])
Contract = {
    'Mensal': 'Month-to-month',
    '1 ano': 'One year',
    '2 anos': 'Two year'
}[Contract]

PaperlessBilling = map_sim_nao[st.selectbox("Fatura digital?", ['Sim', 'Não'])]

PaymentMethod = st.selectbox("Método de pagamento", [
    'Cheque eletrônico', 'Cheque enviado', 'Transferência bancária automática', 'Cartão de crédito automático'
])
PaymentMethod = {
    'Cheque eletrônico': 'Electronic check',
    'Cheque enviado': 'Mailed check',
    'Transferência bancária automática': 'Bank transfer (automatic)',
    'Cartão de crédito automático': 'Credit card (automatic)'
}[PaymentMethod]

MonthlyCharges = st.number_input("Cobrança mensal", min_value=0.0, max_value=200.0, value=70.0)
TotalCharges = st.number_input("Cobrança total", min_value=0.0, max_value=10000.0, value=2000.0)

# Botão de previsão
if st.button("🔎 Verificar Churn"):
    input_dict = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
    }

    input_df = pd.DataFrame([input_dict])

    # Previsão
    X_processed = preprocessor.transform(input_df)
    prob = model.predict_proba(X_processed)[0][1]
    pred = model.predict(X_processed)[0]

    st.subheader("Resultado:")
    st.write(f"🔢 Probabilidade de cancelamento: **{prob:.2%}**")
    if pred == 1:
        st.error("⚠️ Este cliente provavelmente vai cancelar o serviço.")
    else:
        st.success("✅ Este cliente provavelmente vai continuar com o serviço.")
