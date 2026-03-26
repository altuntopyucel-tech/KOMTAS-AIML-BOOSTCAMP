
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("fraud_detection_pipeline.pkl")

st.title("Sigorta Dolandırıcılığı Tahmin Uygulaması")
st.markdown("Lütfen işlemin ayrıntılarını girin ve tahmini görmek için butona basın")

st.markdown("---")

transaction_type = st.selectbox("İşlem Türü", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
amount = st.number_input("Tutar (Amount)", min_value=0.0, value=0.0)
oldbalanceOrg = st.number_input("Gönderenin Eski Bakiyesi ", min_value=0.0, value=00.0)
newbalanceOrig = st.number_input("Gönderenin Yeni Bakiyesi ", min_value=0.0, value=0.0)
oldbalanceDest = st.number_input("Alıcının Eski Bakiyesi ", min_value=0.0, value=0.0)
newbalanceDest = st.number_input("Alıcının Yeni Bakiyesi ", min_value=0.0, value=0.0)
errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
errorBalanceDest = oldbalanceDest + amount - newbalanceDest
type_PAYMENT = 1 if transaction_type=="PAYMENT" else 0
type_TRANSFER = 1 if transaction_type=="TRANSFER" else 0
type_CASH_OUT = 1 if transaction_type=="CASH_OUT" else 0
type_DEPOSIT = 1 if transaction_type=="DEPOSIT" else 0
input_df = pd.DataFrame([{
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "errorBalanceOrig": errorBalanceOrig,
    "errorBalanceDest": errorBalanceDest,
    "type_PAYMENT": type_PAYMENT,
    "type_TRANSFER": type_TRANSFER,
    "type_CASH_OUT": type_CASH_OUT,
    "type_DEPOSIT": type_DEPOSIT
}])

if st.button("Tahmin Et"):
    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.error(f"!!!!!! Bu işlemin 'DOLANDIRICILIK' olma ihtimali yüksek! ({prediction_prob*100:.2f}%)")
    else:
        st.success(f" Bu işlem muhtemelen 'NORMAL' . Dolandırıcılık olasılığı: {prediction_prob*100:.2f}%")



