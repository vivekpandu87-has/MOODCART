import streamlit as st
import pandas as pd
from utils import load_data
from eda import run_eda
from models import train_classification, train_regression, train_clustering, association_mining, save_model, load_model, predict_new

st.set_page_config(page_title="MoodCart", layout="wide")
st.title("MoodCart Analytics Dashboard")

menu = st.sidebar.selectbox("Menu", ["Upload Data","EDA","Classification","Regression","Clustering","Association Rules","Predict New"])

if "df" not in st.session_state:
    st.session_state["df"] = None

if menu == "Upload Data":
    f = st.file_uploader("Upload CSV", type=["csv"])
    if f:
        st.session_state["df"] = load_data(f)
        st.success("Data Loaded")
        st.write(st.session_state["df"].head())

elif menu == "EDA":
    if st.session_state["df"] is not None:
        run_eda(st.session_state["df"])
    else:
        st.warning("Upload data first")

elif menu == "Classification":
    if st.session_state["df"] is not None:
        res, model, le, cols = train_classification(st.session_state["df"])
        st.write(res)
        save_model(model, le, cols)
        st.success("Model Saved")
    else:
        st.warning("Upload data first")

elif menu == "Regression":
    if st.session_state["df"] is not None:
        scores, _ = train_regression(st.session_state["df"])
        st.write(scores)
    else:
        st.warning("Upload data first")

elif menu == "Clustering":
    if st.session_state["df"] is not None:
        labels = train_clustering(st.session_state["df"])
        st.write(pd.Series(labels).value_counts())
    else:
        st.warning("Upload data first")

elif menu == "Association Rules":
    if st.session_state["df"] is not None:
        rules = association_mining(st.session_state["df"])
        st.write(rules.head(20))
    else:
        st.warning("Upload data first")

elif menu == "Predict New":
    f = st.file_uploader("Upload new data", type=["csv"])
    if f:
        df_new = pd.read_csv(f)
        model, le, cols = load_model()
        preds = predict_new(df_new, model, le, cols)
        df_new["Prediction"] = preds
        st.write(df_new.head())
