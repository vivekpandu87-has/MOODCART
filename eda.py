import streamlit as st
import plotly.express as px

def run_eda(df):
    st.subheader("Preview")
    st.write(df.head())

    st.subheader("Target Distribution")
    if "Interest_in_MoodCart" in df.columns:
        st.plotly_chart(px.histogram(df, x="Interest_in_MoodCart"), use_container_width=True)

    if "Income" in df.columns and "Monthly_Spend" in df.columns:
        st.subheader("Income vs Spend")
        st.plotly_chart(px.box(df, x="Income", y="Monthly_Spend"), use_container_width=True)

    if "Mood" in df.columns:
        st.subheader("Mood Distribution")
        st.plotly_chart(px.histogram(df, x="Mood"), use_container_width=True)
