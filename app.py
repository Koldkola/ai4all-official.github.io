import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

st.title("Improved Water & Sanitation EDA and Forecast")

# File upload widget
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, encoding='latin-1', skiprows=12, low_memory=False)
    df = df[["Unnamed: 19", "Total piped (%)", "Sewered facilities"]].dropna()
    df = df.rename(columns={"Unnamed: 19": "Year", "Total piped (%)": "Improved_Water", "Sewered facilities": "Improved_Sanitation"})
    df["Improved_Water"] = pd.to_numeric(df["Improved_Water"], errors='coerce')
    df["Improved_Sanitation"] = pd.to_numeric(df["Improved_Sanitation"], errors='coerce')
    df = df.dropna(subset=["Improved_Water", "Improved_Sanitation"])
    df["gap"] = df["Improved_Sanitation"] - df["Improved_Water"]
    df["total"] = (df["Improved_Water"] + df["Improved_Sanitation"]) / 2

    st.subheader("Data Sample")
    st.dataframe(df.head(20))

    df_agg = df.groupby("Year").mean().reset_index()
    st.subheader("Trends Over Time")
    st.line_chart(df_agg.set_index("Year")[["Improved_Water", "Improved_Sanitation"]])
    st.subheader("Gap Over Time")
    st.line_chart(df_agg.set_index("Year")[["gap"]])

    # Prophet Forecast
    df_prophet = df_agg.copy()
    df_prophet["Year"] = pd.to_datetime(df_prophet["Year"], format='%Y')
    df_prophet = df_prophet.rename(columns={"Year": "ds", "total": "y"})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=5, freq='Y')
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    st.pyplot(fig1)
else:
    st.info("Please upload your CSV file to start.")
