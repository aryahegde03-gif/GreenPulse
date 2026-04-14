import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pipeline.clean import clean_dataframe
from features.prediction import run_carbon_prediction

from config import PLOTLY_DARK_TEMPLATE


def _to_df(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if not df.empty and "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df


def render_prediction_page() -> None:
    st.title("Carbon Emission Prediction")
    st.markdown(
        "<p style='color:#8b949e;'>Supervised ML model (Random Forest Regressor) trained to predict carbon emissions from engineered time-series features.</p>",
        unsafe_allow_html=True,
    )

    pred_results = st.session_state.get("prediction_results", {})

    if not pred_results:
        st.info("No prediction results found. Click Refresh Data in the sidebar to run the prediction pipeline.")
    
    # --- Custom CSV Upload Section ---
    st.markdown("---")
    st.subheader("📤 Predict from Custom CSV")
    st.write("Upload a CSV file with `Timestamp`, `Server_ID`, and `Power_Usage_Watts` to get AI carbon predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="custom_csv_upload")
    
    if uploaded_file is not None:
        if st.button("🚀 Run Prediction on Uploaded File"):
            try:
                # 1. Read the file
                input_df = pd.read_csv(uploaded_file)
                
                # 2. Clean and calculate carbon_kg (needed for feature engineering)
                with st.spinner("Cleaning and preparing data..."):
                    cleaned_df = clean_dataframe(input_df)
                
                # 3. Run prediction pipeline
                with st.spinner("🤖 Running AI Model..."):
                    custom_results = run_carbon_prediction(cleaned_df)
                
                if custom_results:
                    st.session_state.custom_prediction_results = custom_results
                    st.success("Custom prediction complete!")
                else:
                    st.error("Model failed to generate predictions. Ensure your CSV has enough varied data.")
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")

    # Display Custom Results if they exist
    if "custom_prediction_results" in st.session_state:
        st.markdown("---")
        st.subheader("🎯 Custom Prediction Results")
        res = st.session_state.custom_prediction_results
        
        c1, c2, c3 = st.columns(3)
        c1.metric("R² Score", f"{res.get('test_r2', 0):.4f}")
        c2.metric("MAE", f"{res.get('test_mae', 0):.6f}")
        c3.metric("Samples", f"{res.get('test_samples', 0):,}")
        
        custom_df = pd.DataFrame(res.get("predictions", []))
        custom_df["Timestamp"] = pd.to_datetime(custom_df["Timestamp"])
        
        fig = px.line(custom_df, x="Timestamp", y=["actual_carbon_kg", "predicted_carbon_kg"], 
                      title="Custom Data: Actual vs Predicted",
                      labels={"value": "Carbon (kg)", "variable": "Type"},
                      template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        
        st.download_button(
            "📥 Download Custom Predictions CSV",
            data=custom_df.to_csv(index=False).encode("utf-8"),
            file_name="custom_predictions.csv",
            mime="text/csv"
        )
    
    if not pred_results:
        return

    # --- Model Performance Metrics ---
    st.subheader("Model Performance")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("R² Score (Test)", f"{pred_results.get('test_r2', 0):.4f}")
    m2.metric("MAE", f"{pred_results.get('test_mae', 0):.6f}")
    m3.metric("RMSE", f"{pred_results.get('test_rmse', 0):.6f}")
    m4.metric("MAPE", f"{pred_results.get('test_mape', 0):.2f}%")
    m5.metric("R² (Train)", f"{pred_results.get('train_r2', 0):.4f}")

    info1, info2, info3 = st.columns(3)
    info1.write(f"📊 Training Samples: **{pred_results.get('train_samples', 0):,}**")
    info2.write(f"🧪 Test Samples: **{pred_results.get('test_samples', 0):,}**")
    info3.write(f"🔧 Features Used: **{pred_results.get('feature_count', 0)}**")

    # --- Actual vs Predicted Chart ---
    pred_records = pred_results.get("predictions", [])
    if pred_records:
        pred_df = pd.DataFrame(pred_records)
        pred_df["Timestamp"] = pd.to_datetime(pred_df["Timestamp"], errors="coerce")

        # Date filter
        pred_df["_date"] = pred_df["Timestamp"].dt.date.astype(str)
        date_opts = ["All"] + sorted(pred_df["_date"].dropna().unique().tolist())
        sel_date = st.selectbox("Filter Data by Date", options=date_opts, key="pred_date")
        if sel_date != "All":
            pred_df = pred_df[pred_df["_date"] == sel_date]

        st.subheader("Actual vs Predicted Carbon Emissions")

        server_opts = ["All"] + sorted(pred_df["Server_ID"].dropna().unique().tolist())
        selected = st.selectbox("Filter by Server", server_opts, key="pred_server_filter")
        plot_df = pred_df if selected == "All" else pred_df[pred_df["Server_ID"] == selected]

        fig_compare = go.Figure()
        fig_compare.add_trace(
            go.Scatter(
                x=plot_df["Timestamp"],
                y=plot_df["actual_carbon_kg"],
                mode="lines",
                name="Actual",
                line={"color": "#58a6ff", "width": 2},
            )
        )
        fig_compare.add_trace(
            go.Scatter(
                x=plot_df["Timestamp"],
                y=plot_df["predicted_carbon_kg"],
                mode="lines",
                name="Predicted",
                line={"color": "#39d353", "width": 2, "dash": "dash"},
            )
        )
        fig_compare.update_layout(
            title="Actual vs Predicted Carbon (kg CO₂)",
            xaxis_title="Timestamp",
            yaxis_title="Carbon (kg CO₂)",
            **PLOTLY_DARK_TEMPLATE["layout"],
        )
        st.plotly_chart(fig_compare, use_container_width=True)

        # --- Scatter: Actual vs Predicted ---
        left, right = st.columns(2)
        with left:
            fig_scatter = px.scatter(
                plot_df,
                x="actual_carbon_kg",
                y="predicted_carbon_kg",
                color="Server_ID",
                title="Actual vs Predicted (Scatter)",
                opacity=0.6,
            )
            # Perfect prediction line
            min_val = float(min(plot_df["actual_carbon_kg"].min(), plot_df["predicted_carbon_kg"].min()))
            max_val = float(max(plot_df["actual_carbon_kg"].max(), plot_df["predicted_carbon_kg"].max()))
            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="Perfect Prediction",
                    line={"color": "#f85149", "dash": "dot"},
                )
            )
            fig_scatter.update_layout(
                **PLOTLY_DARK_TEMPLATE["layout"],
                xaxis_title="Actual Carbon Emissions (kg CO₂)",
                yaxis_title="Predicted Carbon Emissions (kg CO₂)",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with right:
            # Error distribution
            fig_error = px.histogram(
                plot_df,
                x="error",
                nbins=40,
                color="Server_ID",
                title="Prediction Error Distribution",
                opacity=0.75,
            )
            fig_error.add_vline(x=0, line_dash="dash", line_color="#f85149")
            fig_error.update_layout(
                **PLOTLY_DARK_TEMPLATE["layout"],
                xaxis_title="Prediction Error (Actual - Predicted)",
                yaxis_title="Frequency",
            )
            st.plotly_chart(fig_error, use_container_width=True)

    # --- Feature Importance ---
    st.subheader("Feature Importance (Top 15)")
    feat_imp = pred_results.get("feature_importance", [])
    if feat_imp:
        imp_df = pd.DataFrame(feat_imp)
        imp_df = imp_df.sort_values("importance", ascending=True)

        fig_imp = px.bar(
            imp_df,
            x="importance",
            y="feature",
            orientation="h",
            title="Random Forest Feature Importance",
            color="importance",
            color_continuous_scale=["#161b22", "#39d353"],
        )
        fig_imp.update_layout(
            **PLOTLY_DARK_TEMPLATE["layout"],
            showlegend=False,
            coloraxis_showscale=False,
            height=500,
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # --- Model Explanation ---
    with st.expander("How the Prediction Model Works"):
        st.markdown(
            """
            **Algorithm**: Random Forest Regressor (100 trees, max_depth=15)

            **Type**: Supervised Learning (Regression)

            **Target Variable**: `carbon_kg` — the carbon emissions per reading

            **Features Engineered**:
            - **Time features**: hour_of_day, day_of_week, is_weekend, cyclical hour encoding (sin/cos)
            - **Lag features**: Power and carbon values from 1, 3, 6, 12 readings ago
            - **Rolling statistics**: Mean and std of power over 6, 12, 24 reading windows
            - **Rate of change**: Power difference and percentage change between readings
            - **Cumulative energy**: Daily running total of energy per server

            **Train/Test Split**: 80/20 (time-ordered — last 20% is the test set)

            **Metrics**:
            - **R² Score**: How well the model explains variance (1.0 = perfect)
            - **MAE**: Average absolute error in kg CO₂
            - **RMSE**: Root mean squared error (penalizes large errors)
            - **MAPE**: Mean absolute percentage error
            """
        )

    # --- Prediction Table ---
    if pred_records:
        st.subheader("Prediction Details")
        show_df = pred_df[["Timestamp", "Server_ID", "actual_carbon_kg", "predicted_carbon_kg", "error", "abs_error"]]
        show_df = show_df.sort_values("Timestamp", ascending=False).head(50)
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        csv_bytes = pred_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Predictions CSV",
            data=csv_bytes,
            file_name="greenpulse_predictions.csv",
            mime="text/csv",
        )
