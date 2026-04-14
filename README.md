# GreenPulse
Carbon Footprint Analysis and Optimization System for Data Center Servers using Machine Learning

Overview

GreenPulse is a full-stack software system designed to analyze, monitor, and reduce carbon emissions from data center servers using machine learning and data analytics.


Problem

Data centers consume massive electricity, leading to significant CO₂ emissions. However:
    •    Carbon impact is invisible
    •    No real-time monitoring exists
    •    No optimization based on energy cleanliness
    •    No prediction of future emissions


Solution

GreenPulse solves this by providing:
    •     Real-time carbon footprint calculation
    •     Anomaly detection using Isolation Forest & DBSCAN
    •     Workload shifting for carbon optimization
    •     Carbon emission prediction using ML


Tech Stack
    •    Python
    •    Pandas, NumPy
    •    Scikit-learn
    •    Streamlit
    •    MongoDB


Features
    •    Per-server carbon tracking
    •    Real-time alerts
    •    Interactive dashboard
    •    ML-based anomaly detection
    •    Carbon reduction recommendations
    •    Future emission forecasting


ML Models Used
    •    Isolation Forest
    •    DBSCAN
    •    Gradient Boosting Regressor


How to Run

pip install -r requirements.txt
streamlit run app.py


Dataset
    •    Timestamp
    •    Server ID
    •    Power Usage (Watts)


Conclusion

GreenPulse transforms raw power data into a complete carbon intelligence system, making emissions visible, predictable, and optimizable.
