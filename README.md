# Developing deep LSTMs with later temporal attention for predicting COVID-19 severity,clinical outcome, and antibody level byscreening serological indicators over time

- **Objective：**
    This repo holds code for [Developing deep LSTMs with later temporal attention for predicting COVID-19 severity,clinical outcome, and antibody level byscreening serological indicators over time
]()
    This work discusses screening serologic indicators over time to predict COVID-19 disease severity, clinical outcomes, and Spike antibody levels. The clinical course, as well as the immunological reaction to COVID-19, is notable for its extreme variability. Identifying the main associated factors might help understand the variability, disease progression, and physiological status in patients. Deriving the dynamic changes of the antibody against Spike protein is crucial for understanding the immune response.

- **Methods:**
    This work explores critical serological indicators and combines them with deep Long Short Term Memory (LSTM) time series models to accurately predict COVID-19 severity, clinical progression, and antibody prediction. We use feature selection techniques to filter feature subsets highly correlated with the target. Then, we propose two temporal deep learning models to predict disease severity and clinical outcome. Moreover, We also use ensemble and temporal deep learning models to predict the Spike antibody level.
- **Results:**  
    In disease severity prediction, The LSTM model
has the highest classification accuracy. In clinical outcome prediction, the Temporal Attention Long Short Term Memory (TA-LSTM) model has the highest accuracy classification. In Spike antibody level prediction, the XGBoost model
has the highest R2 value using non-time series; The LSTM model has the highest R2 value using the time series model.
- **Discussion:**
    In conclusion, the significance of our work is threefold. 
  Firstly, we provide high-risk factors of disease
severity and clinical outcome and reveal clinical characteristics highly correlated with the dynamic changes in the Spike antibody level. Secondly, we introduce the attention mechanism into the temporal deep learning model
for clinical outcome prediction, demonstrating the temporal
attention (TA) block’s effectiveness in enhancing global
temporal dependencies. Finally, the proposed models can provide a computer-aided medical diagnostics system to facilitate developing countries during this pandemic.

----------

1. All the files are in the All figures and screenshots in the paper folder
2. All codes are in the COVID-19 Prediction Code folder
3. All experimental data is in the COVID-19.xlsx
