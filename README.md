# Prediction and attributes analysis of COVID-19 time series by ensemble learning and temporal deep learning models 

- **Objective：** 
    This work explores the prediction and attributes analysis of COVID-19 time series by ensemble learning and temporal deep learning models.
- **Methods:**
    We use feature selection techniques to filter feature subsets highly correlated with the target.
  Then, we propose two temporal deep learning models to predict disease severity and clinical outcome of COVID-19. Moreover, the dynamic changes of the antibody against Spike protein are crucial for understanding the immune response. We also utilize ensemble and temporal deep learning models to predict the Spike antibody level.
- **Results:**  
    In disease severity prediction, using the feature subsets of Hypertension,  LYMPH$\%$, Age, UA, A/G LDH, Diabetes, ALB, and Sex, the LSTM model achieves a classification accuracy of  0.76622. In clinical outcome prediction, using the feature subsets of Mono$\%$, INR, Neu$\#$, ALB, hs-CRP, PLT, Urea, and LDH, the TA-LSTM model achieves a classification accuracy of 0.98855. 
  In Spike antibody level prediction, the proposed XGBoost model demonstrates the value of 0.353 in R2\_Squared using non-time-series data; the proposed LSTM model demonstrates the value of 0.494 in R2\_Squared using time-series data.
- **Discussion:**
    In conclusion, the significance of our work is threefold. 
  Firstly, we provide not only high-risk factors of disease severity and clinical outcome but also reveal clinical characteristics that highly correlate with the dynamic changes in the Spike antibody level. Secondly,  we introduce the attention mechanism into the temporal deep learning model for clinical outcome prediction, demonstrating the temporal attention (TA) block's effectiveness in enhancing global temporal dependencies. Thirdly, the proposed models can provide a computer-aided medical diagnostics system to facilitate developing countries which does not have vaccination facility during this pandemic scenario. 

----------

1. All the files are in the All figures and screenshots in the paper folder
2. All codes are in the COVID-19 Prediction Code folder
3. All experimental data is in the COVID-19.xlsx