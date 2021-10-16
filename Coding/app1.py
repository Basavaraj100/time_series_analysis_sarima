# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 00:24:03 2021

@author: HP
"""

# importing libraries

# for data manupulation
import pandas as pd
import numpy as np

#  for visualization
import matplotlib.pyplot as plt
import seaborn

# to manage the directories
import os


#  for decomposing
from statsmodels.tsa.seasonal import seasonal_decompose

# ARIMA and auto ARIMA
from statsmodels.tsa.arima_model import ARIMA

# test for stationarity
from statsmodels.tsa.stattools import adfuller

#  for plotting acf and pacf
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

# statsmodels
import statsmodels.api as sm


#  front end frmalwork
import streamlit as st

#  for loading the model
import pickle
# ===============================================================================



# loading model
model=pickle.load(open(r'champain_prediction.pkl','rb'))

def main():
    st.header('TIME SERIES FORECASTING USING SARIMAX MODEL')
    
    st.subheader('ABOUT')
    st.write('- SARIMAX model used to forcase the Champaign ')
    
    
    with st.beta_expander('Forecast the champaign'):
        starting_ind=st.number_input('Enter starting index',min_value=0,step=1)
        ending_ind=st.number_input('Enter the ending index',min_value=1,step=1)
        
        if st.button('Forecast'):
            prediction=model.predict(start=starting_ind,end=ending_ind)
            fig1=plt.figure()
            plt.plot(model.fittedvalues)
            plt.plot(prediction)
            # figure.plot(model.fittedvalues)
            
            st.pyplot(fig1)
            
            
       
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

if __name__=='__main__':
    main()




















