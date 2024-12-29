import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import datetime as dt
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the dataset
@st.cache_data
def load_data():
    train = pd.read_csv(r'intel.csv')
    train['Date'] = pd.to_datetime(train['Date'], utc=True)
    train.set_index('Date', inplace=True)
    return train

train = load_data()

# Streamlit Title
st.title("Intel Stock Data Analysis 1980-2024")

# Show basic info
st.subheader('Dataset Overview')
st.write(f"Dataset shape: {train.shape}")
st.write(train.describe())
st.write(train.info())

# Missing values
st.subheader("Missing Values")
missing_values = train.isnull().sum()
st.write(missing_values)

# Feature Engineering
train['Daily Return'] = train['Close'].pct_change() * 100
train['7-Day MA'] = train['Close'].rolling(window=7).mean()
train['30-Day MA'] = train['Close'].rolling(window=30).mean()

# Fill missing values for moving averages
train['7-Day MA'] = train['7-Day MA'].ffill().bfill()
train['30-Day MA'] = train['30-Day MA'].ffill().bfill()
train['Daily Return'] = train['Daily Return'].fillna(0)

# Normalize the data
scaler = MinMaxScaler()
columns_to_scale = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily Return', '7-Day MA', '30-Day MA']
train_scaled = train.copy()
train_scaled[columns_to_scale] = scaler.fit_transform(train[columns_to_scale])

# Visualization of the data
st.subheader("Normalized Close Price Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_scaled.index, train_scaled["Close"], label="Close Price")
ax.set_title('Normalized Close Price Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Close Price')
ax.legend()
st.pyplot(fig)

# Visualization of daily returns
st.subheader("Normalized Daily Returns Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_scaled.index, train_scaled['Daily Return'], label='Daily Returns', color='orange', linewidth=1)
ax.set_title('Normalized Daily Returns Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Daily Return')
plt.legend()
plt.grid()
st.pyplot(fig)

# Moving Averages Visualization
st.subheader("7-Day and 30-Day Moving Averages Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_scaled.index, train_scaled['7-Day MA'], label='7-Day Moving Average', color='green', linewidth=1)
ax.plot(train_scaled.index, train_scaled['30-Day MA'], label='30-Day Moving Average', color='red', linewidth=1)
ax.set_title('7-Day and 30-Day Moving Averages Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Moving Averages')
plt.legend()
plt.grid()
st.pyplot(fig)

# Trading volume visualization
st.subheader("Normalized Volume Over Time")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_scaled.index, train_scaled['Volume'], label='Volume', color='purple', linewidth=1)
ax.set_title('Normalized Volume Over Time')
ax.set_xlabel('Date')
ax.set_ylabel('Normalized Volume')
ax.legend()
ax.grid()
st.pyplot(fig)

# Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(train_scaled.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
ax.set_title('Feature Correlation Heatmap')
st.pyplot(fig)

# Histogram of Daily Returns
st.subheader("Histogram of Daily Returns")
fig, ax = plt.subplots(figsize=(8, 6))
ax.hist(train_scaled['Daily Return'], bins=50, color='teal', edgecolor='black')
ax.set_title('Histogram of Daily Returns')
ax.set_xlabel('Normalized Daily Return')
ax.set_ylabel('Frequency')
ax.grid()
st.pyplot(fig)

# # Seasonal Decomposition
# st.subheader("Seasonal Decomposition of Close Price")
# result = seasonal_decompose(train['Close'], model='multiplicative', period=365)
# plt.figure(figsize=(10, 8))
# result.plot()
# plt.tight_layout()
# st.pyplot()



# Add a link## to the Kaggle Notebook
st.markdown('[View the full Code](https://github.com/abdmankhan/DS-intel-1980-24/edit/main/intelJPNcopy.py)')

