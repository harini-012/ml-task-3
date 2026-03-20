import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error,r2_score
st.title("🚗 MPG Prediction using Polynomial Regression")
st.write("Predict fuel efficiency based on car features")
@st.cache_data
def load_data():
    df=pd.read_csv("auto-mpg.csv")
    return df 
df=load_data()

df.columns=df.columns.str.strip() 
df['horsepower'] = pd.to_numeric(df['horsepower'], errors='coerce')
st.subheader("Data Set preview")
st.write(df.head())
st.subheader("📊 Dataset Statistics")
st.write(df.describe())

st.title("🚗 MPG Prediction using Polynomial Regression")
st.write("Predict fuel efficiency based on car features")
features=['displacement','horsepower','weight','acceleration']
target='mpg'
X=df[features]
y=df[target]
X=X.fillna(X.mean())
y=y.fillna(y.mean())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
poly=PolynomialFeatures(degree=2)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)
model=LinearRegression()
model.fit(X_train_poly,y_train)
y_pred=model.predict(X_test_poly)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"**MSE:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")
st.subheader("📉 Actual vs Predicted MPG")

fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel("Actual MPG")
ax.set_ylabel("Predicted MPG")
st.pyplot(fig)
st.subheader("Predict MPG")
displacement = st.slider("Displacement", 50.0, 500.0, 200.0)
horsepower = st.slider("Horsepower", 40.0, 250.0, 100.0)
weight = st.slider("Weight", 1500.0, 5000.0, 3000.0)
acceleration = st.slider("Acceleration", 5.0, 25.0, 15.0)
input_data=np.array([[displacement,horsepower,weight,acceleration]])
input_poly=poly.transform(input_data)
prediction=model.predict(input_poly)
st.success(f" Predicted MPG: {prediction[0]:.2f}")
