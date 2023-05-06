import streamlit as st
import pandas as pd
import numpy as np
from pickle import load
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Diamond Price App",
    page_icon="💎",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

df = pd.read_csv("diamonds.csv")

label_encoder = load(open('label_encoder.pkl', 'rb'))
scaler = load(open('standard_scaler.pkl', 'rb'))
knn_regressor = load(open("knn_model.pkl", "rb"))
image = Image.open('Diamond.jpg')

# st.markdown(”This text is :red[colored red], and this is **:blue[colored]** and bold.”)

st.title(":blue[Diamond Price Predictor] :gem:")
st.image(image)
st.subheader(':black[Enter Diamond Details]')

Carat = st.text_input("Carat", placeholder="Enter Diamond Carat: ")
Depth_Percent = st.text_input("Depth Percent", placeholder="Enter Diamond DepthPercent: ")
Table = st.text_input("Table", placeholder="Enter Diamond Table")

Cut = st.selectbox("Cut", df['cut'].unique())
Color = st.selectbox("Color", df['color'].unique())
Clarity = st.selectbox("Clarity", df['clarity'].unique())

Length = st.text_input("Lenght", placeholder="Enter Diamond Lenght")
Width = st.text_input("Width", placeholder="Enter Diamond Width")
Depth = st.text_input("Depth", placeholder="Enter Diamond Depth")

btn_click = st.button("Predict")

if btn_click == True:
    if Carat and Depth_Percent and Table and Cut and Color and Clarity and Length and Width and Depth:
        # cat_data = [[Cut, Color, Clarity]]
        # num_data = [[Carat, Depth_Percent, Table, Length, Width, Depth]]

        diamond_details = np.array([float(Carat), float(Depth_Percent), float(Table), float(Length), float(Width), float(Depth)]).reshape(1, -1)
        lable_encoder_cols = np.array([Cut, Color, Clarity])

        diamond_details_transformed = scaler.transform(diamond_details)
        
        lable_encoder_cols_le = label_encoder.fit_transform(lable_encoder_cols)
        lable_encoder_cols = lable_encoder_cols_le.reshape(1, -1)
        query_point = np.concatenate((diamond_details_transformed, lable_encoder_cols), axis = 1)

        pred = knn_regressor.predict(query_point)
        
        st.success('The predicted price of selecte Diamond is : $ ' + str(float((pred)[0])))
    else:
        st.error("Enter the values properly.")
