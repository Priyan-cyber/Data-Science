import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

data = pd.read_excel('World_development_mesurement.xlsx')
    
for col in data.select_dtypes(include=['object']).columns:
        if data[col].str.contains('%').any():
            data[col] = data[col].str.rstrip('%').astype('float') / 100



def kmeans_clustering(data, n_clusters):
    scaler = MinMaxScaler()
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

    
    imputer = SimpleImputer(strategy='mean') 
    data_imputed = data.copy()
    data_imputed[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    
    scaled_data = scaler.fit_transform(data_imputed[numeric_cols])
    data_scaled = pd.DataFrame(scaled_data, columns=numeric_cols)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data_scaled['Cluster'] = kmeans.fit_predict(data_scaled)
    
    return data_scaled, kmeans

st.title('KMeans Clustering with MinMaxScaler')
st.sidebar.title('Upload Data')

uploaded_file = st.sidebar.file_uploader("Choose a excel file", type="xlsx")


    
st.write("### Data Preview")
st.write(data.head())
    
n_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=3)
    
if st.button('Run KMeans Clustering'):
        data_with_clusters, kmeans = kmeans_clustering(data, n_clusters)
        
        st.write(f"### Data with {n_clusters} Clusters")
        st.write(data_with_clusters.head())
        
        fig, ax = plt.subplots()
        plt.scatter(data_with_clusters.iloc[:, 0], data_with_clusters.iloc[:, 1], c=data_with_clusters['Cluster'], cmap='viridis')
        plt.title('KMeans Clustering')
        plt.xlabel(data.columns[0])
        plt.ylabel(data.columns[1])
        st.pyplot(fig)

        st.write("### Cluster Centers")
        st.write(kmeans.cluster_centers_)
