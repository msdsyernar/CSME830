### Streamlit app for House Price Prediction in the USA
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="House Price EDA - USA",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 House Price Prediction - USA")
st.markdown("### Exploratory Data Analysis Dashboard")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Overview", "Missing Values Analysis", "Data Preprocessing"])

# Function to load data
@st.cache_data
def load_data():
    try:
        data = pd.read_csv('train_set.csv')
        return data
    except FileNotFoundError:
        st.error("❌ Error: 'train_set.csv' file not found. Please upload your data file.")
        return None

# Function to calculate missing values
def missing_report(df):
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing_values, 
        'Percentage': missing_percentage.round(2)
    })
    return missing_data.sort_values(by="Percentage", ascending=False)

# Load data
data = load_data()

if data is not None:
    df = data.copy()
    
    # Dataset Overview Page
    if page == "Dataset Overview":
        st.header("📊 Dataset Overview")
        
        # About Dataset
        with st.expander("ℹ️ About Dataset", expanded=True):
            st.markdown("""
            This dataset contains Real Estate listings in the US broken by State and zip code.
            
            **Content:**
            - **brokered_by**: Categorically encoded agency/broker
            - **status**: Housing status (ready for sale or ready to build)
            - **price**: Housing price (current listing or recently sold)
            - **bed**: Number of beds
            - **bath**: Number of bathrooms
            - **acre_lot**: Property/Land size in acres
            - **street**: Categorically encoded street address
            - **city**: City name
            - **state**: State name
            - **zip_code**: Postal code
            - **house_size**: House area/size in square feet
            - **prev_sold_date**: Previously sold date
            
            **Note:** *brokered_by* and *street* addresses were categorically encoded due to data privacy policy.
            *acre_lot* means total land area, and *house_size* denotes living space/building area.
            """)
        
        # Dataset shape
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{data.shape[0]:,}")
        with col2:
            st.metric("Total Columns", data.shape[1])
        with col3:
            st.metric("Memory Usage", f"{data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        # Display first few rows
        st.subheader("📋 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Dataset info
        st.subheader("🔍 Dataset Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Data Types:**")
            dtype_df = pd.DataFrame({
                'Column': df.dtypes.index,
                'Data Type': df.dtypes.values
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        with col2:
            st.write("**Statistical Summary:**")
            st.dataframe(df.describe(), use_container_width=True)
        
        # Drop brokered_by and street
        st.subheader("🗑️ Dropping Encoded Columns")
        st.info("Dropping 'brokered_by' and 'street' columns as they were encoded for privacy and don't provide meaningful insights.")
        
        if st.button("Drop Encoded Columns"):
            df = df.drop(columns=['brokered_by', 'street'])
            st.success(f"✅ Columns dropped! New shape: {df.shape}")
            st.write(f"Memory reduced from {data.memory_usage(deep=True).sum() / 1024**2:.1f} MB to {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Missing Values Analysis Page
    elif page == "Missing Values Analysis":
        st.header("🔎 Missing Values Analysis")
        
        # Drop encoded columns for analysis
        if 'brokered_by' in df.columns:
            df = df.drop(columns=['brokered_by', 'street'])
        
        # Missing values report
        st.subheader("📊 Missing Values Report")
        missing_df = missing_report(df)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.dataframe(missing_df, use_container_width=True)
        
        with col2:
            # Visualization
            fig = px.bar(
                missing_df[missing_df['Percentage'] > 0],
                x=missing_df[missing_df['Percentage'] > 0].index,
                y='Percentage',
                title='Missing Values Percentage by Column',
                labels={'x': 'Columns', 'Percentage': 'Missing %'},
                color='Percentage',
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Previous Sold Date Analysis
        st.subheader("📅 Previous Sold Date Analysis")
        df['has_prev_sale'] = np.where(df['prev_sold_date'].isnull(), 0, 1)
        
        col1, col2 = st.columns(2)
        with col1:
            status_prev = df.groupby('status')['has_prev_sale'].value_counts(normalize=True).unstack()
            st.write("**Missing Pattern by Status:**")
            st.dataframe(status_prev, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                df, 
                x='status', 
                color=df['has_prev_sale'].astype(str),
                barmode='group',
                title='Previous Sale History by Status',
                labels={'color': 'Has Previous Sale'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Insight:** 50.9% of properties "for sale" and 100% of "ready_to_build" properties 
        have no previous sale date. This is expected as they are new listings or constructions.
        """)
        
        # Missing Value Overlap Analysis
        st.subheader("🔄 Missing Value Overlap Analysis")
        
        bed_missing = df['bed'].isnull()
        bath_missing = df['bath'].isnull()
        house_size_missing = df['house_size'].isnull()
        
        combinations = {
            'None missing': (~bed_missing & ~bath_missing & ~house_size_missing).sum(),
            'Only bed missing': (bed_missing & ~bath_missing & ~house_size_missing).sum(),
            'Only bath missing': (~bed_missing & bath_missing & ~house_size_missing).sum(),
            'Only house_size missing': (~bed_missing & ~bath_missing & house_size_missing).sum(),
            'Bed and bath missing': (bed_missing & bath_missing & ~house_size_missing).sum(),
            'Bed and house_size missing': (bed_missing & ~bath_missing & house_size_missing).sum(),
            'Bath and house_size missing': (~bed_missing & bath_missing & house_size_missing).sum(),
            'All missing': (bed_missing & bath_missing & house_size_missing).sum()
        }
        
        colors_detailed = ['#2ecc71', '#f39c12', '#e67e22', '#3498db', 
                          '#e74c3c', '#c0392b', '#e74c3c', '#8e44ad']
        
        fig = go.Figure(data=[go.Pie(
            labels=list(combinations.keys()),
            values=list(combinations.values()),
            marker=dict(colors=colors_detailed),
            textinfo='label+percent',
            textposition='auto',
            hole=0.4,
            pull=[0, 0, 0, 0, 0.05, 0.05, 0.05, 0.15]
        )])
        
        fig.update_layout(
            title='Missing Value Pattern Distribution',
            showlegend=True,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation Matrix
        st.subheader("🔗 Missing Value Correlation Matrix")
        critical_cols = ['bed', 'bath', 'house_size', 'acre_lot', 'prev_sold_date']
        missing_indicators = df[critical_cols].isnull().astype(int)
        corr_matrix = missing_indicators.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdYlGn_r',
            zmid=0.5,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Missing Value Correlation',
            xaxis_title='Features',
            yaxis_title='Features',
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Data Preprocessing Page
    elif page == "Data Preprocessing":
        st.header("⚙️ Data Preprocessing")
        
        # Drop encoded columns
        if 'brokered_by' in df.columns:
            df = df.drop(columns=['brokered_by', 'street'])
        
        st.subheader("🧹 Handling Missing Values")
        
        # Step 1: Drop rows with critical missing values
        st.write("**Step 1: Dropping rows with missing zip_code, state, city, and price**")
        original_shape = df.shape[0]
        df = df.dropna(subset=['zip_code', 'state', 'city', 'price'])
        dropped_rows = original_shape - df.shape[0]
        st.success(f"✅ Dropped {dropped_rows:,} rows. Remaining: {df.shape[0]:,} rows")
        
        # Step 2: Create has_prev_sale feature
        st.write("**Step 2: Creating 'has_prev_sale' feature**")
        df['has_prev_sale'] = np.where(df['prev_sold_date'].isnull(), 0, 1)
        st.success("✅ New binary feature created to indicate if property has previous sale history")
        
        # Step 3: Handle properties with missing house characteristics
        st.write("**Step 3: Handling properties with missing house_size, bed, and bath**")
        
        for_sale = df['status'].isin(['for_sale', 'sold'])
        for_sale_missing = df[for_sale & df['house_size'].isnull()]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Properties with missing house_size (for_sale/sold)", 
                     f"{len(for_sale_missing):,}")
        with col2:
            st.metric("Percentage of dataset", 
                     f"{(len(for_sale_missing)/len(df)*100):.2f}%")
        
        if st.button("Drop Land-Only Listings"):
            df = df.drop(for_sale_missing.index)
            st.success(f"✅ Dropped {len(for_sale_missing):,} land-only listings. New shape: {df.shape}")
            
            # Show updated missing values
            st.write("**Updated Missing Values Report:**")
            st.dataframe(missing_report(df), use_container_width=True)
        
        # Final dataset summary
        st.subheader("📈 Cleaned Dataset Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        with col4:
            st.metric("Completeness", f"{(1 - df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.1f}%")
        
        # Download processed data
        st.subheader("💾 Download Processed Data")
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Cleaned Dataset",
            data=csv,
            file_name="cleaned_house_data.csv",
            mime="text/csv"
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Data Source:** realtor.com  
**Project:** House Price Prediction  
**Status:** EDA Phase
""")