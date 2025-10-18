import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Page config
st.set_page_config(
    page_title="House Price Analysis - USA",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        background-color: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🏠 House Price Prediction - USA")
st.markdown("### Real Estate Analysis and Missing Data Exploration")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Data Overview",
    "🔍 Missing Value Analysis",
    "🧹 Data Preprocessing",
    "📈 Analysis"
])

# Helper Functions
@st.cache_data
def load_data():
    """Load the dataset"""
    df = pd.read_csv('train_set.csv')
    return df

def missing_report(df):
    """Generate missing value report"""
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    missing_data = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage.round(2)
    })
    return missing_data.sort_values(by="Percentage", ascending=False)

# Load data
df = load_data()

# ========================================
# PAGE 1: DATA OVERVIEW
# ========================================
if page == "📊 Data Overview":
    st.header("Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    st.subheader("Dataset Info")
    st.markdown("""
    This dataset contains Real Estate listings in the US broken by State and zip code.
    
    **Features:**
    - `status`: Housing status (ready for sale/ready to build)
    - `price`: Current listing or recently sold price
    - `bed`: Number of bedrooms
    - `bath`: Number of bathrooms
    - `acre_lot`: Property/Land size in acres
    - `city`: City name
    - `state`: State name
    - `zip_code`: Postal code
    - `house_size`: House area in square feet
    - `prev_sold_date`: Previously sold date
    """)
    
    st.subheader("First Few Rows")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.subheader("Data Types")
    type_df = pd.DataFrame({
        'Column': df.dtypes.index,
        'Data Type': df.dtypes.values
    })
    st.dataframe(type_df, use_container_width=True)

# ========================================
# PAGE 2: MISSING VALUE ANALYSIS
# ========================================
elif page == "🔍 Missing Value Analysis":
    st.header("Missing Value Analysis")
    
    # Initial missing report
    st.subheader("Initial Missing Values")
    missing_df = missing_report(df)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.dataframe(missing_df, use_container_width=True)
    
    with col2:
        fig = px.bar(
            missing_df[missing_df['Percentage'] > 0],
            x=missing_df[missing_df['Percentage'] > 0].index,
            y='Percentage',
            title="Missing Value Percentages",
            labels={'x': 'Column', 'Percentage': 'Missing %'},
            color='Percentage',
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Previous Sold Date Analysis
    st.subheader("📅 Previous Sold Date Analysis")
    st.markdown("""
    **Hypothesis:** Missing `prev_sold_date` values are NOT random. They likely represent:
    1. New construction properties (never sold before)
    2. First-time listings
    """)
    
    # Create binary indicator
    df_temp = df.copy()
    df_temp['has_prev_sale'] = np.where(df_temp['prev_sold_date'].isnull(), 0, 1)
    
    # Status breakdown
    status_breakdown = df_temp.groupby('status')['has_prev_sale'].value_counts(normalize=True).unstack()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='No Previous Sale',
        x=status_breakdown.index,
        y=status_breakdown[0] * 100,
        marker_color='lightcoral'
    ))
    fig.add_trace(go.Bar(
        name='Has Previous Sale',
        x=status_breakdown.index,
        y=status_breakdown[1] * 100,
        marker_color='lightgreen'
    ))
    fig.update_layout(
        title='Previous Sale History by Property Status',
        xaxis_title='Status',
        yaxis_title='Percentage',
        barmode='stack',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Key Finding:** 100% of 'ready_to_build' properties have no previous sale history,
    which confirms our hypothesis that missing values are informative (MNAR - Missing Not At Random).
    """)
    
    # Bed, Bath, House Size Analysis
    st.subheader("🛏️ Bed, Bath & House Size Missing Pattern")
    
    # Create overlap analysis
    bed_missing = df_temp['bed'].isnull()
    bath_missing = df_temp['bath'].isnull()
    house_size_missing = df_temp['house_size'].isnull()
    
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
    
    colors = ['#2ecc71', '#f39c12', '#e67e22', '#3498db', 
              '#e74c3c', '#c0392b', '#e74c3c', '#8e44ad']
    
    fig = go.Figure(data=[go.Pie(
        labels=list(combinations.keys()),
        values=list(combinations.values()),
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='auto',
        hole=0.4,
        pull=[0, 0, 0, 0, 0.05, 0.05, 0.05, 0.15]
    )])
    
    fig.update_layout(
        title='Missing Value Pattern Distribution',
        annotations=[dict(
            text=f'Total: {len(df_temp):,}<br>rows',
            x=0.5, y=0.5,
            font=dict(size=16, family='Arial Black'),
            showarrow=False
        )],
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.warning("""
    **Critical Finding:** Over 80% of rows with missing `house_size` also have missing `bed` and `bath`.
    This suggests these properties are likely land-only listings, not actual houses.
    """)
    
    # Correlation Matrix
    st.subheader("🔗 Missing Value Correlation")
    
    critical_cols = ['bed', 'bath', 'house_size', 'acre_lot', 'prev_sold_date']
    missing_indicators = df_temp[critical_cols].isnull().astype(int)
    corr_matrix = missing_indicators.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale='RdYlGn_r',
        zmid=0.5,
        text=corr_matrix.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 14, "family": "Arial Black"},
        colorbar=dict(title="Correlation"),
        hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<br><extra></extra>'
    ))
    
    fig.update_layout(
        title='Missing Value Correlation Matrix',
        xaxis_title='Features',
        yaxis_title='Features',
        height=700
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("""
    **Insight:** High correlation (>0.8) between bed, bath, and house_size missingness
    confirms they are likely missing together (land-only listings).
    """)

# ========================================
# PAGE 3: DATA PREPROCESSING
# ========================================
elif page == "🧹 Data Preprocessing":
    st.header("Data Preprocessing")
    
    # Make a copy for preprocessing
    df_clean = df.copy()
    
    st.subheader("Step 1: Drop Problematic Rows")
    
    # Drop rows where price, city, state, zip_code are missing
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna(subset=['zip_code', 'state', 'city', 'price'])
    dropped_rows = initial_rows - len(df_clean)
    
    st.info(f"✅ Dropped {dropped_rows:,} rows with missing price/location data")
    
    st.subheader("Step 2: Handle prev_sold_date")
    
    # Create has_prev_sale indicator
    df_clean['has_prev_sale'] = np.where(df_clean['prev_sold_date'].isnull(), 0, 1)
    st.success("✅ Created `has_prev_sale` binary indicator instead of imputing")
    
    st.subheader("Step 3: Remove Outliers")
    
    # Remove extreme house_size outliers
    extreme_outliers = df_clean[df_clean['house_size'] > 100000]
    df_clean = df_clean[df_clean['house_size'] <= 100000].copy()
    
    st.info(f"✅ Removed {len(extreme_outliers):,} extreme outliers (house_size > 100,000 sqft)")
    
    st.subheader("Step 4: Identify Land-Only Listings")
    
    # Identify land sales
    for_sale = df_clean['status'].isin(['for_sale', 'sold'])
    land = df_clean[for_sale & df_clean['house_size'].isnull() & 
                     df_clean['bed'].isnull() & df_clean['bath'].isnull()]
    
    df_clean = df_clean.drop(land.index)
    
    st.info(f"✅ Removed {len(land):,} land-only listings")
    
    st.subheader("Step 5: Impute acre_lot")
    
    with st.spinner("Imputing acre_lot..."):
        # State median imputation
        df_clean['acre_lot'] = df_clean.groupby('state')['acre_lot'].transform(
            lambda x: x.fillna(x.median())
        )
        df_clean['acre_lot'].fillna(0, inplace=True)
    
    st.success("✅ Imputed acre_lot with state-level medians")
    
    st.subheader("Step 6: Regression Imputation for bed, bath, house_size")
    
    with st.spinner("Running regression imputation..."):
        # Encode categoricals
        le_state = LabelEncoder()
        le_city = LabelEncoder()
        df_clean['state_encoded'] = le_state.fit_transform(df_clean['state'].astype(str))
        df_clean['city_encoded'] = le_city.fit_transform(df_clean['city'].astype(str))
        
        def fast_regression_impute(df, target, predictors):
            train = df[df[target].notna()]
            predict = df[df[target].isna()]
            
            if len(predict) == 0:
                return df[target]
            
            X_train = train[predictors].fillna(0)
            y_train = train[target]
            X_pred = predict[predictors].fillna(0)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            predictions = model.predict(X_pred)
            
            result = df[target].copy()
            result.loc[predict.index] = predictions
            return result
        
        # Impute
        df_clean['bed'] = fast_regression_impute(df_clean, 'bed', 
            ['bath', 'house_size', 'price', 'state_encoded', 'city_encoded'])
        
        df_clean['bath'] = fast_regression_impute(df_clean, 'bath',
            ['bed', 'house_size', 'price', 'state_encoded', 'city_encoded'])
        
        df_clean['house_size'] = fast_regression_impute(df_clean, 'house_size',
            ['bed', 'bath', 'acre_lot', 'price', 'state_encoded', 'city_encoded'])
        
        # Post-process
        df_clean['bed'] = df_clean['bed'].round().clip(0, 20).astype(int)
        df_clean['bath'] = df_clean['bath'].round().clip(0, 20).astype(int)
        df_clean['house_size'] = df_clean['house_size'].clip(100, 50000)
        
        # Clean up
        df_clean = df_clean.drop(['state_encoded', 'city_encoded'], axis=1)
    
    st.success("✅ Regression imputation complete!")
    
    # Final missing report
    st.subheader("Final Missing Values Report")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        final_missing = missing_report(df_clean)
        st.dataframe(final_missing, use_container_width=True)
    
    with col2:
        remaining_missing = final_missing[final_missing['Percentage'] > 0]
        if len(remaining_missing) > 0:
            fig = px.bar(
                remaining_missing,
                x=remaining_missing.index,
                y='Percentage',
                title="Remaining Missing Values",
                color='Percentage',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No missing values remaining!")
    
    # Download button
    st.subheader("Download Cleaned Data")
    csv = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

# ========================================
# PAGE 4: ANALYSIS
# ========================================
elif page == "📈 Analysis":
    st.header("Exploratory Data Analysis")
    
    # Load cleaned data if available
    try:
        df_analysis = pd.read_csv('cleaned_data.csv')
        st.success("Using cleaned data")
    except:
        df_analysis = df.copy()
        st.warning("Using original data - please run preprocessing first")
    
    # Price distribution
    st.subheader("Price Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.histogram(
            df_analysis,
            x='price',
            nbins=50,
            title='Price Distribution',
            labels={'price': 'Price ($)'},
            color_discrete_sequence=['#3498db']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(
            df_analysis,
            y='price',
            title='Price Box Plot',
            labels={'price': 'Price ($)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Price by state
    st.subheader("Average Price by State")
    
    state_avg = df_analysis.groupby('state')['price'].mean().sort_values(ascending=False).head(15)
    
    fig = px.bar(
        x=state_avg.index,
        y=state_avg.values,
        title='Top 15 States by Average Price',
        labels={'x': 'State', 'y': 'Average Price ($)'},
        color=state_avg.values,
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    
    numeric_cols = ['bed', 'bath', 'house_size', 'acre_lot', 'price']
    corr = df_analysis[numeric_cols].corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale='RdBu',
        text=corr.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title='Correlation Matrix',
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>House Price Analysis Dashboard | Data Science Project</p>
    </div>
    """, unsafe_allow_html=True)