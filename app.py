import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="House Price Prediction - USA",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='background: linear-gradient(90deg, #1f77b4 0%, #9467bd 100%); 
                padding: 2rem; border-radius: 15px; margin-bottom: 2rem;'>
        <h1 style='color: white; margin: 0;'>🏠 House Price Prediction - USA</h1>
        <p style='color: #e0e0e0; margin-top: 0.5rem; font-size: 1.1rem;'>
            Real Estate Data Analysis & Preprocessing Pipeline
        </p>
        <div style='background: rgba(255,255,255,0.2); padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
            <p style='color: white; margin: 0;'><strong>Dataset:</strong> 2,226,382 real estate listings across the United States</p>
            <p style='color: white; margin: 0.5rem 0 0 0;'><strong>Source:</strong> realtor.com - Data collected from property listings</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/home.png", width=80)
    st.title("Navigation")
    st.markdown("---")
    st.info("This dashboard presents a comprehensive analysis of the house price prediction dataset.")
    st.markdown("---")
    st.markdown("### About the Dataset")
    st.markdown("""
    - **Total Entries:** 2,226,382
    - **Features:** 10
    - **States Covered:** 51
    - **Memory Usage:** 135+ MB
    """)

# Load data option
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload your cleaned CSV (optional)", type=['csv'])

# If user uploads data, use it; otherwise use sample data
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Data loaded successfully!")
    data_loaded = True
else:
    st.sidebar.info("Using sample data for visualizations")
    data_loaded = False
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    df = pd.DataFrame({
        'price': np.random.lognormal(12, 0.8, n_samples),
        'bed': np.random.randint(1, 6, n_samples),
        'bath': np.random.randint(1, 5, n_samples),
        'acre_lot': np.random.exponential(0.5, n_samples),
        'house_size': np.random.normal(2000, 800, n_samples),
        'state': np.random.choice(['California', 'Texas', 'Florida', 'New York', 'Pennsylvania'], n_samples),
        'city': np.random.choice(['Los Angeles', 'Houston', 'Miami', 'New York City', 'Philadelphia'], n_samples),
        'status': np.random.choice(['for_sale', 'sold', 'ready_to_build'], n_samples),
        'has_prev_sale': np.random.choice([0, 1], n_samples)
    })

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "⚠️ Missing Values", "📈 Analysis", "💡 Key Insights"])

# ==================== TAB 1: DATA OVERVIEW ====================
with tab1:
    st.header("Dataset Overview")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        padding: 20px; border-radius: 10px; text-align: center;'>
                <h3 style='color: white; margin: 0;'>2.2M</h3>
                <p style='color: #e0e0e0; margin: 0;'>Total Entries</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                        padding: 20px; border-radius: 10px; text-align: center;'>
                <h3 style='color: white; margin: 0;'>10</h3>
                <p style='color: #e0e0e0; margin: 0;'>Features</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                        padding: 20px; border-radius: 10px; text-align: center;'>
                <h3 style='color: white; margin: 0;'>51</h3>
                <p style='color: #e0e0e0; margin: 0;'>States</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); 
                        padding: 20px; border-radius: 10px; text-align: center;'>
                <h3 style='color: white; margin: 0;'>135MB</h3>
                <p style='color: #e0e0e0; margin: 0;'>Memory</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Dataset Features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background: #e3f2fd; padding: 1.5rem; border-radius: 10px; 
                        border-left: 5px solid #1f77b4;'>
                <h3 style='color: #1f77b4;'>📋 Dataset Features (Part 1)</h3>
                <ul style='color: #333;'>
                    <li><strong>status:</strong> Housing status (for sale / ready to build)</li>
                    <li><strong>price:</strong> Current listing or recently sold price</li>
                    <li><strong>bed:</strong> Number of bedrooms</li>
                    <li><strong>bath:</strong> Number of bathrooms</li>
                    <li><strong>acre_lot:</strong> Property/Land size in acres</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: #f3e5f5; padding: 1.5rem; border-radius: 10px; 
                        border-left: 5px solid #9467bd;'>
                <h3 style='color: #9467bd;'>📋 Dataset Features (Part 2)</h3>
                <ul style='color: #333;'>
                    <li><strong>city:</strong> City name</li>
                    <li><strong>state:</strong> State name</li>
                    <li><strong>zip_code:</strong> Postal code</li>
                    <li><strong>house_size:</strong> Living space in square feet</li>
                    <li><strong>prev_sold_date:</strong> Previously sold date</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data Privacy Note
    st.warning("""
    **🔒 Data Privacy Note:**
    
    The columns *brokered_by* and *street* were categorically encoded due to data privacy policy 
    and have been dropped from analysis. This reduced memory usage from 163+ MB to 135+ MB.
    """)
    
    # Key Distinctions
    st.info("""
    **📌 Key Distinctions:**
    
    - **acre_lot:** Total land area of the property
    - **house_size:** Living space/building area only
    """)

# ==================== TAB 2: MISSING VALUES ====================
with tab2:
    st.header("Missing Values Analysis")
    
    st.error("""
    **⚠️ Initial Missing Data Report**
    
    The dataset contained significant missing values that required careful handling based on the nature of missingness.
    """)
    
    # Missing data
    missing_data_initial = pd.DataFrame({
        'Feature': ['prev_sold_date', 'house_size', 'bath', 'bed', 'acre_lot', 
                    'zip_code', 'state', 'city', 'price', 'status'],
        'Percentage': [31.69, 23.47, 20.64, 20.61, 18.35, 0.02, 0.00, 0.00, 0.00, 0.00],
        'Count': [705289, 522542, 459498, 458830, 408471, 445, 0, 0, 0, 0]
    })
    
    # Bar chart
    fig = px.bar(missing_data_initial, 
                 x='Percentage', 
                 y='Feature',
                 orientation='h',
                 title='Missing Value Percentages by Feature',
                 color='Percentage',
                 color_continuous_scale='Reds',
                 text='Percentage')
    fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Missing Value Pattern Distribution (Pie Chart)
    st.subheader("Missing Value Pattern Distribution")
    st.markdown("Interactive breakdown of bed, bath, house_size combinations")
    
    if data_loaded and 'bed' in df.columns and 'bath' in df.columns and 'house_size' in df.columns:
        # Calculate actual missing patterns from your data
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
    else:
        # Actual values from your notebook
        combinations = {
            'None missing': 1274266,
            'Only bed missing': 5793,
            'Only bath missing': 24917,
            'Only house_size missing': 78111,
            'Bed and bath missing': 10286,
            'Bed and house_size missing': 1643,
            'Bath and house_size missing': 6674,
            'All missing': 338275
        }
    
    colors_detailed = ['#2ecc71', '#f39c12', '#e67e22', '#3498db', 
                      '#e74c3c', '#c0392b', '#e74c3c', '#8e44ad']
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=list(combinations.keys()),
        values=list(combinations.values()),
        marker=dict(colors=colors_detailed),
        textinfo='label+percent',
        textposition='auto',
        hole=0.4,
        pull=[0, 0, 0, 0, 0.05, 0.05, 0.05, 0.15]
    )])
    
    total_rows = sum(combinations.values())
    fig_pie.update_layout(
        title={
            'text': '<b>Missing Value Pattern Distribution</b><br>' +
                    f'<sub>Interactive breakdown of bed, bath, house_size combinations</sub>',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        showlegend=True,
        annotations=[dict(
            text=f'Total: {total_rows:,}<br>rows',
            x=0.5, y=0.5,
            font=dict(size=16),
            showarrow=False
        )],
        height=600
    )
    
    st.plotly_chart(fig_pie, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Correlation matrix of missing values
    st.subheader("Missing Value Correlation Matrix")
    st.markdown("Higher values = features tend to be missing together")
    
    # Create sample correlation data
    critical_cols = ['bed', 'bath', 'house_size', 'acre_lot', 'prev_sold_date']
    corr_values = np.array([
    [ 1.00,  0.93,  0.84, -0.15,  0.37],
    [ 0.93,  1.00,  0.82, -0.08,  0.40],
    [ 0.84,  0.82,  1.00, -0.09,  0.33],
    [-0.15, -0.08, -0.09,  1.00,  0.06],
    [ 0.37,  0.40,  0.33,  0.06,  1.00]
])
   
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=corr_values,
        x=critical_cols,
        y=critical_cols,
        colorscale='RdYlGn_r',
        zmid=0.5,
        text=corr_values,
        texttemplate='%{text:.2f}',
        textfont={"size": 14},
        colorbar=dict(title="Correlation"),
        hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig_heatmap.update_layout(
        title='Missing Value Correlation Matrix',
        xaxis_title='Features',
        yaxis_title='Features',
        height=600
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # House Size Missingness by Status
    st.subheader("House Size Missingness by Status")
    
    status_missing_data = pd.DataFrame({
        'Status': ['for_sale', 'for_sale', 'sold', 'sold', 'ready_to_build', 'ready_to_build'],
        'Has house_size': ['Missing', 'Present', 'Missing', 'Present', 'Missing', 'Present'],
        'Count': [35000, 720000, 90000, 560000, 0, 19807]
    })
    # sold 90k jok, 560k bar; for sale 350k jok, 720k bar, ready 19807
    
    fig_status = px.bar(status_missing_data, 
                       x='Status', 
                       y='Count',
                       color='Has house_size',
                       title='House Size Missingness by Status',
                       barmode='group',
                       color_discrete_map={'Missing': '#e74c3c', 'Present': '#2ecc71'})
    
    fig_status.update_layout(height=400)
    st.plotly_chart(fig_status, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Detailed explanations
    st.subheader("Missingness Analysis by Feature")
    
    # prev_sold_date
    with st.expander("🟣 prev_sold_date (31.69% missing) - MNAR", expanded=True):
        st.markdown("""
        **Finding:** 50.4% of "for_sale" properties and 100% of "ready_to_build" properties had missing prev_sold_date.
        
        **Reason:** These are new constructions that have never been sold before.
        
        **Solution:** Created binary feature `has_prev_sale` (1 = previously sold, 0 = new property) 
        instead of imputing fake dates.
        """)
    
    # house_size, bed, bath
    with st.expander("🔵 house_size, bed, bath (20-23% missing) - MAR", expanded=True):
        st.markdown("""
        **Finding:** Over 90% of missing house_size also had missing bed and bath. 
        24% were from "for_sale" or "sold" properties.
        
        **Reason:** These are likely land-only listings without structures.
        
        **Solution:** 
        - Dropped rows where all three were missing (land-only listings)
        - Imputed remaining values using Linear Regression based on available features 
          (price, state, city, acre_lot)
        """)
    
    # acre_lot
    with st.expander("🟢 acre_lot (18.35% missing) - MNAR", expanded=True):
        st.markdown("""
        **Finding:** Properties without acre_lot had smaller house sizes (mean: ~1,800 sqft) 
        compared to those with lot info (mean: ~2,400 sqft).
        
        **Reason:** These are likely condos/apartments in urban areas where lot size isn't applicable.
        
        **Solution:** Imputed with state median for properties without lot information. 
        This preserves the systematic difference between single-family homes and condos/apartments.
        """)
        
        # Boxplot comparison
        st.markdown("**Comparison: Properties with vs without acre_lot information**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Price Distribution**")
            fig_price_box = go.Figure()
            
            # Simulate data for boxplot
            with_lot_price = np.random.lognormal(12.5, 0.6, 500)
            without_lot_price = np.random.lognormal(12.3, 0.5, 500)
            
            fig_price_box.add_trace(go.Box(y=np.log10(with_lot_price), name='Has acre_lot', marker_color='#2ecc71'))
            fig_price_box.add_trace(go.Box(y=np.log10(without_lot_price), name='No acre_lot', marker_color='#e74c3c'))
            
            fig_price_box.update_layout(
                yaxis_title='Log(Price)',
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig_price_box, use_container_width=True)
        
        with col2:
            st.markdown("**House Size Distribution**")
            fig_size_box = go.Figure()
            
            # Simulate data for boxplot
            with_lot_size = np.random.normal(2400, 800, 500)
            without_lot_size = np.random.normal(1800, 600, 500)
            
            fig_size_box.add_trace(go.Box(y=np.log10(with_lot_size), name='Has acre_lot', marker_color='#2ecc71'))
            fig_size_box.add_trace(go.Box(y=np.log10(without_lot_size), name='No acre_lot', marker_color='#e74c3c'))
            
            fig_size_box.update_layout(
                yaxis_title='Log(House Size)',
                showlegend=True,
                height=400
            )
            st.plotly_chart(fig_size_box, use_container_width=True)
    
    # Other features
    with st.expander("🟡 zip_code, city, price (0-0.02% missing) - MCAR", expanded=False):
        st.markdown("""
        **Solution:** Dropped these minimal missing values as they represent random errors 
        and are critical for analysis.
        """)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data Cleaning Steps
    st.success("""
    **✅ Data Cleaning Steps Summary**
    
    1. Removed duplicates
    2. Dropped rows with missing zip_code, state, city, and price
    3. Created has_prev_sale binary feature
    4. Removed extreme outliers (house_size > 50,000 sqft)
    5. Dropped land-only listings (missing house_size, bed, and bath)
    6. Imputed acre_lot with state median
    7. Imputed bed, bath, house_size using Linear Regression
    8. Post-processed: rounded bed/bath, clipped values to reasonable ranges
    """)

# ==================== TAB 3: ANALYSIS ====================
with tab3:
    st.header("Exploratory Data Analysis")
    
    # Price Distribution
    st.subheader("📊 Step 3.0: Price Distribution")
    
    fig_price_dist = go.Figure()
    
    # Create histogram with KDE
    price_log = np.log1p(df['price'])
    fig_price_dist.add_trace(go.Histogram(
        x=price_log,
        nbinsx=50,
        name='Price Distribution',
        marker_color='skyblue',
        opacity=0.7,
        histnorm='probability density'
    ))
    
    # Add KDE line
    from scipy import stats
    kde = stats.gaussian_kde(price_log)
    x_range = np.linspace(price_log.min(), price_log.max(), 100)
    fig_price_dist.add_trace(go.Scatter(
        x=x_range,
        y=kde(x_range),
        mode='lines',
        name='KDE',
        line=dict(color='red', width=2)
    ))
    
    fig_price_dist.update_layout(
        title='Log-Transformed House Price Distribution',
        xaxis_title='log(price)',
        yaxis_title='Density',
        height=500,
        showlegend=True
    )
    
    st.plotly_chart(fig_price_dist, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Top expensive states
    st.subheader("📍 Step 3.1: Where are the expensive houses in the US?")
    
    if data_loaded:
        state_avg = df.groupby('state')['price'].mean().sort_values(ascending=False).head(15)
        top_states_data = pd.DataFrame({
            'State': state_avg.index,
            'Average Price': state_avg.values
        })
    else:
        top_states_data = pd.DataFrame({
            'State': [
                'Hawaii', 'Virgin Islands', 'California', 'New York', 'Montana', 
                'Utah', 'Colorado', 'District of Columbia', 'Wyoming', 'Nevada',
                'Massachusetts', 'Connecticut', 'Washington', 'Idaho', 'Guam'
            ],
            'Average Price': [
                1385356, 1355761, 1076531, 1009949, 933092, 
                917660, 899917, 875530, 766264, 749856,
                747500, 725768, 703298, 682064, 656807
            ]
        })
    
    fig1 = px.bar(top_states_data, 
                  x='State', 
                  y='Average Price',
                  title='Top 15 Most Expensive States by Average House Price',
                  color='Average Price',
                  color_continuous_scale='Blues')
    fig1.update_layout(height=500, showlegend=False, xaxis_tickangle=-45)
    fig1.update_traces(texttemplate='$%{y:,.0f}', textposition='outside')
    st.plotly_chart(fig1, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Log-Price Distribution by State (Boxplot)
    st.subheader("Log-Price Distribution by State (Top 15 States)")
    
    if data_loaded and 'state' in df.columns:
        top_states = df['state'].value_counts().head(15).index
        df_top = df[df['state'].isin(top_states)]
        
        fig_box_states = px.box(df_top, 
                               x='state', 
                               y=np.log1p(df_top['price']),
                               title='',
                               color='state')
        fig_box_states.update_layout(
            height=500,
            xaxis_tickangle=-45,
            yaxis_title='log(price)',
            xaxis_title='State',
            showlegend=False
        )
        st.plotly_chart(fig_box_states, use_container_width=True)
    else:
        # Sample visualization
        sample_states = ['California', 'Texas', 'Florida', 'New York', 'Pennsylvania']
        sample_data = []
        for state in sample_states:
            prices = np.random.lognormal(12 + np.random.rand(), 0.6, 100)
            for p in prices:
                sample_data.append({'State': state, 'Price': p})
        
        df_sample = pd.DataFrame(sample_data)
        fig_box_states = px.box(df_sample, x='State', y=np.log1p(df_sample['Price']),
                               color='State')
        fig_box_states.update_layout(height=500, yaxis_title='log(price)', showlegend=False)
        st.plotly_chart(fig_box_states, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Correlation Heatmap
    st.subheader("🔥 Step 3.2: Correlation Heat map")
    
    if data_loaded:
        num_cols = ['price', 'bed', 'bath', 'acre_lot', 'house_size', 'has_prev_sale']
        available_cols = [col for col in num_cols if col in df.columns]
        corr = df[available_cols].corr()
    else:
        num_cols = ['price', 'bed', 'bath', 'acre_lot', 'house_size', 'has_prev_sale']
        corr_values = np.array([
            [ 1.00,  0.19,  0.33,  0.01,  0.14, -0.02], # price
            [ 0.19,  1.00,  0.67,  0.00,  0.34, -0.03], # bed
            [ 0.33,  0.67,  1.00, -0.00,  0.43, -0.04], # bath
            [ 0.01,  0.00, -0.00,  1.00,  0.00, -0.00], # acre_lot
            [ 0.14,  0.34,  0.43,  0.00,  1.00, -0.03], # house_size
            [-0.02, -0.03, -0.04, -0.00, -0.03,  1.00]  # has_prev_sale
        ])
        corr = pd.DataFrame(corr_values, columns=num_cols, index=num_cols)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    corr_masked = corr.where(~mask)
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_masked.values,
        x=corr_masked.columns,
        y=corr_masked.index,
        colorscale='RdBu_r',
        zmid=0,
        text=corr_masked.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        colorbar=dict(title="Correlation"),
        hovertemplate='%{y} × %{x}<br>Correlation: %{z:.3f}<extra></extra>'
    ))
    
    fig_corr.update_layout(
        title='Correlation Heatmap of Housing Features',
        height=600,
        xaxis=dict(side='bottom'),
        yaxis=dict(autorange='reversed')
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature correlations with price
    st.info("""
    **📊 Feature Correlations with Price**
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="bath × bed", value="0.67", delta="Moderate positive")
    
    with col2:
        st.metric(label="bath × price", value="0.32", delta="Weak positive")
    
    with col3:
        st.metric(label="house × bath", value="0.43", delta="Moderate positive")
    
    with col4:
        st.metric(label="house size × bed", value="0.34", delta="Weak positive")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Price per sqft
    st.subheader("💰 Estimated Price per Square Foot by State")
    
    price_sqft_data = pd.DataFrame({
        'State': ['Arizona', 'California', 'Washington', 'Florida', 'Virginia',
                'Texas', 'Minnesota', 'New York', 'Maryland', 'Ohio',
                'Pennsylvania', 'New Jersey', 'Illinois', 'Georgia', 'North Carolina'],
                'Price per SqFt': [405.14, 401.21, 261.07, 253.90, 203.31,
                            148.55, 144.14, 72.49, 56.28, 27.23,
                            27.00, 25.41, 22.16, 18.33, 15.50]
        })
    
    fig2 = px.bar(price_sqft_data, 
                  y='State', 
                  x='Price per SqFt',
                  orientation='h',
                  title='',
                  color='Price per SqFt',
                  color_continuous_scale='Reds')
    fig2.update_layout(height=600, showlegend=False)
    fig2.update_traces(texttemplate='$%{x}/sqft', textposition='outside')
    st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Simpson's Paradox
    st.warning("""
    **🎯 Simpson's Paradox in New York**
    
    Although New York appears to have lower average prices compared to states like Hawaii or California, 
    this is misleading due to aggregation across **1,719 cities**.
    
    **Reality:** New York City itself has significantly higher prices than most other states, 
    but averaging across all cities in New York state (including rural areas) brings the state average 
    down considerably.
    
    **Example:** New York has 1,719 cities, and the average price for NYC is way higher than in other states.
               
               The same with the correlation. To define the true correlation, we should do more sensitive analysis. E.g. make a segment, differentiate the states etc.
    """)

# ==================== TAB 4: KEY INSIGHTS ====================
with tab4:
    st.header("Key Insights & Findings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); 
                        padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;
                        border-left: 5px solid #1f77b4;'>
                <h4 style='color: #1565c0; margin-top: 0;'>✅ Missingness Types Identified</h4>
                <p style='color: #333; margin-bottom: 0;'>
                Successfully categorized missing data: MNAR (prev_sold_date, acre_lot), 
                MAR (house_size, bed, bath), and MCAR (zip_code, city, price).
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%); 
                        padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;
                        border-left: 5px solid #9c27b0;'>
                <h4 style='color: #7b1fa2; margin-top: 0;'>✅ Property Type Distinction</h4>
                <p style='color: #333; margin-bottom: 0;'>
                Identified systematic differences between single-family homes and condos/apartments 
                based on acre_lot missingness patterns.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: linear-gradient(135deg, #ffebee 0%, #ffcdd2 100%); 
                        padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;
                        border-left: 5px solid #f44336;'>
                <h4 style='color: #c62828; margin-top: 0;'>✅ Geographic Price Variations</h4>
                <p style='color: #333; margin-bottom: 0;'>
                Hawaii, California, and Massachusetts show highest price per square foot, 
                indicating strong regional economic factors.
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); 
                        padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;
                        border-left: 5px solid #4caf50;'>
                <h4 style='color: #2e7d32; margin-top: 0;'>✅ Smart Feature Engineering</h4>
                <p style='color: #333; margin-bottom: 0;'>
                Created has_prev_sale binary feature to capture new vs. resale properties, 
                preserving valuable information instead of dropping or imputing.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%); 
                        padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;
                        border-left: 5px solid #ff9800;'>
                <h4 style='color: #e65100; margin-top: 0;'>✅ Land-Only Listings Removed</h4>
                <p style='color: #333; margin-bottom: 0;'>
                Removed properties with missing house_size, bed, and bath (land-only), 
                ensuring the model predicts house prices, not land prices.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: linear-gradient(135deg, #fff9c4 0%, #fff59d 100%); 
                        padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;
                        border-left: 5px solid #fbc02d;'>
                <h4 style='color: #f57f17; margin-top: 0;'>✅ Correlation Insights</h4>
                <p style='color: #333; margin-bottom: 0;'>
                Bath and bedrooms show strongest correlation with price
                </p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Data Quality Achievements
    st.markdown("""
        <div style='background: linear-gradient(135deg, #e8eaf6 0%, #c5cae9 100%); 
                    padding: 2rem; border-radius: 15px; border: 2px solid #5c6bc0;'>
            <h3 style='color: #3f51b5; text-align: center; margin-top: 0;'>
                🎯 Data Quality Achievements
            </h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem;'>
                <h2 style='color: #5c6bc0; margin: 0;'>163 → 135</h2>
                <p style='color: #666; margin: 0.5rem 0 0 0;'>MB Memory Saved</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem;'>
                <h2 style='color: #9c27b0; margin: 0;'>2.2M+</h2>
                <p style='color: #666; margin: 0.5rem 0 0 0;'>Records Processed</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='text-align: center; padding: 1.5rem;'>
                <h2 style='color: #e91e63; margin: 0;'>0%</h2>
                <p style='color: #666; margin: 0.5rem 0 0 0;'>Missing Values Remaining</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Summary of Missingness
    st.subheader("📋 Summary of Missingness Types")
    
    missingness_summary = pd.DataFrame({
        'Feature': ['prev_sold_date', 'acre_lot', 'house_size', 'bath', 'bed', 'zip_code', 'price', 'city'],
        'Missingness Type': ['MNAR', 'MNAR', 'MAR', 'MAR', 'MAR', 'MCAR', 'MCAR', 'MCAR'],
        'Initial %': [31.69, 18.35, 23.47, 20.64, 20.61, 0.02, 0.00, 0.00],
        'Solution': [
            'Created has_prev_sale feature',
            'Imputed with state median',
            'Linear Regression imputation',
            'Linear Regression imputation',
            'Linear Regression imputation',
            'Dropped rows',
            'Dropped rows',
            'Dropped rows'
        ]
    })
    
    # Create color mapping for missingness types
    def color_missingness(val):
        if val == 'MNAR':
            return 'background-color: #e1bee7'
        elif val == 'MAR':
            return 'background-color: #bbdefb'
        elif val == 'MCAR':
            return 'background-color: #c8e6c9'
        return ''
    
    st.dataframe(
        missingness_summary.style.applymap(color_missingness, subset=['Missingness Type']),
        use_container_width=True,
        height=350
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Next Steps
    st.success("""
    **🚀 Next Steps for Modeling**
    
    1. Feature scaling and normalization for numerical features
    2. Encoding categorical variables (state, city, status)
    3. Train-test split with stratification by state
    4. Model selection and hyperparameter tuning
    5. Cross-validation and performance evaluation
    """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>Data Source: realtor.com | Dataset: 2,226,382 Real Estate Listings | Analysis Pipeline</p>
        <p style='font-size: 0.9rem; margin-top: 0.5rem;'>
            Built with Streamlit 🎈 | Data Analysis & Visualization Dashboard
        </p>
    </div>
""", unsafe_allow_html=True)