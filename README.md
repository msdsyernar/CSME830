# CSME830: Foundation of Data Science - House Price Predictoin in the US

This repository contains the project for **CSME830: Foundation of Data Science**. Analyzing and predicting housing prices across the United States using 2.2M+ property records.
## 📁 Project Structure

```
├── data/
│   ├── realtor-data.csv (raw)
│   └── train_set.csv (processed)
│   └── test_set.csv
│   └── secondary.csv (before cleaning it was additional.csv)
├── notebooks/
│   ├── main.ipynb 
│   ├── secondary.ipynb
│   └── split.ipynb
├── streamlit/
│   └── app.py
└── README.md
```

---
##  Why This Dataset?

**Chose this dataset because:**
- **Real-world relevance**: Housing prices affect everyone 
- **Rich features**: Location, size, temporal data
- **Large scale**: 2.2M entries for robust ML training
- **Complex patterns**: Interesting relationships to discover
- **Practical value**: Results applicable to real estate decisions

---

## 🔍 What I've Learned from EDA

### Key Discoveries:

**1. Three Types of Missing Data:**
- **MNAR**: `prev_sold_date` (30%), `acre_lot` (14%) - Missing = informative
- **MAR**: `house_size` (6%), `bath` (3%), `bed` (1%) - Incomplete listings
- **MCAR**: Location variables (<0.1%) - Random errors

**2. Major Insight: Outliers Distorted Everything!**
```
Before cleaning: Mean house_size = 6 475 sqft (inflated!)
After cleaning:  Mean house_size = 1 641 sqft (realistic)

Problem: Found values up to 1 BILLION sqft (impossible)
Solution: Removed properties >50,000 sqft
```

**3. Property Type Patterns:**
- Properties **without** acre_lot = Smaller (1641 sqft) = Condos/apartments
- Properties **with** acre_lot = Larger (2113 sqft) = Single-family homes
- Missing acre_lot is **informative**, not random!

**4. Data Quality Reality:**
- Started: 2,226,382 rows
- After cleaning: ~1,400,000 rows (37% removed)
- Removed: Land sales, incomplete listings, extreme outliers

---

## 🛠️ Preprocessing Steps Completed

### ✅ Phase 1: Initial Cleaning
- Dropped `brokered_by` and `street` (privacy-encoded, not useful)
- Removed 1,237 rows with missing `price` (target variable)
- Removed 1,364 rows with missing location data (<0.1%)

### ✅ Phase 2: Outlier Removal
- Identified and removed 800K+ land sales (missing house_size, bed, bath)
- Removed extreme outliers (house_size > 50,000 sqft)
- Focused on actual residential properties

### ✅ Phase 3: Missing Value Imputation

**MNAR features (kept as-is):**
```python
# Created binary features instead of imputing
df['has_prev_sale'] = df['prev_sold_date'].notna().astype(int)
df['has_individual_lot'] = (df['acre_lot'] > 0).astype(int)
```

**MAR features (used Linear Regression):**
```python
# Imputed bed, bath, house_size using other features
# Why Linear Regression? KNN took 1.5 hour+, this took 2 minutes
# Captures relationships between features
```

### ✅ Phase 4: Feature Engineering
```python
# Created new features
- has_prev_sale (binary)
```

### ✅ Phase 5: Validation
- Verified no missing values in critical features
- Checked distributions before/after imputation
- Confirmed no unrealistic values introduced

---

## 🎨 Streamlit Progress

### What I've Built:


**Current Features:**
- Interactive data explorer
- Missing value visualizations
- Price distribution charts
- Property comparison tool

**Planned:**
- Real-time price prediction
- Interactive map
- What-if analysis tool
- Model performance dashboard

---

## 🎓 Biggest Lessons

1. **Always check medians vs means** - Outliers can mislead!
2. **Missing ≠ Always bad** - Sometimes it's informative (MNAR)
3. **Data cleaning is 80% of the work** 
4. **Domain knowledge matters** - Understood condos have no lots
5. **Speed matters** - KNN took more than an hour, Linear Regression took 2min

---

## 🔮 Next Steps

- Build baseline models (Linear Regression, Random Forest)
- Feature selection and engineering
- Hyperparameter tuning
- Deploy final model
- Complete Streamlit dashboard

---

**Last Updated:** October 19, 2025  
**Status:** ✅ EDA Complete | 🚧 Modeling In Progress