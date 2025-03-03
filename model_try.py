import streamlit as st
import pandas as pd
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

import s3fs
import zipfile
from datetime import datetime
import holidays
import plotly.express as px

# ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# ---------------------------
# 1) Load & Preprocess Data
# ---------------------------
@st.cache_data
def load_data():
    # Update these to match your S3 bucket and file path
    s3_bucket = "streamlittest0303"  # e.g., "my-awesome-bucket"
    s3_key = "BOSNYC_EDA_cleaned.parquet.zip"  # Path to the zip file in S3

    s3_path = f"s3://{s3_bucket}/{s3_key}"

    # Create a filesystem object for S3
    fs = s3fs.S3FileSystem()

    # Read the zip file directly from S3 into memory
    with fs.open(s3_path, "rb") as f:
        with zipfile.ZipFile(f, "r") as zip_ref:
            zip_ref.extractall("extracted_data")

    # Now read the Parquet file that was extracted
    df = pd.read_parquet("extracted_data/BOSNYC_EDA_cleaned.parquet")
    df['shop_date'] = pd.to_datetime(df['shop_date'])
    df['depdate'] = pd.to_datetime(df['depdate'])
    return df

df = load_data()

# Create holiday indicator outside cache to avoid hashing issues
us_holidays = holidays.US()
df['holiday'] = df['shop_date'].apply(lambda x: 1 if x in us_holidays else 0)

# Additional date-based features
df['week_day_shop'] = df['shop_date'].dt.dayofweek
df['week_day_dep'] = df['depdate'].dt.dayofweek
df['shop_month'] = df['shop_date'].dt.month
df['dep_month'] = df['depdate'].dt.month

# One-hot encode AP_group so we can selectively choose which groups to include
if "AP_group" in df.columns:
    df = pd.get_dummies(df, columns=["AP_group"], prefix="AP_group", dummy_na=False)

# Identify one-hot encoded AP_group columns
ap_group_features = sorted([col for col in df.columns if col.startswith("AP_group_")])
st.write("Loaded data shape:", df.shape)

# ---------------------------
# 2) Split Data (Train/Val/Test)
# ---------------------------
train_df = df[df['shop_date'] < '2024-10-01']
val_df = df[(df['shop_date'] >= '2024-10-01') & (df['shop_date'] <= '2024-12-31')]
test_df = df[df['shop_date'].dt.year == 2025]

st.write("Training set:", train_df.shape)
st.write("Validation set:", val_df.shape)
st.write("Test set:", test_df.shape)

# ---------------------------
# 3) Sidebar Controls
# ---------------------------
st.sidebar.title("Dashboard Controls")

# 3.1) cheapestfarenumcnx Filter
st.sidebar.header("CheapestFarenumcnx Filter")
cheapestfarenumcnx_option = st.sidebar.selectbox(
    "Select cheapestfarenumcnx (optional)",
    options=["All", "0", "1"],
    index=0
)

# 3.2) Optional Feature Selection
st.sidebar.header("Optional Feature Selection")
base_optional_features = [
    "ap", 
    "cheapestfarenumcnx", 
    "week_day_shop", 
    "week_day_dep", 
    "shop_month",
    "dep_month", 
    "holiday"
]
all_optional_features = base_optional_features + ap_group_features
selected_optional_features = st.sidebar.multiselect(
    "Select additional features",
    options=all_optional_features,
    default=[]
)

# 3.3) Random Forest Parameters
st.sidebar.header("Random Forest Parameters")
st.sidebar.markdown(
"""
**n_estimators**: Number of trees. More trees can improve performance but increase computation time.  
**max_depth**: Maximum depth of trees. Deeper trees capture more details but may overfit.  
**min_samples_split**: Minimum samples needed to split a node. Larger values make the tree more conservative.  
**min_samples_leaf**: Minimum samples required at a leaf node. Higher values lead to smoother predictions.
"""
)
rf_params = {
    "n_estimators": st.sidebar.slider("Number of Estimators", 50, 500, 100, step=10),
    "max_depth": st.sidebar.slider("Max Depth (0 for None)", 0, 15, 0),
    "min_samples_split": st.sidebar.slider("Min Samples Split", 2, 10, 2),
    "min_samples_leaf": st.sidebar.slider("Min Samples Leaf", 1, 10, 1)
}

# ---------------------------
# 4) Data Filtering & Aggregation
# ---------------------------
def filter_by_cheapestfarenumcnx(df, option):
    if option == "0":
        return df[df["cheapestfarenumcnx"] == 0]
    elif option == "1":
        return df[df["cheapestfarenumcnx"] == 1]
    else:
        return df

def preprocess_data(df, mode, selected_features):
    keys = ["shop_date", "depdate"]
    features = keys.copy()
    if mode == "filtered" and "cheapestfarenumcnx" not in features:
        if "cheapestfarenumcnx" in df.columns:
            features.append("cheapestfarenumcnx")
    for feat in selected_features:
        if feat not in features:
            features.append(feat)
    if mode == "aggregated":
        agg_dict = {"CheapestFareAmt_per_pax": "min"}
        for feat in features:
            if feat not in keys and feat in df.columns:
                agg_dict[feat] = "first"
        df_agg = df.groupby(keys).agg(agg_dict).reset_index()
        X = df_agg[features]
        y = df_agg["CheapestFareAmt_per_pax"]
    else:
        X = df[features]
        y = df["CheapestFareAmt_per_pax"]
    return X, y

if cheapestfarenumcnx_option in ["0", "1"]:
    mode = "filtered"
else:
    mode = "aggregated"

if mode == "filtered":
    train_df_mode = filter_by_cheapestfarenumcnx(train_df, cheapestfarenumcnx_option)
    val_df_mode = filter_by_cheapestfarenumcnx(val_df, cheapestfarenumcnx_option)
    test_df_mode = filter_by_cheapestfarenumcnx(test_df, cheapestfarenumcnx_option)
else:
    train_df_mode = train_df.copy()
    val_df_mode = val_df.copy()
    test_df_mode = test_df.copy()

X_train, y_train = preprocess_data(train_df_mode, mode, selected_optional_features)
X_val, y_val = preprocess_data(val_df_mode, mode, selected_optional_features)
X_test, y_test = preprocess_data(test_df_mode, mode, selected_optional_features)

st.write("Training features shape:", X_train.shape)
st.write("Validation features shape:", X_val.shape)
st.write("Test features shape:", X_test.shape)

# ---------------------------
# 5) Convert Date Features -> Ordinals
# ---------------------------
def convert_dates(X):
    for col in ["shop_date", "depdate"]:
        if col in X.columns:
            X[col + "_ord"] = X[col].apply(lambda x: x.toordinal() if pd.notnull(x) else np.nan)
            X.drop(columns=[col], inplace=True)
    return X

X_train = convert_dates(X_train.copy())
X_val = convert_dates(X_val.copy())
X_test = convert_dates(X_test.copy())

# ---------------------------
# 6) Fill NAs
# ---------------------------
X_train.fillna(0, inplace=True)
X_val.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# ---------------------------
# 7) Model Training, Evaluation & Progress Bars
# ---------------------------
if st.sidebar.button("Train Model"):
    st.header("Training Random Forest Model")

    # First progress bar: simulate training (0% to 50%)
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_steps = 50
    for i in range(total_steps):
        time.sleep(0.02)
        progress_bar.progress(i + 1)
        status_text.text(f"Training progress: {i + 1}%")
    
    # Initialize Random Forest
    model = RandomForestRegressor(
        n_estimators=rf_params["n_estimators"],
        max_depth=rf_params["max_depth"] if rf_params["max_depth"] != 0 else None,
        min_samples_split=rf_params["min_samples_split"],
        min_samples_leaf=rf_params["min_samples_leaf"],
        random_state=42
    )
    
    # Fit model on training set
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_val_pred = model.predict(X_val)
    rmse_val = mean_squared_error(y_val, y_val_pred) ** 0.5
    r2_val = r2_score(y_val, y_val_pred)
    mape_val = mean_absolute_percentage_error(y_val, y_val_pred) * 100.0

    st.subheader("Validation Metrics")
    st.write("RMSE:", round(rmse_val, 1))
    st.write("R²:", round(r2_val, 1))
    st.write("MAPE:", f"{round(mape_val, 1)}%")
    
    # Second progress bar: final evaluation (simulate from 50% to 100%)
    progress_bar2 = st.progress(50)
    status_text2 = st.empty()
    total_steps2 = 50
    for i in range(total_steps2):
        time.sleep(0.02)
        progress_val = 50 + int((i + 1) * 50 / total_steps2)
        progress_bar2.progress(progress_val)
        status_text2.text(f"Final evaluation progress: {progress_val}%")
    
    # Combine train+val for final training
    X_train_val = pd.concat([X_train, X_val], axis=0)
    y_train_val = pd.concat([y_train, y_val], axis=0)
    model.fit(X_train_val, y_train_val)
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test)
    rmse_test = mean_squared_error(y_test, y_test_pred) ** 0.5
    r2_test = r2_score(y_test, y_test_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred) * 100.0

    st.subheader("Test Metrics (Trained on Train + Val)")
    st.write("RMSE:", round(rmse_test, 1))
    st.write("R²:", round(r2_test, 1))
    st.write("MAPE:", f"{round(mape_test, 1)}%")
    
    # ---------------------------
    # 8) Plotly Visualization: Actual vs Prediction on Test Set
    # ---------------------------
    test_results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_test_pred
    }).reset_index(drop=True)
    
    def assign_color(row):
        actual = row["Actual"]
        pred = row["Predicted"]
        if pred >= actual * 0.9 and pred <= actual * 1.1:
            return "green"
        elif pred > actual * 1.1:
            return "red"
        elif pred < actual * 0.9:
            return "blue"
    
    test_results["color"] = test_results.apply(assign_color, axis=1)
    
    total = len(test_results)
    green_count = (test_results["color"] == "green").sum()
    red_count = (test_results["color"] == "red").sum()
    blue_count = (test_results["color"] == "blue").sum()
    perc_green = round(green_count / total * 100, 1)
    perc_red = round(red_count / total * 100, 1)
    perc_blue = round(blue_count / total * 100, 1)
    
    fig = px.scatter(
        test_results,
        x="Actual",
        y="Predicted",
        color="color",
        title="Test Set: Actual vs Predicted",
        labels={"Actual": "Actual CheapestFareAmt_per_pax", "Predicted": "Predicted CheapestFareAmt_per_pax"},
        color_discrete_map={"green": "green", "red": "red", "blue": "blue"}
    )
    min_val = min(test_results["Actual"].min(), test_results["Predicted"].min())
    max_val = max(test_results["Actual"].max(), test_results["Predicted"].max())
    fig.add_shape(
        type="line",
        x0=min_val, y0=min_val, x1=max_val, y1=max_val,
        line=dict(dash="dash", color="gray")
    )
    st.plotly_chart(fig)
    st.write(f"Within 10% (green): {perc_green}% | Over 10% (red): {perc_red}% | Under 10% (blue): {perc_blue}%")
    
    # ---------------------------
    # 9) Feature Importance
    # ---------------------------
    st.subheader("Feature Importance")
    try:
        importances = model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": X_train_val.columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)
        st.dataframe(importance_df)
        st.bar_chart(importance_df.set_index("Feature"))
    except AttributeError:
        st.write("Random Forest does not provide feature importances.")
