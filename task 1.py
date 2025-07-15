import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os

def load_data(path):
    """Load CSV data into a DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found: {path}")
    return pd.read_csv(path)

def split_features_target(df, target_column):
    """Split the DataFrame into features (X) and target (y)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

def build_preprocessing_pipeline(X, cardinality_threshold=100):
    """Build preprocessing pipeline for numeric and low-cardinality categorical data."""
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    all_categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_features = [col for col in all_categorical if X[col].nunique() < cardinality_threshold]

    print(f"✅ Using {len(categorical_features)} categorical columns (filtered from {len(all_categorical)}):")
    print(categorical_features)

    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))  # sparse output by default
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ])

    return preprocessor

def save_data(X_train, X_test, y_train, y_test, output_dir="processed_data"):
    """Save processed data to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save sparse matrices or convert to dense if necessary
    np.save(os.path.join(output_dir, "X_train.npy"), X_train.toarray() if hasattr(X_train, "toarray") else X_train)
    np.save(os.path.join(output_dir, "X_test.npy"), X_test.toarray() if hasattr(X_test, "toarray") else X_test)
    
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    
    print(f"✅ Data saved in folder: '{output_dir}'")

def run_pipeline(csv_path, target_column, test_size=0.2, random_state=42):
    """Run the full preprocessing pipeline."""
    df = load_data(csv_path)
    X, y = split_features_target(df, target_column)
    preprocessor = build_preprocessing_pipeline(X)
    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=test_size, random_state=random_state
    )

    print("✅ Data preprocessing complete.")
    print(f"🔹 Training set shape: {X_train.shape}")
    print(f"🔹 Test set shape: {X_test.shape}")

    save_data(X_train, X_test, y_train, y_test)
    return X_train, X_test, y_train, y_test

# Entry point
if __name__ == "__main__":
    csv_path = r"C:\Users\viswa\OneDrive\Downloads\annual-enterprise-survey-2023-financial-year-provisional.csv"
    target_column = "Variable_category"  # ✅ Update to your chosen target column
    run_pipeline(csv_path, target_column)
