import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def convert_sqft_to_num(x):
    """Convert sqft string ranges to numerical values"""
    if isinstance(x, str):
        tokens = x.split('-')
        if len(tokens) == 2:
            try:
                return (float(tokens[0]) + float(tokens[1])) / 2
            except ValueError:
                return None
        try:
            return float(x)
        except ValueError:
            return None
    return x


def remove_bhk_outliers(df):
    """Remove outliers based on BHK comparison within locations"""
    exclude_indices = np.array([])

    for location, location_df in df.groupby('location'):
        bhk_stats = {}

        # Collect BHK statistics for this location
        for bhk, bhk_df in location_df.groupby('size'):
            bhk_stats[bhk] = {
                'mean': bhk_df.price_per_sqft.mean(),
                'std': bhk_df.price_per_sqft.std(),
                'count': bhk_df.shape[0]
            }

        # Compare higher BHK with lower BHK
        for bhk, bhk_df in location_df.groupby('size'):
            stats = bhk_stats.get(bhk - 1)

            if stats and stats['count'] > 5:
                exclude_indices = np.append(
                    exclude_indices,
                    bhk_df[bhk_df.price_per_sqft < stats['mean']].index.values
                )

    return df.drop(exclude_indices, axis='index')


def build_pipeline(num_attribs, cat_attribs):
    """Build preprocessing pipeline for numerical and categorical features"""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    
    return full_pipeline


if not os.path.exists(MODEL_FILE):
    # TRAINING PHASE
    print("Starting training phase...")
    
    # Load and preprocess data
    housing_data = pd.read_csv("Bengaluru_House_Data2.csv")
    housing_data = housing_data.drop(['area_type', 'balcony'], axis=1)
    
    # Convert total_sqft to numerical
    housing_data["total_sqft"] = housing_data["total_sqft"].apply(convert_sqft_to_num)
    
    # Convert size (BHK) to numerical
    housing_data['size'] = housing_data['size'].apply(
        lambda x: int(x.split(' ')[0]) if isinstance(x, str) else x
    )
    
    
    # Create price categories for stratified split
    housing_data["price_cat"] = pd.cut(
        housing_data["price"],
        bins=[0, 30, 60, 90, 200, np.inf],
        labels=[1, 2, 3, 4, 5]
    )
    
    # Remove rows with missing values and reset index
    housing_data = housing_data.dropna(subset=['total_sqft', 'size']).reset_index(drop=True)
    
    # Stratified split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing_data, housing_data["price_cat"]):
        housing_data.loc[test_index].drop("price_cat", axis=1).to_csv("input.csv", index=False)
        strat_train_set = housing_data.loc[train_index].drop("price_cat", axis=1)
    
    # Scale price to actual value
    strat_train_set["price"] = strat_train_set["price"] * 100000
    
    # Calculate price per sqft for outlier removal
    strat_train_set["price_per_sqft"] = (
        strat_train_set["price"] / strat_train_set["total_sqft"]
    )
    
    # Remove price per sqft outliers
    strat_train_set = strat_train_set[
        (strat_train_set.price_per_sqft > 500) &
        (strat_train_set.price_per_sqft < 30000)
    ]
    
    # Remove BHK outliers
    strat_train_set = remove_bhk_outliers(strat_train_set)
    
    # Prepare features and labels
    housing_features = strat_train_set.drop(["price", "price_per_sqft"], axis=1)
    housing_labels = strat_train_set["price"].copy()
    
    # Define numerical and categorical attributes
    num_attribs = [col for col in housing_features.columns if col != 'location']
    cat_attribs = ["location"]
    
    # Build and fit pipeline
    pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = pipeline.fit_transform(housing_features)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(housing_prepared, housing_labels)
    
    # Save model and pipeline
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    
    print("Model trained and saved successfully!")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Pipeline saved to: {PIPELINE_FILE}")

else:
    # INFERENCE PHASE
    print("Loading saved model and pipeline...")
    
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)
    
    # Load test data
    input_data = pd.read_csv("input.csv")
    
    # Preprocess test data (same conversions as training)
    input_data["total_sqft"] = input_data["total_sqft"].apply(convert_sqft_to_num)
    input_data['size'] = input_data['size'].apply(
        lambda x: int(x.split(' ')[0]) if isinstance(x, str) else x
    )
    
    # Transform and predict
    transformed_input = pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    
    # Add predictions to output
    input_data["predicted_price"] = predictions
    
    # Save results
    input_data.to_csv("output.csv", index=False)
    
    print("Inference complete!")
    print(f"Results saved to: output.csv")
    print(f"Number of predictions: {len(predictions)}")
