import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib

def main():
    # Load dataset
    df = pd.read_excel("PROJECT_DATASET.xlsx")

    # Preprocess
    df['datetime_iso'] = pd.to_datetime(df['datetime_iso'], errors='coerce')
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    le = LabelEncoder()
    df['source_type_encoded'] = le.fit_transform(df['source_type'].astype(str))
    le_region = LabelEncoder()
    df['region_encoded'] = le_region.fit_transform(df['region'].astype(str))

    df.fillna(0, inplace=True)

    # Features and target
    feature_cols = [
        'new_residential_units', 'commercial_space_added_sqm', 'population_growth',
        'real_estate_index', 'wind_generation_mw', 'installed_capacity_mw', 'is_holiday',
        'temperature_c', 'humidity_percent', 'radiation_wm2', 'wind_speed_kmph',
        'rainfall_mm', 'power_purchased_mw', 'cost_per_mw', 'industrial_load_mw',
        'commercial_load_mw', 'residential_load_mw', 'solar_generation_mw',
        'solar_irradiance', 'reserve_generation_mw', 'grid_balancing_mw',
        'source_type_encoded', 'region_encoded', 'peak_demand_flag'
    ]

    X = df[feature_cols]
    y = df['total_demand_mw']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "verbosity": 1
    }

    model = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, "eval")], early_stopping_rounds=20)

    y_pred = model.predict(dtest)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"Test RMSE: {rmse:.3f}")

    # Save model and encoders
    joblib.dump(model, "energy_demand_xgb_model.joblib")
    joblib.dump(le, "source_type_encoder.joblib")
    joblib.dump(le_region, "region_encoder.joblib")
    print("Model and encoders saved.")

if __name__ == "__main__":
    main()
