import pandas as pd
import xgboost as xgb
import pickle


def train(df, seed=42):
    y = df['sleep_efficiency'].to_numpy()
    X = df.drop('sleep_efficiency', axis=1)

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=50,
        learning_rate=5e-2,
        max_depth=2,
        random_state=seed
    )

    model.fit(X, y)

    return model


if __name__ == "__main__":
    df = pd.read_csv("data/sleep-efficiency_cleaned.csv")

    model = train(df)

    with open("bin/model.pkl", "wb") as f:
        pickle.dump(model, f)
