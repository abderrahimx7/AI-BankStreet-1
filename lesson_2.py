import pandas as pd 

df = pd.DataFrame({

"close" :[100, 102, 99, 101, 98,
        96, 100, 103, 101, 97,
        90, 98, 100, 102, 99]

})

print(df)
df["daily_return"] = df["close"].pct_change()
print(df)
df["close_in_5_sessions"] = df["close"].shift(-5)

df["future_return_5d"] = (df["close_in_5_sessions"]/df["close"])-1
print(df) 
known_future = df["future_return_5d"].notna()
df["target"] = pd.Series(pd.NA, index=df.index, dtype="Int64"
                    )
df.loc[known_future , "target"] = (
df.loc[known_future , "future_return_5d"] <= -0.03
).astype(int)

print(df.to_string())
df["close_in_5_sessions"] = df["close"].shift(-5)
print(df)
df["future_return_5d"] = (df["close_in_5_sessions"]/df["close"]-1
                           )
known_future = df["future_return_5d"].notna()
df["target"] = pd.Series (
    pd.NA, 
    index=df.index,
    dtype="Int64"
)
df.loc[known_future, "target"] =(
    df.loc[known_future, "future_return_5d"] <= -0.03
).astype(int)
df["momentum"] = (
    df["close"] / df["close"].shift(3) - 1
)
print(df.to_string())

feature_columns = ["daily_return", "momentum"]

training_data = df.dropna(
    subset=feature_columns + ["target"]
).copy()

X = training_data[feature_columns]
y = training_data["target"].astype(int)

print("X - Features: ")
print(X)
print("\ny - Target: ")
print(y)