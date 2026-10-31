from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def train_model(train, feature_columns):
    X_train = train[feature_columns]
    y_train = train["target"].astype(int)

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    model.fit(X_train, y_train)

    baseline_probability = y_train.mean()

    return model, baseline_probability