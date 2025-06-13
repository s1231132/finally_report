import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 讀入資料並建立功率欄位
df = pd.read_csv("ai4i2020.csv")
df["Power"] = (2 * np.pi * df["Torque [Nm]"] * df["Rotational speed [rpm]"]) / 60

# 定義模型
models = {
    "Linear": LinearRegression(),
    "Polynomial2": make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.01),
    "ElasticNet": ElasticNet(alpha=0.01, l1_ratio=0.5),
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": make_pipeline(StandardScaler(), SVR(kernel='rbf'))
}

# 定義變數組合
pairs = [
    ("Air temperature [K]", "Process temperature [K]"),
    ("Rotational speed [rpm]", "Torque [Nm]"),
    ("Rotational speed [rpm]", "Power"),
    ("Torque [Nm]", "Power")
]


# 定義 MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-10))) * 100


# 畫圖函數：預測圖
def plot_model_prediction_single(X, y, model, x_label, y_label, title):
    plt.figure(figsize=(6, 4))
    model.fit(X, y)
    y_pred = model.predict(X)
    plt.scatter(X, y, color='blue', alpha=0.3, label='Actual')
    plt.scatter(X, y_pred, color='red', alpha=0.3, label='Predicted')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 畫圖函數：回歸評估指標
def plot_metrics_bar(subset, x_col, y_col):
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Regression Metrics for {x_col} vs. {y_col}", fontsize=16)
    sns.barplot(x="Model", y="R2", data=subset, ax=axs[0, 0])
    axs[0, 0].set_title("R²")
    sns.barplot(x="Model", y="RMSE", data=subset, ax=axs[0, 1])
    axs[0, 1].set_title("RMSE")
    sns.barplot(x="Model", y="MAE", data=subset, ax=axs[1, 0])
    axs[1, 0].set_title("MAE")
    sns.barplot(x="Model", y="MAPE", data=subset, ax=axs[1, 1])
    axs[1, 1].set_title("MAPE")
    for ax in axs.flat:
        ax.tick_params(axis='x', rotation=45)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# 執行主流程
for x_col, y_col in pairs:
    X = df[[x_col]].values
    y = df[y_col].values
    metrics = []

    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        title = f'{name} | {x_col} vs. {y_col}'
        plot_model_prediction_single(X, y, model, x_col, y_col, title)
        metrics.append({
            "Model": name,
            "R2": r2_score(y, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
            "MAE": mean_absolute_error(y, y_pred),
            "MAPE": mean_absolute_percentage_error(y, y_pred)
        })

    metrics_df = pd.DataFrame(metrics)
    plot_metrics_bar(metrics_df, x_col, y_col)