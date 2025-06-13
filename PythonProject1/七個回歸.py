import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

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

# 單張圖畫圖函數
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

# 每組變數，每個模型各畫一張圖
for x_col, y_col in pairs:
    X = df[[x_col]].values
    y = df[y_col].values
    for name, model in models.items():
        title = f'{name} | {x_col} vs. {y_col}'
        plot_model_prediction_single(X, y, model, x_col, y_col, title)