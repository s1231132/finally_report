import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter # 新增的導入
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

# 設定 matplotlib 顯示中文字體 (根據您的作業系統和字體庫可能需要調整)
# 嘗試常用的中文字體設定
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS'] # 備選字體列表
    plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
except Exception as e:
    print(f"設置中文字體時發生錯誤: {e}")
    print("如果圖表中文顯示為方塊，請確保已安裝合適的中文字體，並在上述列表中指定。")


# --- 1. 載入資料 ---
try:
    # 使用使用者提供的 encoding 參數
    df = pd.read_csv('ai4i2020.csv', encoding='utf-8-sig')
except FileNotFoundError:
    print("錯誤：找不到 'ai4i2020.csv' 檔案。請確保檔案與腳本在同一目錄下，或提供正確的路徑。")
    exit()
except Exception as e:
    print(f"讀取 CSV 檔案時發生錯誤: {e}")
    exit()

print("--- 資料集基本資訊 ---")
print(df.head())
print("\n--- 資料集欄位資訊 ---")
df.info()
print("\n--- 資料集描述性統計 (數值型) ---")
print(df.describe())
print("\n--- 資料集描述性統計 (物件型/類別型) ---")
print(df.describe(include=['object']))

# 識別數值型和類別型欄位
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

# 移除 'UDI' 和 'Product ID' 這種唯一識別符，它們通常對分析無用
# UDI 在此資料集中是數值型
if 'UDI' in numerical_cols:
    numerical_cols.remove('UDI')
if 'Product ID' in categorical_cols:
    categorical_cols.remove('Product ID')

print(f"\n識別出的數值型欄位 (用於分析): {numerical_cols}")
print(f"識別出的類別型欄位 (用於分析): {categorical_cols}")


# --- 2. 探索資料的分布情形 ---
print("\n--- 2.1 資料分布可視化 ---")

# 直方圖 (Histograms) for numerical features
print("\n繪製數值型特徵的直方圖...")
if numerical_cols:
    plt.figure(figsize=(15, max(10, 2 * len(numerical_cols)))) # 動態調整高度
    num_plot_cols = 2
    num_plot_rows = (len(numerical_cols) + num_plot_cols - 1) // num_plot_cols
    for i, col in enumerate(numerical_cols):
        plt.subplot(num_plot_rows, num_plot_cols, i + 1)
        sns.histplot(df[col], kde=True)
        plt.title(f'{col} 的分布狀況')
        plt.xlabel(col)
        plt.ylabel('頻率')
    plt.tight_layout()
    plt.show()
else:
    print("沒有可繪製直方圖的數值型欄位。")


# 長條圖 (Bar charts) for categorical features
print("\n繪製類別型特徵的長條圖...")
if categorical_cols:
    plt.figure(figsize=(12, max(6, 3 * len(categorical_cols))))
    for i, col in enumerate(categorical_cols):
        plt.subplot(len(categorical_cols), 1, i + 1)
        sns.countplot(y=df[col], order = df[col].value_counts().index)
        plt.title(f'{col} 的分布狀況')
        plt.xlabel('數量')
        plt.ylabel(col)
    plt.tight_layout()
    plt.show()

    # 使用 Counter 顯示類別計數 (作為範例)
    for col in categorical_cols:
        print(f"\n使用 collections.Counter 統計欄位 '{col}':")
        counts = Counter(df[col])
        for item, count in counts.items():
            print(f"  {item}: {count}")
else:
    print("沒有找到適合繪製長條圖的類別型欄位。")


# 盒狀圖 (Box plots) for numerical features
print("\n繪製數值型特徵的盒狀圖 (用於觀察離群值)...")
if numerical_cols:
    plt.figure(figsize=(15, max(10, 2 * len(numerical_cols))))
    num_plot_cols = 2
    num_plot_rows = (len(numerical_cols) + num_plot_cols - 1) // num_plot_cols
    for i, col in enumerate(numerical_cols):
        plt.subplot(num_plot_rows, num_plot_cols, i + 1)
        sns.boxplot(y=df[col])
        plt.title(f'{col} 的盒狀圖')
        plt.ylabel(col)
    plt.tight_layout()
    plt.show()
else:
    print("沒有可繪製盒狀圖的數值型欄位。")


# 盒狀圖 (按類別分組) - 以 'Type' 和 'Machine failure' 為例
# 'Machine failure' 在此資料集中是數值型 (0或1)
target_col = 'Machine failure' # 假設這是目標變數
group_col = 'Type' # 假設這是主要的類別分組變數

if group_col in categorical_cols and target_col in df.columns and df[target_col].dtype in [np.int64, np.float64]:
    print(f"\n繪製數值型特徵按 '{group_col}' 分組的盒狀圖...")
    for num_col in numerical_cols:
        # 排除已是二元 (0/1) 的故障類型欄位本身，以及目標變數本身
        if num_col not in [target_col, 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=group_col, y=num_col, data=df)
            plt.title(f'{num_col} by {group_col}')
            plt.show()

    print(f"\n繪製數值型特徵按 '{target_col}' 分組的盒狀圖...")
    for num_col in numerical_cols:
        if num_col not in [target_col, 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
            plt.figure(figsize=(8, 5))
            sns.boxplot(x=target_col, y=num_col, data=df)
            plt.title(f'{num_col} by {target_col}')
            plt.show()
else:
    print(f"無法繪製按 '{group_col}' 或 '{target_col}' 分組的盒狀圖，請檢查欄位是否存在且類型正確。")


# --- 3. 分析主要變數之間的相關性 ---
print("\n--- 3.1 變數相關性分析 ---")

# 相關係數矩陣 (Correlation Matrix)
# 包含所有數值型欄位以及二元故障類型欄位
cols_for_corr = numerical_cols[:] # 複製列表
binary_failure_cols = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF', target_col]
for b_col in binary_failure_cols:
    if b_col in df.columns and b_col not in cols_for_corr:
        cols_for_corr.append(b_col)

# 確保所有用於相關性分析的欄位都存在於 DataFrame 中
cols_for_corr = [col for col in cols_for_corr if col in df.columns]

if cols_for_corr:
    correlation_matrix = df[cols_for_corr].corr()
    plt.figure(figsize=(max(10, len(cols_for_corr)*0.8), max(8, len(cols_for_corr)*0.7)))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, annot_kws={"size": 8})
    plt.title('數值型變數相關係數矩陣')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    print("\n相關係數矩陣:")
    print(correlation_matrix)
else:
    print("沒有足夠的數值型欄位來計算相關性矩陣。")


# 散佈圖 (Scatter plots) - 觀察特定變數對的關係
print("\n繪製特定變數間的散佈圖...")
scatter_pairs = [
    ('Air temperature [K]', 'Process temperature [K]'),
    ('Rotational speed [rpm]', 'Torque [Nm]'),
    ('Torque [Nm]', 'Tool wear [min]') # 新增一個觀察對
]

for x_col, y_col in scatter_pairs:
    if x_col in df.columns and y_col in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=x_col, y=y_col, data=df, hue=target_col if target_col in df.columns else None, alpha=0.6)
        plt.title(f'{x_col} vs {y_col}')
        plt.show()
    else:
        print(f"無法繪製散佈圖，因為欄位 '{x_col}' 或 '{y_col}' 不存在。")


# --- 4. 資料清洗與預處理 ---
print("\n--- 4. 資料清洗與預處理 ---")
df_cleaned = df.copy() # 建立一個副本進行清洗

# --- 4.1 處理缺失值 ---
print("\n--- 4.1.1 檢查缺失值 ---")
missing_values = df_cleaned.isnull().sum()
print("各欄位缺失值數量:")
print(missing_values[missing_values > 0])

if missing_values.sum() == 0:
    print("\n資料集中沒有缺失值。")
else:
    print("\n處理缺失值策略:")
    imputer_numerical = SimpleImputer(strategy='median')
    imputer_categorical = SimpleImputer(strategy='most_frequent')

    for col in df_cleaned.columns: # 遍歷所有欄位以防萬一
        if df_cleaned[col].isnull().any():
            if col in numerical_cols: # 使用先前定義的數值欄位列表
                df_cleaned[col] = imputer_numerical.fit_transform(df_cleaned[[col]]).ravel()
                print(f"數值型欄位 '{col}' 的缺失值已使用中位數插補。")
            elif col in categorical_cols: # 使用先前定義的類別欄位列表
                df_cleaned[col] = imputer_categorical.fit_transform(df_cleaned[[col]]).ravel()
                print(f"類別型欄位 '{col}' 的缺失值已使用眾數插補。")
            else: # 如果有欄位未被分類 (例如 UDI, Product ID 如果未被移除)
                if pd.api.types.is_numeric_dtype(df_cleaned[col]):
                    df_cleaned[col] = imputer_numerical.fit_transform(df_cleaned[[col]]).ravel()
                    print(f"未分類但為數值型的欄位 '{col}' 的缺失值已使用中位數插補。")
                else:
                    df_cleaned[col] = imputer_categorical.fit_transform(df_cleaned[[col]]).ravel()
                    print(f"未分類但為物件型的欄位 '{col}' 的缺失值已使用眾數插補。")


    print("\n處理缺失值後，再次檢查:")
    current_missing = df_cleaned.isnull().sum()
    if current_missing.sum() > 0:
        print(current_missing[current_missing > 0])
    else:
        print("所有缺失值已處理完畢。")


# --- 4.2 處理異常值 (Outliers) ---
print("\n--- 4.2.1 檢測異常值 (基於IQR) 並說明處理方式 ---")
# 我們已經透過盒狀圖視覺化檢測了異常值。
# 處理異常值的方法有很多種，這裡我們以IQR方法為例檢測，並說明處理方式。

# 選擇要檢查異常值的數值型欄位 (排除二元欄位和識別碼)
cols_for_outlier_check = [col for col in numerical_cols if col not in [target_col, 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']]

for col in cols_for_outlier_check:
    if col in df_cleaned.columns: # 確保欄位存在
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df_cleaned[(df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)]
        if not outliers.empty:
            print(f"\n欄位 '{col}' 中檢測到 {len(outliers)} 個潛在異常值 (IQR法):")
            print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
            print(f"  正常範圍下限: {lower_bound:.2f}, 正常範圍上限: {upper_bound:.2f}")
            # print(f"  異常值樣本 (最多顯示5個): \n{outliers[col].head()}") # 取消註解以查看樣本

            # 示例處理方式：Capping (將超出範圍的值替換為邊界值)
            # df_cleaned[col] = np.clip(df_cleaned[col], lower_bound, upper_bound)
            # print(f"  (示例) 若執行Capping，欄位 '{col}' 的異常值將被替換為邊界值。")
            print(f"  處理建議：對於欄位 '{col}' 的異常值，可以考慮Capping、移除樣本 (如果確定為錯誤且量少)、"
                  f"或進行資料轉換 (如log轉換)。具體操作需根據業務理解和模型需求決定。")
        else:
            print(f"\n欄位 '{col}' 中未檢測到明顯的潛在異常值 (IQR法)。")
    else:
        print(f"\n欄位 '{col}' 在 df_cleaned 中不存在，跳過異常值檢測。")


# --- 4.3 欄位轉換 ---
print("\n--- 4.3 欄位轉換 ---")

# --- 4.3.1 日期轉換 ---
print("\n資料集中沒有明顯的日期欄位需要轉換。") # 根據 ai4i2020 資料集的特性

# --- 4.3.2 類別編碼 ---
# 'Type' 欄位是類別型
if 'Type' in categorical_cols and 'Type' in df_cleaned.columns:
    print("\n對類別型欄位 'Type' 進行編碼...")
    # 選項1: Label Encoding (如果 'L', 'M', 'H' 有序)
    label_encoder = LabelEncoder()
    df_cleaned['Type_LabelEncoded'] = label_encoder.fit_transform(df_cleaned['Type'])
    print("使用 Label Encoding 對 'Type' 進行編碼:")
    print(df_cleaned[['Type', 'Type_LabelEncoded']].head())
    type_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print(f"  'Type' 欄位的類別映射 (編碼 -> 原始值): {type_mapping}")

    # 選項2: One-Hot Encoding (如果 'L', 'M', 'H' 無序或用於線性模型等)
    # df_cleaned = pd.get_dummies(df_cleaned, columns=['Type'], prefix='Type', drop_first=True)
    # print("\n若使用 One-Hot Encoding (示例，會替換原 'Type' 欄並增加新欄):")