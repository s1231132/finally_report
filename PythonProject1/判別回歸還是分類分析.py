import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. 設定圖形樣式與中文字體 ---
# 設定 seaborn 的樣式，使圖形更美觀
sns.set_theme(style="ticks")

# 設定中文字體，請確保您的環境中已安裝這些字體
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"字體設定警告: {e}")
    print("將使用 Matplotlib 預設字體，中文可能無法正常顯示。")


# --- 2. 載入並準備資料 ---
# 載入您之前使用的 'ai4i2020.csv' 檔案
try:
    df = pd.read_csv('ai4i2020.csv')
    print("成功載入 ai4i2020.csv 資料集。")
    print(f"原始資料筆數: {len(df)}")
except FileNotFoundError:
    print("錯誤：ai4i2020.csv 檔案未找到。請確保檔案與腳本在同一目錄下。")
    exit()

# --- 3. (新增) 資料取樣以加速繪圖 ---
# 當資料量大時，繪製 pairplot 會非常耗時。可以透過取樣來快速預覽。
# 將 use_sampling 設定為 False 即可使用全部資料進行繪圖。
use_sampling = True
sample_size = 2000

if use_sampling and len(df) > sample_size:
    print(f"\n注意：已啟用資料取樣功能，將隨機抽取 {sample_size} 筆資料進行繪圖以加快速度。")
    df_for_plot = df.sample(n=sample_size, random_state=42)
else:
    print("\n將使用全部資料進行繪圖。")
    df_for_plot = df

# 為了讓圖形更清晰，我們可以選擇圖中顯示的幾個關鍵數值特徵
# 根據您提供的圖片，我們選擇以下幾個欄位
# 您也可以調整這個列表來觀察不同的變數組合
columns_to_plot = [
    'Air temperature [K]',
    'Process temperature [K]',
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]',
    'Machine failure' # 'Machine failure' 用於顏色區分，也必須包含在內
]

# 從原始 DataFrame 中選取這些欄位
# 使用 df_for_plot (可能為取樣後或完整的資料)
df_subset = df_for_plot[columns_to_plot]

print(f"\n將使用以下欄位繪製成對圖: {df_subset.columns.tolist()}")


# --- 4. 繪製 Pairplot 成對圖 ---
print("\n>>> 正在繪製 Pairplot... (若使用全部資料，可能需要數十秒至一分鐘，請耐心等候)")

# 使用 seaborn.pairplot 進行繪圖
# - data: 指定要使用的 DataFrame
# - hue: 指定用於顏色分類的欄位，這裡我們用 'Machine failure' 來區分是否故障
# - corner=True: (可選) 只繪製矩陣的下半部分，可以節省空間並避免重複
# - plot_kws: (可選) 傳遞給非對角線上的散點圖的額外參數，例如設定點的大小和透明度
# - diag_kind: (可選) 設定對角線上圖形的種類，'auto', 'hist', 'kde'
pair_plot_figure = sns.pairplot(
    df_subset,
    hue='Machine failure',
    diag_kind='kde', # 在對角線上繪製核密度估計圖
    plot_kws={'alpha': 0.6, 's': 40, 'edgecolor': 'w', 'linewidth': 0.5}, # 設定散點圖樣式
    corner=False # 設定為 False 以繪製完整的矩陣圖，與您的範例圖一致
)


# --- 5. 客製化與顯示圖形 ---
# 添加一個整體的標題
pair_plot_figure.fig.suptitle('機器故障樣本分布觀察 (Pairplot)', y=1.02, fontsize=16)

# 顯示圖形
plt.show()

print("\n繪圖完成！")