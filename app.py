import streamlit as st
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import os
from PIL import Image
import joblib
from pathlib import Path
import json

# =============================================================================
# 基础配置
# =============================================================================
warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)
st.set_page_config(
    page_title="西宁地区风环境分析系统",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 学术风格CSS样式
st.markdown("""
<style>
    /* 全局字体和背景 */
    .main {
        background-color: #fafbfc;
    }

    /* 标题样式 */
    h1 {
        color: #1a365d;
        font-weight: 600;
        border-bottom: 2px solid #2c5282;
        padding-bottom: 10px;
        margin-bottom: 20px;
    }

    h2, h3 {
        color: #2d3748;
        font-weight: 500;
    }

    /* 卡片样式 */
    .academic-card {
        background: linear-gradient(135deg, #ffffff 0%, #f7fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }

    .academic-card-header {
        color: #1a365d;
        font-size: 16px;
        font-weight: 600;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid #e2e8f0;
    }

    /* 信息框样式 */
    .info-box {
        background-color: #ebf8ff;
        border-left: 4px solid #3182ce;
        padding: 15px 20px;
        border-radius: 0 6px 6px 0;
        margin: 15px 0;
    }

    .warning-box {
        background-color: #fffaf0;
        border-left: 4px solid #dd6b20;
        padding: 15px 20px;
        border-radius: 0 6px 6px 0;
        margin: 15px 0;
    }

    .success-box {
        background-color: #f0fff4;
        border-left: 4px solid #38a169;
        padding: 15px 20px;
        border-radius: 0 6px 6px 0;
        margin: 15px 0;
    }

    /* 指标展示 */
    .metric-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 15px;
        text-align: center;
    }

    .metric-value {
        font-size: 24px;
        font-weight: 600;
        color: #2c5282;
    }

    .metric-label {
        font-size: 13px;
        color: #718096;
        margin-top: 5px;
    }

    /* 侧边栏样式 */
    .css-1d391kg {
        background-color: #f7fafc;
    }

    /* 分隔线 */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(to right, transparent, #cbd5e0, transparent);
        margin: 25px 0;
    }

    /* 表格样式 */
    .dataframe {
        font-size: 13px;
    }

    /* 公式展示框 */
    .formula-box {
        background-color: #f7fafc;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 15px;
        font-family: 'Times New Roman', serif;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# 绘图风格设置
plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "STHeiti", "DejaVu Sans"],
    'axes.unicode_minus': False,
    'figure.dpi': 500,
    'font.size': 11,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
    'mathtext.default': 'regular',
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#4a5568',
    'axes.labelcolor': '#2d3748',
    'xtick.color': '#4a5568',
    'ytick.color': '#4a5568',
    'grid.color': '#e2e8f0',
    'grid.linewidth': 0.5
})

# =============================================================================
# 常量定义
# =============================================================================
KNOTS_TO_MS = 0.5144
LOOK_BACK = 7
EPOCHS = 80
BATCH_SIZE = 16
FUTURE_STEPS = 120
TREND_WINDOW = 14
TRAIN_RATIO = 0.8
AIR_DENSITY = 1.225  # kg/m³

# 学术配色方案
COLORS = {
    'primary': '#2c5282',
    'secondary': '#4a5568',
    'actual': '#1a365d',
    'train_pred': '#c53030',
    'test_pred': '#2b6cb0',
    'future_pred': '#38a169',
    'imf1': '#6b7280',
    'imf2': '#d69e2e',
    'imf3': '#9f7aea',
    'scatter_train': '#3182ce',
    'scatter_test': '#ed8936',
    'residual_pos': '#48bb78',
    'residual_neg': '#f56565',
    'grid': '#e2e8f0'
}

# 模型路径配置
MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
AUX_SCALER_PATH = MODEL_DIR / "aux_scaler.pkl"
GB_MODEL_PATH = MODEL_DIR / "residual_gb.pkl"
META_PATH = MODEL_DIR / "model_meta.json"

# 本地数据路径配置
DATA_PATHS = [
    r"C:\Users\Lenovo\Desktop\新建文件夹 (2)\西宁气象数据2021.csv",
    r"C:\Users\Lenovo\Desktop\新建文件夹 (2)\西宁气象数据2022.csv",
    r"C:\Users\Lenovo\Desktop\新建文件夹 (2)\西宁气象数据2023.csv",
    r"C:\Users\Lenovo\Desktop\新建文件夹 (2)\西宁气象数据2024.csv",
    r"C:\Users\Lenovo\Desktop\新建文件夹 (2)\西宁气象数据2025.csv"
]

SIM_ROOT_ZONGHE = r"C:\Users\Lenovo\Desktop\新建文件夹 (2)\综合楼仿真结果"
SIM_ROOT_BAJIAO = r"C:\Users\Lenovo\Desktop\新建文件夹 (2)\八角亭仿真结果"


# =============================================================================
# 辅助函数
# =============================================================================
def component_paths(idx):
    return (
        MODEL_DIR / f"imf{idx+1}_bilstm.keras",
        MODEL_DIR / f"imf{idx+1}_scaler.pkl"
    )


def metadata_matches(meta_path, aux_cols, look_back, k, train_ratio):
    if not meta_path.exists():
        return False
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
    except json.JSONDecodeError:
        return False
    return (
        saved.get('LOOK_BACK') == look_back and
        saved.get('K') == k and
        saved.get('aux_cols') == aux_cols and
        abs(saved.get('train_ratio', train_ratio) - train_ratio) < 1e-9
    )


def beautify_axis(ax, grid_alpha=0.4):
    """学术风格坐标轴美化"""
    ax.grid(True, alpha=grid_alpha, linestyle='-', linewidth=0.5, color=COLORS['grid'])
    ax.tick_params(axis='both', labelsize=10, colors=COLORS['secondary'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLORS['secondary'])
    ax.spines['bottom'].set_color(COLORS['secondary'])
    ax.set_facecolor('#fafbfc')


def vmd_decompose(signal, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7, max_iter=500):
    """变分模态分解"""
    T = len(signal)
    f_mirror = np.zeros(2 * T)
    f_mirror[:T//2] = signal[T//2-1::-1]
    f_mirror[T//2:3*T//2] = signal
    f_mirror[3*T//2:] = signal[-1:-T//2-1:-1]
    f_hat = np.fft.fftshift(np.fft.fft(f_mirror))
    N = len(f_hat)
    freqs_ext = np.arange(N) / N - 0.5
    u_hat = np.zeros((K, N), dtype=complex)
    omega = np.zeros((K, max_iter))
    for k in range(K):
        omega[k, 0] = (0.5 / K) * k
    lambda_hat = np.zeros(N, dtype=complex)

    for n in range(max_iter - 1):
        sum_uk = np.sum(u_hat, axis=0)
        for k in range(K):
            sum_other = sum_uk - u_hat[k]
            numerator = f_hat - sum_other + lambda_hat / 2
            denominator = 1 + 2 * alpha * (freqs_ext - omega[k, n]) ** 2
            u_hat[k] = numerator / denominator
            freq_sq = freqs_ext ** 2
            numerator_omega = np.sum(freq_sq * np.abs(u_hat[k]) ** 2)
            denominator_omega = np.sum(np.abs(u_hat[k]) ** 2) + 1e-10
            omega[k, n + 1] = numerator_omega / denominator_omega

        sum_uk = np.sum(u_hat, axis=0)
        lambda_hat = lambda_hat + tau * (f_hat - sum_uk)
        if np.sum(np.abs(omega[:, n + 1] - omega[:, n]) ** 2) < tol:
            break

    u = np.zeros((K, T))
    for k in range(K):
        u_full = np.fft.ifft(np.fft.ifftshift(u_hat[k]))
        u[k] = np.real(u_full[T//2:3*T//2])

    sort_idx = np.argsort(omega[:, n])
    return u[sort_idx], omega[:, n][sort_idx]


def create_multivariate_dataset(imf_data, aux_data, look_back):
    X, y = [], []
    for i in range(len(imf_data) - look_back):
        imf_seq = imf_data[i:(i + look_back)].reshape(-1, 1)
        aux_seq = aux_data[i:(i + look_back)]
        X.append(np.hstack([imf_seq, aux_seq]))
        y.append(imf_data[i + look_back])
    return np.array(X), np.array(y)


def build_future_aux_row(date, history_wind, env_defaults, aux_cols, min_date, trend_window):
    row = dict(env_defaults)
    arr = np.array(history_wind)
    day_of_year = date.timetuple().tm_yday
    row['Sin_Year'] = np.sin(2 * np.pi * day_of_year / 365)
    row['Cos_Year'] = np.cos(2 * np.pi * day_of_year / 365)
    row['Time_Index'] = float((date - min_date).days)

    def rolling_mean(window):
        subset = arr[-window:] if len(arr) >= window else arr
        return float(np.mean(subset))

    def rolling_std(window):
        subset = arr[-window:] if len(arr) >= window else arr
        return float(np.std(subset)) if len(subset) > 1 else 0.0

    def lag(l):
        return float(arr[-l-1]) if len(arr) > l else float(arr[0])

    if 'WDSP_MA3' in row:
        row['WDSP_MA3'] = rolling_mean(3)
    if 'WDSP_MA5' in row:
        row['WDSP_MA5'] = rolling_mean(5)
    if 'WDSP_STD3' in row:
        row['WDSP_STD3'] = rolling_std(3)
    if 'WDSP_Lag1' in row:
        row['WDSP_Lag1'] = lag(1)
    if 'WDSP_Lag2' in row:
        row['WDSP_Lag2'] = lag(2)
    if 'WDSP_Lag3' in row:
        row['WDSP_Lag3'] = lag(3)
    if 'dWDSP_1d' in row:
        row['dWDSP_1d'] = float(arr[-1] - arr[-2]) if len(arr) >= 2 else 0.0
    if 'WDSP_Trend14' in row:
        if len(arr) > trend_window:
            row['WDSP_Trend14'] = float(arr[-1] - arr[-trend_window]) / trend_window
        else:
            row['WDSP_Trend14'] = 0.0

    row['dSLP_1d'] = 0.0
    row['dSLP_2d'] = 0.0
    row['dTEMP_1d'] = 0.0
    return {col: row.get(col, env_defaults.get(col, 0.0)) for col in aux_cols}


def forecast_future(component_states, aux_scaler_obj, aux_cols, full_df, wind_hist, steps, min_date, trend_window):
    history = list(wind_hist)
    env_defaults = full_df[aux_cols].iloc[-1].to_dict()
    future_preds = []
    future_dates = []
    future_aux_rows = []
    current_date = full_df['DATE'].iloc[-1]
    for _ in range(steps):
        component_scaled_preds = []
        component_actual_preds = []
        for state in component_states:
            model_input = np.hstack([state['seq_imf'].reshape(-1, 1), state['seq_aux']])
            pred_scaled = state['model'].predict(model_input[np.newaxis], verbose=0)[0, 0]
            pred_actual = state['scaler_imf'].inverse_transform([[pred_scaled]])[0, 0]
            component_scaled_preds.append(pred_scaled)
            component_actual_preds.append(pred_actual)
        agg_pred = np.sum(component_actual_preds)
        future_preds.append(agg_pred)
        history.append(agg_pred)

        current_date = current_date + pd.Timedelta(days=1)
        future_dates.append(current_date)
        aux_row = build_future_aux_row(current_date, history, env_defaults, aux_cols, min_date, trend_window)
        future_aux_rows.append(aux_row)
        aux_row_values = np.array([[aux_row[col] for col in aux_cols]])
        aux_row_scaled = aux_scaler_obj.transform(aux_row_values)[0]

        for idx, state in enumerate(component_states):
            state['seq_imf'] = np.append(state['seq_imf'][1:], component_scaled_preds[idx])
            state['seq_aux'] = np.vstack([state['seq_aux'][1:], aux_row_scaled])

    future_aux_matrix = np.array([[row[col] for col in aux_cols] for row in future_aux_rows])
    return np.array(future_preds), np.array(future_dates), future_aux_matrix


# =============================================================================
# 数据加载与预处理
# =============================================================================
@st.cache_data
def load_and_preprocess_data(paths):
    frames = []
    for path in paths:
        try:
            df = None
            for sep in [',', '\t', ';', ' ']:
                try:
                    temp = pd.read_csv(path, sep=sep, encoding='utf-8')
                    if len(temp.columns) > 3:
                        df = temp
                        break
                except:
                    continue
            if df is None:
                df = pd.read_csv(path, sep=None, engine='python', encoding='utf-8')

            if 'DATE' in df.columns:
                parsed = pd.to_datetime(df['DATE'], errors='coerce')
                if parsed.isna().mean() > 0.5:
                    for fmt in ['%Y/%m/%d', '%Y-%m-%d', '%Y%m%d']:
                        parsed = pd.to_datetime(df['DATE'], format=fmt, errors='coerce')
                        if parsed.isna().mean() < 0.3:
                            break
                df['DATE'] = parsed
            frames.append(df)
        except Exception as e:
            st.warning(f"读取文件 {path} 失败: {e}")

    if not frames:
        return None

    data = pd.concat(frames, ignore_index=True).dropna(subset=['DATE'])
    data = data.sort_values('DATE').reset_index(drop=True)

    base_date = data['DATE'].min()
    features = ['WDSP', 'TEMP', 'SLP', 'DEWP', 'MAX', 'MIN', 'STP']
    for col in features:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce').replace(999.9, np.nan)

    if 'WDSP' in data.columns:
        data.loc[(data['WDSP'] > 50) | (data['WDSP'] < 0), 'WDSP'] = np.nan
    for col in ['TEMP', 'MAX', 'MIN', 'DEWP']:
        if col in data.columns:
            data.loc[(data[col] > 150) | (data[col] < -60), col] = np.nan
    for col in ['SLP', 'STP']:
        if col in data.columns:
            data.loc[(data[col] > 1100) | (data[col] < 850), col] = np.nan

    data = data.interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(method='ffill')
    data['WDSP_ms'] = data['WDSP'] * KNOTS_TO_MS

    data['Time_Index'] = (data['DATE'] - base_date).dt.days.astype(float)
    data['DayOfYear'] = data['DATE'].dt.dayofyear
    data['Month'] = data['DATE'].dt.month
    data['Sin_Year'] = np.sin(2 * np.pi * data['DayOfYear'] / 365)
    data['Cos_Year'] = np.cos(2 * np.pi * data['DayOfYear'] / 365)

    data['dSLP_1d'] = data['SLP'].diff(1).fillna(0)
    data['dSLP_2d'] = data['SLP'].diff(2).fillna(0)
    data['dTEMP_1d'] = data['TEMP'].diff(1).fillna(0)
    data['dTEMP_2d'] = data['TEMP'].diff(2).fillna(0)
    data['dWDSP_1d'] = data['WDSP_ms'].diff(1).fillna(0)
    data['WDSP_Trend14'] = (data['WDSP_ms'].diff(TREND_WINDOW) / TREND_WINDOW).fillna(0)

    for window in [3, 5, 7]:
        data[f'WDSP_MA{window}'] = data['WDSP_ms'].rolling(window=window, min_periods=1).mean()
        data[f'WDSP_STD{window}'] = data['WDSP_ms'].rolling(window=window, min_periods=1).std().fillna(0)
        data[f'SLP_MA{window}'] = data['SLP'].rolling(window=window, min_periods=1).mean()

    for lag in [1, 2, 3]:
        data[f'WDSP_Lag{lag}'] = data['WDSP_ms'].shift(lag).fillna(method='bfill')
        data[f'SLP_Lag{lag}'] = data['SLP'].shift(lag).fillna(method='bfill')
        data[f'TEMP_Lag{lag}'] = data['TEMP'].shift(lag).fillna(method='bfill')

    data['TEMP_Range'] = data['MAX'] - data['MIN']
    data['TEMP_DEWP_Diff'] = data['TEMP'] - data['DEWP']
    data = data.fillna(method='bfill').fillna(method='ffill')

    return data, base_date


# =============================================================================
# 页面一：深度学习风速预测
# =============================================================================
def page_prediction():
    st.title("深度学习风速预测与拟合分析")

    st.markdown("""
    <div class="info-box">
    <strong>模块说明</strong><br>
    本模块采用VMD-BiLSTM-GBR混合模型对西宁地区日平均风速进行预测分析。
    模型通过变分模态分解提取风速信号的多尺度特征，结合双向长短期记忆网络与梯度提升残差修正，
    实现高精度的风速时序预测。
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    with st.spinner("正在加载数据并执行模型运算..."):
        result = load_and_preprocess_data(DATA_PATHS)
        if result is None:
            st.error("数据加载失败，请检查数据文件路径配置。")
            return

        full_data, base_date = result
        wind_series = full_data['WDSP_ms'].values
        dates = full_data['DATE'].values
        BASE_DATE = base_date

        aux_feature_cols = [
            'Sin_Year', 'Cos_Year', 'Time_Index', 'dSLP_1d', 'dSLP_2d', 'dTEMP_1d',
            'WDSP_MA3', 'WDSP_MA5', 'WDSP_STD3', 'WDSP_Lag1', 'WDSP_Lag2', 'WDSP_Lag3',
            'SLP_Lag1', 'TEMP_Lag1', 'TEMP_Range', 'dWDSP_1d', 'WDSP_Trend14'
        ]
        aux_feature_cols = [col for col in aux_feature_cols if col in full_data.columns]
        aux_features = full_data[aux_feature_cols].values
        K = 3

        metadata_ok = metadata_matches(META_PATH, aux_feature_cols, LOOK_BACK, K, TRAIN_RATIO)
        component_models_ready = all(
            component_paths(idx)[0].exists() and component_paths(idx)[1].exists()
            for idx in range(K)
        )
        required_ready = (
            component_models_ready and
            AUX_SCALER_PATH.exists() and
            GB_MODEL_PATH.exists() and
            metadata_ok
        )

        if not required_ready:
            st.markdown("""
            <div class="warning-box">
            <strong>模型状态检测</strong><br>
            未检测到完整的预训练模型文件，请先运行模型训练脚本。
            </div>
            """, unsafe_allow_html=True)

            st.write("**模型组件检查：**")
            col1, col2, col3 = st.columns(3)
            with col1:
                status = "已就绪" if component_models_ready else "缺失"
                st.write(f"- 分量预测模型: {status}")
            with col2:
                status = "已就绪" if AUX_SCALER_PATH.exists() else "缺失"
                st.write(f"- 辅助特征缩放器: {status}")
            with col3:
                status = "已就绪" if GB_MODEL_PATH.exists() else "缺失"
                st.write(f"- 残差修正模型: {status}")
            return

        st.markdown("""
        <div class="success-box">
        <strong>模型加载成功</strong><br>
        已检测到完整的预训练模型，正在执行预测计算...
        </div>
        """, unsafe_allow_html=True)

        aux_scaler = joblib.load(AUX_SCALER_PATH)
        aux_scaled = aux_scaler.transform(aux_features)
        xgb_model = joblib.load(GB_MODEL_PATH)

        component_states = []
        all_train_preds = []
        all_test_preds = []
        component_metrics = []
        split_idx = int(len(wind_series) * TRAIN_RATIO)

        try:
            imfs, omega = vmd_decompose(wind_series, alpha=2000, K=K)
        except Exception as e:
            st.warning(f"VMD分解异常，采用替代方案：{e}")
            trend = pd.Series(wind_series).rolling(window=30, center=True, min_periods=1).mean().values
            seasonal = pd.Series(wind_series).rolling(window=7, center=True, min_periods=1).mean().values - trend
            residual = wind_series - trend - seasonal
            imfs = np.array([trend, seasonal, residual])

        for k in range(K):
            imf_k = imfs[k]
            model_path, scaler_path = component_paths(k)
            scaler_imf = joblib.load(scaler_path)
            imf_scaled = scaler_imf.transform(imf_k.reshape(-1, 1)).flatten()
            model = load_model(model_path)

            X, y = create_multivariate_dataset(imf_scaled, aux_scaled, LOOK_BACK)
            train_size = split_idx - LOOK_BACK
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]

            train_pred = scaler_imf.inverse_transform(model.predict(X_train, verbose=0)).flatten()
            test_pred = scaler_imf.inverse_transform(model.predict(X_test, verbose=0)).flatten()
            y_train_actual = scaler_imf.inverse_transform(y_train.reshape(-1, 1)).flatten()
            y_test_actual = scaler_imf.inverse_transform(y_test.reshape(-1, 1)).flatten()

            component_metrics.append({
                'name': f'IMF{k+1}',
                'r2_train': r2_score(y_train_actual, train_pred),
                'r2_test': r2_score(y_test_actual, test_pred)
            })
            all_train_preds.append(train_pred)
            all_test_preds.append(test_pred)
            component_states.append({
                'model': model,
                'scaler_imf': scaler_imf,
                'seq_imf': imf_scaled[-LOOK_BACK:].copy(),
                'seq_aux': aux_scaled[-LOOK_BACK:].copy()
            })

        final_train_pred = np.sum(all_train_preds, axis=0)
        final_test_pred = np.sum(all_test_preds, axis=0)

        X_train_xgb = aux_features[LOOK_BACK:split_idx]
        X_test_xgb = aux_features[split_idx:]
        final_train_pred_corrected = final_train_pred + xgb_model.predict(X_train_xgb)
        final_test_pred_corrected = final_test_pred + xgb_model.predict(X_test_xgb)

        future_pred, future_dates, future_aux_features = forecast_future(
            component_states, aux_scaler, aux_feature_cols, full_data, wind_series,
            FUTURE_STEPS, BASE_DATE, TREND_WINDOW
        )
        future_pred_corrected = future_pred + xgb_model.predict(future_aux_features)

        y_train_true = wind_series[LOOK_BACK:split_idx]
        y_test_true = wind_series[split_idx:]
        train_dates = dates[LOOK_BACK:split_idx]
        test_dates = dates[split_idx:]
        all_dates = np.concatenate([train_dates, test_dates])
        all_actual = np.concatenate([y_train_true, y_test_true])

        def calculate_metrics(y_true, y_pred):
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            mask = y_true != 0
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.any(mask) else 0
            corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
            return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2, 'Corr': corr}

        train_metrics = calculate_metrics(y_train_true, final_train_pred_corrected)
        test_metrics = calculate_metrics(y_test_true, final_test_pred_corrected)

    # 模型评价指标展示
    st.subheader("模型性能评估")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**训练集评价指标**")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("决定系数 R²", f"{train_metrics['R2']:.4f}")
        with metrics_col2:
            st.metric("均方根误差 RMSE", f"{train_metrics['RMSE']:.4f} m/s")
        with metrics_col3:
            st.metric("平均绝对误差 MAE", f"{train_metrics['MAE']:.4f} m/s")

    with col2:
        st.markdown("**测试集评价指标**")
        metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
        with metrics_col1:
            st.metric("决定系数 R²", f"{test_metrics['R2']:.4f}")
        with metrics_col2:
            st.metric("均方根误差 RMSE", f"{test_metrics['RMSE']:.4f} m/s")
        with metrics_col3:
            st.metric("平均绝对误差 MAE", f"{test_metrics['MAE']:.4f} m/s")

    st.markdown("---")

    # 图1：完整预测结果
    st.subheader("图1 完整时序预测结果")
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    fig1.patch.set_facecolor('#fafbfc')

    ax1.plot(pd.to_datetime(all_dates), all_actual,
             color=COLORS['actual'], linewidth=1.2, label='观测值', alpha=0.9)
    ax1.plot(pd.to_datetime(train_dates), final_train_pred_corrected,
             color=COLORS['train_pred'], linewidth=1.0, linestyle='--',
             label=f'训练拟合 ($R²$={train_metrics["R2"]:.3f})', alpha=0.8)
    ax1.plot(pd.to_datetime(test_dates), final_test_pred_corrected,
             color=COLORS['test_pred'], linewidth=1.2, linestyle='-.',
             label=f'测试预测 ($R²$={test_metrics["R2"]:.3f})', alpha=0.9)
    ax1.plot(future_dates, future_pred_corrected,
             color=COLORS['future_pred'], linewidth=1.5, linestyle='--',
             label=f'未来预测（{FUTURE_STEPS}天）', alpha=0.9)

    split_date = pd.to_datetime(test_dates[0])
    ax1.axvline(x=split_date, color='#718096', linestyle=':', linewidth=1.5, alpha=0.7, label='训练/测试划分线')
    ax1.axvspan(pd.to_datetime(test_dates[-1]), future_dates[-1], color=COLORS['future_pred'], alpha=0.06)

    ax1.set_xlabel('日期', fontsize=11, fontweight='medium')
    ax1.set_ylabel('风速 (m/s)', fontsize=11, fontweight='medium')
    ax1.set_title('西宁地区日平均风速预测', fontsize=13, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', frameon=True, fancybox=False, edgecolor='#e2e8f0', fontsize=9)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.tick_params(axis='x', rotation=20)
    beautify_axis(ax1)
    plt.tight_layout()
    st.pyplot(fig1)

    st.markdown("---")

    # 图2：详细分析四图
    st.subheader("图2 预测性能详细分析")
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.patch.set_facecolor('#fafbfc')

    # (a) 训练集拟合
    ax2_1 = axes2[0, 0]
    ax2_1.plot(pd.to_datetime(train_dates), y_train_true, color=COLORS['actual'], linewidth=1.0, label='观测值', alpha=0.9)
    ax2_1.plot(pd.to_datetime(train_dates), final_train_pred_corrected,
               color=COLORS['train_pred'], linewidth=0.9, linestyle='--', label='训练拟合', alpha=0.8)
    ax2_1.set_title(f'(a) 训练集拟合 ($R²$={train_metrics["R2"]:.4f})', fontsize=11, fontweight='bold', loc='left')
    ax2_1.set_xlabel('日期', fontsize=10)
    ax2_1.set_ylabel('风速 (m/s)', fontsize=10)
    ax2_1.legend(frameon=True, fancybox=False, edgecolor='#e2e8f0', fontsize=9)
    ax2_1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2_1.tick_params(axis='x', rotation=20)
    beautify_axis(ax2_1)

    # (b) 测试集预测
    ax2_2 = axes2[0, 1]
    ax2_2.plot(pd.to_datetime(test_dates), y_test_true, color=COLORS['actual'], linewidth=1.2, label='观测值', alpha=0.9)
    ax2_2.plot(pd.to_datetime(test_dates), final_test_pred_corrected,
               color=COLORS['test_pred'], linewidth=1.0, linestyle='--', label='测试预测', alpha=0.85)
    ax2_2.set_title(f'(b) 测试集预测 ($R²$={test_metrics["R2"]:.4f})', fontsize=11, fontweight='bold', loc='left')
    ax2_2.set_xlabel('日期', fontsize=10)
    ax2_2.set_ylabel('风速 (m/s)', fontsize=10)
    ax2_2.legend(frameon=True, fancybox=False, edgecolor='#e2e8f0', fontsize=9)
    ax2_2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2_2.tick_params(axis='x', rotation=20)
    beautify_axis(ax2_2)

    # (c) 散点图
    ax2_3 = axes2[1, 0]
    ax2_3.scatter(y_train_true, final_train_pred_corrected,
                  c=COLORS['scatter_train'], alpha=0.4, s=25, edgecolors='none', label='训练集')
    ax2_3.scatter(y_test_true, final_test_pred_corrected,
                  c=COLORS['scatter_test'], alpha=0.5, s=30, edgecolors='none', label='测试集')
    min_val = min(all_actual.min(), final_test_pred_corrected.min()) - 0.2
    max_val = max(all_actual.max(), final_test_pred_corrected.max()) + 0.2
    ax2_3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=1.5, label='1:1 参考线', alpha=0.7)
    ax2_3.set_xlabel('观测风速 (m/s)', fontsize=10)
    ax2_3.set_ylabel('预测风速 (m/s)', fontsize=10)
    ax2_3.set_title('(c) 观测值与预测值散点图', fontsize=11, fontweight='bold', loc='left')
    ax2_3.legend(frameon=True, fancybox=False, edgecolor='#e2e8f0', fontsize=9)
    ax2_3.set_xlim([min_val, max_val])
    ax2_3.set_ylim([min_val, max_val])
    ax2_3.set_aspect('equal', adjustable='box')
    beautify_axis(ax2_3)

    # (d) 测试集局部放大
    ax2_4 = axes2[1, 1]
    display_len = min(80, len(test_dates))
    ax2_4.plot(pd.to_datetime(test_dates[:display_len]), y_test_true[:display_len],
               color=COLORS['actual'], linewidth=1.5, marker='o', markersize=3, label='观测值', alpha=0.9)
    ax2_4.plot(pd.to_datetime(test_dates[:display_len]), final_test_pred_corrected[:display_len],
               color=COLORS['test_pred'], linewidth=1.2, linestyle='--', marker='^', markersize=3, label='预测值', alpha=0.85)
    ax2_4.set_title(f'(d) 测试集局部放大（前 {display_len} 天）', fontsize=11, fontweight='bold', loc='left')
    ax2_4.set_xlabel('日期', fontsize=10)
    ax2_4.set_ylabel('风速 (m/s)', fontsize=10)
    ax2_4.legend(frameon=True, fancybox=False, edgecolor='#e2e8f0', fontsize=9)
    ax2_4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax2_4.tick_params(axis='x', rotation=35)
    beautify_axis(ax2_4)

    plt.tight_layout()
    st.pyplot(fig2)

    st.markdown("---")

    # 图3：VMD分解结果
    st.subheader("图3 变分模态分解(VMD)结果")
    fig_vmd, axes_vmd = plt.subplots(K + 1, 1, figsize=(14, 2.5 * (K + 1)))
    fig_vmd.patch.set_facecolor('#fafbfc')

    axes_vmd[0].plot(pd.to_datetime(dates), wind_series, color=COLORS['actual'], linewidth=0.9, alpha=0.9)
    axes_vmd[0].set_title('(a) 原始风速序列', fontsize=11, fontweight='bold', loc='left')
    axes_vmd[0].set_ylabel('风速 (m/s)', fontsize=10)
    beautify_axis(axes_vmd[0])

    imf_titles = ['(b) IMF1 - 趋势分量', '(c) IMF2 - 周期分量', '(d) IMF3 - 残差分量']
    imf_colors = [COLORS['imf1'], COLORS['imf2'], COLORS['imf3']]
    for i in range(K):
        axes_vmd[i + 1].plot(pd.to_datetime(dates), imfs[i], color=imf_colors[i], linewidth=0.8, alpha=0.9)
        axes_vmd[i + 1].set_title(imf_titles[i], fontsize=11, fontweight='bold', loc='left')
        axes_vmd[i + 1].set_ylabel('振幅 (m/s)', fontsize=10)
        axes_vmd[i + 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        beautify_axis(axes_vmd[i + 1])
    axes_vmd[-1].set_xlabel('日期', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_vmd)

    st.markdown("---")

    # 图4：各分量预测性能
    st.subheader("图4 各IMF分量预测性能")
    fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
    fig3.patch.set_facecolor('#fafbfc')

    imf_full_names = ['IMF1（趋势）', 'IMF2（周期）', 'IMF3（残差）']
    bar_colors = [COLORS['imf1'], COLORS['imf2'], COLORS['imf3']]

    for i, metrics in enumerate(component_metrics):
        ax = axes3[i]
        bars = ax.bar(['训练集', '测试集'],
                      [metrics['r2_train'], metrics['r2_test']],
                      color=bar_colors[i], alpha=0.75, edgecolor='#4a5568', linewidth=0.8, width=0.5)
        ax.set_title(f'{imf_full_names[i]}', fontsize=11, fontweight='bold')
        ax.set_ylabel('$R²$ 指标', fontsize=10)
        ax.set_ylim([0, 1.1])
        beautify_axis(ax)
        for bar, val in zip(bars, [metrics['r2_train'], metrics['r2_test']]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='medium')
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("---")

    # 图5：残差分析
    st.subheader("图5 预测残差分析")
    residuals = y_test_true - final_test_pred_corrected
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 4.5))
    fig4.patch.set_facecolor('#fafbfc')

    ax4_1 = axes4[0]
    ax4_1.plot(pd.to_datetime(test_dates), residuals, color='#5c6784', linewidth=0.8, alpha=0.8)
    ax4_1.axhline(y=0, color=COLORS['test_pred'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax4_1.fill_between(pd.to_datetime(test_dates), residuals, 0,
                       where=(residuals > 0), color=COLORS['residual_pos'], alpha=0.3)
    ax4_1.fill_between(pd.to_datetime(test_dates), residuals, 0,
                       where=(residuals < 0), color=COLORS['residual_neg'], alpha=0.3)
    ax4_1.set_title('(a) 残差时间序列', fontsize=11, fontweight='bold', loc='left')
    ax4_1.set_xlabel('日期', fontsize=10)
    ax4_1.set_ylabel('残差 (m/s)', fontsize=10)
    ax4_1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax4_1.tick_params(axis='x', rotation=20)
    beautify_axis(ax4_1)

    ax4_2 = axes4[1]
    ax4_2.hist(residuals, bins=30, color=COLORS['primary'], edgecolor='white', alpha=0.75)
    ax4_2.axvline(x=0, color=COLORS['test_pred'], linestyle='--', linewidth=1.5, alpha=0.7, label='零残差')
    ax4_2.axvline(x=np.mean(residuals), color=COLORS['train_pred'], linestyle='-', linewidth=1.5,
                  label=f'平均值: {np.mean(residuals):.3f}', alpha=0.8)
    ax4_2.set_title(f'(b) 残差分布 (SD={np.std(residuals):.3f} m/s)', fontsize=11, fontweight='bold', loc='left')
    ax4_2.set_xlabel('残差 (m/s)', fontsize=10)
    ax4_2.set_ylabel('频数', fontsize=10)
    ax4_2.legend(frameon=True, fancybox=False, edgecolor='#e2e8f0', fontsize=9)
    beautify_axis(ax4_2)

    plt.tight_layout()
    st.pyplot(fig4)

    st.markdown("---")

    # 图6：未来预测
    st.subheader("图6 未来风速预测")
    fig5, ax5 = plt.subplots(figsize=(14, 4.5))
    fig5.patch.set_facecolor('#fafbfc')

    history_len = min(60, len(test_dates))
    if history_len > 0:
        ax5.plot(pd.to_datetime(test_dates[-history_len:]), y_test_true[-history_len:],
                 color=COLORS['actual'], linewidth=1.2, marker='o', markersize=2.5, label='近期观测', alpha=0.9)
        ax5.plot(pd.to_datetime(test_dates[-history_len:]), final_test_pred_corrected[-history_len:],
                 color=COLORS['test_pred'], linewidth=1.0, linestyle='--', marker='^', markersize=2.5, label='近期预测', alpha=0.8)
    ax5.plot(future_dates, future_pred_corrected,
             color=COLORS['future_pred'], linewidth=1.5, linestyle='-', marker='s', markersize=2.5, label='未来预测', alpha=0.9)
    ax5.axvspan(future_dates[0], future_dates[-1], color=COLORS['future_pred'], alpha=0.06)
    ax5.set_title(f'未来 {FUTURE_STEPS} 天风速预测', fontsize=13, fontweight='bold', pad=10)
    ax5.set_xlabel('日期', fontsize=11)
    ax5.set_ylabel('风速 (m/s)', fontsize=11)
    ax5.legend(frameon=True, fancybox=False, edgecolor='#e2e8f0', fontsize=9)
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax5.tick_params(axis='x', rotation=30)
    beautify_axis(ax5)
    plt.tight_layout()
    st.pyplot(fig5)


# =============================================================================
# 页面二：风速仿真结果比对
# =============================================================================
def get_sim_images(region_type, speed, direction=None):
    """根据输入参数检索本地图片路径"""
    speed_str = f"{float(speed):g}"
    folder_name = ""
    base_path = ""

    if region_type == "综合楼":
        base_path = SIM_ROOT_ZONGHE
        if speed == 8:
            folder_name = "风速8（极端风速）-无防护林"
        else:
            folder_name = f"风速{speed_str}-无防护林"
    elif region_type == "八角亭":
        base_path = SIM_ROOT_BAJIAO
        if direction:
            folder_name = f"{direction}风{speed_str}"
    elif region_type == "图书馆":
        base_path = os.path.join(os.getcwd(), "图书馆仿真结果")
        folder_name = f"风速{speed_str}-狭管效应"
    else:
        base_path = os.path.join(os.getcwd(), "其他仿真结果")
        folder_name = f"风速{speed_str}-其他区域"

    full_folder_path = os.path.join(base_path, folder_name)

    target_files = {
        "涡粘性系数": "Eddy Viscosity Coefficient",
        "速度云图": "Magnitude of Velocity（速度标量大小云图）.png",
        "速度云图(带标记)": "Magnitude of Velocity（速度标量大小云图）(带标记)",
        "风速流向": "oil flow",
        "地面气压": "Pressure",
        "湍流耗散率": "Turbulence Dissipation Rate",
        "湍流能量": "Turbulence Energy"
    }

    found_images = {}

    if os.path.exists(full_folder_path):
        for filename in os.listdir(full_folder_path):
            for key, pattern in target_files.items():
                if pattern.split('(')[0].strip() in filename or pattern in filename:
                    found_images[key] = os.path.join(full_folder_path, filename)
    else:
        os.makedirs(full_folder_path, exist_ok=True)

    return found_images, full_folder_path


def page_simulation():
    st.title("CFD风速仿真结果比对与物理计算")

    st.markdown("""
    <div class="info-box">
    <strong>模块说明</strong><br>
    本模块基于计算流体力学(CFD)仿真结果，结合物理理论计算，对校园不同区域的风场特性进行定量评估与可视化分析。
    核心功能包括：多区域风场分析、物理公式计算、仿真结果可视化对比。
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col_input, col_info = st.columns([1, 2])

    with col_input:
        st.markdown("**参数设置**")

        location = st.selectbox(
            "选择仿真场景",
            [
                "综合楼广场及两侧道路风场分析",
                "八角亭及其四周风场分析",
                "图书馆与财经学院狭管效应分析",
                "其他分析"
            ],
            help="选择要分析的校园区域"
        )

        region_type = {
            "综合楼广场及两侧道路风场分析": "综合楼",
            "八角亭及其四周风场分析": "八角亭",
            "图书馆与财经学院狭管效应分析": "图书馆",
            "其他分析": "其他"
        }[location]

        direction = None

        if region_type == "八角亭":
            direction = st.selectbox("选择风向", ["北", "南", "东", "西"])
            avail_speeds = [0.5, 0.8, 0.9, 1.2, 1.4, 1.6, 1.8, 2.0]
            wind_speed = st.select_slider(
                "选择风速 (m/s)",
                options=avail_speeds,
                value=1.4,
                help="八角亭区域的风速范围: 0.5-2.0 m/s"
            )
        elif region_type == "综合楼":
            avail_speeds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8,
                            1.9, 2.0, 8.0]
            wind_speed = st.select_slider(
                "选择风速 (m/s)",
                options=avail_speeds,
                value=1.6,
                help="综合楼区域的风速范围: 0.1-8.0 m/s"
            )
        else:
            avail_speeds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
            wind_speed = st.select_slider(
                "选择风速 (m/s)",
                options=avail_speeds,
                value=1.5,
                help="默认风速范围: 0.5-3.0 m/s"
            )
            direction = st.selectbox("选择风向", ["北", "南", "东", "西"])

    with col_info:
        st.markdown("**物理理论计算结果**")

        dynamic_pressure = 0.5 * AIR_DENSITY * (wind_speed ** 2)
        reynolds_proxy = wind_speed * 100000

        region_k_factors = {
            "综合楼广场及两侧道路风场分析": 1.35,
            "八角亭及其四周风场分析": 1.12,
            "图书馆与财经学院狭管效应分析": 1.68,
            "其他分析": 1.00
        }
        k_factor = region_k_factors[location]

        col_metrics = st.columns(3)
        with col_metrics[0]:
            st.metric(label="输入风速", value=f"{wind_speed} m/s", help="气象站预报的大环境风速")
        with col_metrics[1]:
            st.metric(label="理论动压", value=f"{dynamic_pressure:.2f} Pa", help="基于伯努利方程计算")
        with col_metrics[2]:
            st.metric(label="参考雷诺数", value=f"{reynolds_proxy:.1e}", help="流场湍流特性参考")

        st.markdown(f"""
        <div class="formula-box">
        <strong>计算公式：</strong><br>
        动压：P = 0.5 × ρ × v²<br>
        其中：空气密度 ρ = {AIR_DENSITY} kg/m³，风速 v = {wind_speed} m/s<br>
        <em>计算结果：P = 0.5 × {AIR_DENSITY} × ({wind_speed})² = {dynamic_pressure:.2f} Pa</em>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader(f"仿真结果可视化：{location} - {wind_speed}m/s")

    images, folder_path = get_sim_images(region_type, wind_speed, direction)

    show_order = [
        ("速度云图", "速度标量分布"),
        ("速度云图(带标记)", "速度标量分布（含标记）"),
        ("地面气压", "地面气压云图"),
        ("风速流向", "流线/油膜图"),
        ("湍流能量", "湍流动能"),
        ("湍流耗散率", "湍流耗散率")
    ]

    if not images:
        st.markdown("""
        <div class="warning-box">
        <strong>提示</strong><br>
        未找到对应路径下的仿真结果文件。请将仿真图片放入指定目录。
        </div>
        """, unsafe_allow_html=True)

        st.write(f"**建议的文件夹路径：** `{folder_path}`")

        cols = st.columns(3)
        for idx, (img_key, label) in enumerate(show_order):
            with cols[idx % 3]:
                st.info(f"待放置：{label}")
    else:
        cols = st.columns(3)
        for idx, (img_key, label) in enumerate(show_order):
            path = images.get(img_key)
            with cols[idx % 3]:
                if path and os.path.exists(path):
                    try:
                        image = Image.open(path)
                        st.image(image, caption=label, use_column_width=True)
                    except Exception as e:
                        st.error(f"无法加载图片: {img_key}")
                else:
                    st.info(f"暂无 {label}")

    st.markdown("---")
    st.subheader("结果比对分析")

    extreme_note = "注意：此为极端风速工况，重点关注角部流分离现象。" if wind_speed >= 8 else "此为常规风速工况，流场相对平稳。"

    st.markdown(f"""
    **区域分析**：当前分析区域为 **{location}**，风速放大系数约为 **{k_factor:.2f}**。

    **数值对比**：理论计算的风压为 **{dynamic_pressure:.2f} Pa**。若有仿真结果，请对比"表面压力分布"云图中受风面的高压区数值。

    **流场特性**：观察"速度标量场"中是否存在明显的风速放大或减小区域，特别关注建筑角部和狭窄通道。

    **工况说明**：当前风速 {wind_speed} m/s。{extreme_note}
    """)


# =============================================================================
# 页面三：校园风沙环境综合评估
# =============================================================================
def page_wind_sand_evaluation():
    st.title("校园风沙环境多物理场综合评估")

    st.markdown("""
    <div class="info-box">
    <strong>模块说明</strong><br>
    本模块基于宏观气象数据与微观环境特性，通过物理模型耦合计算，实现校园特定区域的风沙环境定量评估。
    核心算法包括：局部风速映射模型、起沙临界阈值判断、Bagnold输沙率公式。
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("输入参数设置")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**气象站预报风速 (V_inlet)**")
        inlet_velocity = st.number_input(
            "输入气象站预报风速 (m/s)",
            min_value=0.0,
            max_value=30.0,
            value=10.0,
            step=0.1,
            help="模拟校园所处大环境风场强度"
        )

    with col2:
        st.markdown("**风速放大系数 (K) 设置**")
        mode = st.radio(
            "选择K值获取方式",
            ["数据库模式", "手动输入模式"],
            help="风速放大系数描述局部风速相对于大环境风速的倍数"
        )

        if mode == "数据库模式":
            locations = {
                "综合楼广场及两侧道路": 1.35,
                "八角亭及其四周": 1.12,
                "图书馆与财经学院狭管区": 1.68,
                "其他区域": 1.00
            }
            selected_location = st.selectbox("选择预设评估地点", list(locations.keys()))
            k_factor = locations[selected_location]
            st.info(f"该地点风速放大系数：K = {k_factor}")
        else:
            k_factor = st.number_input(
                "手动输入风速放大系数 (K)",
                min_value=0.1,
                max_value=5.0,
                value=1.25,
                step=0.01,
                help="通过CFD计算或现场实测得到"
            )

    st.markdown("---")
    st.subheader("内置物理模型参数")

    col_params1, col_params2, col_params3 = st.columns(3)
    with col_params1:
        st.metric("起沙临界风速 (V_critical)", "6.0 m/s")
    with col_params2:
        st.metric("Bagnold经验常数 (C)", "0.002")
    with col_params3:
        st.metric("输入风速 (V_inlet)", f"{inlet_velocity} m/s")

    st.markdown("---")
    st.subheader("计算流程")

    # 计算逻辑
    local_velocity = k_factor * inlet_velocity
    V_critical = 6.0
    C = 0.002

    if local_velocity > V_critical:
        Q = C * (local_velocity - V_critical) ** 3
    else:
        Q = 0.0

    # 风险等级判断
    if Q > 0.5:
        risk_level = "极高风险"
        risk_color = "#c53030"
        risk_desc = "强风条件下输沙量剧增，可能造成严重的风沙危害，需立即采取防护措施"
    elif 0.1 < Q <= 0.5:
        risk_level = "高风险"
        risk_color = "#dd6b20"
        risk_desc = "风速较大，输沙量显著，建议采取相应防护措施"
    elif 0.0 < Q <= 0.1:
        risk_level = "中低风险"
        risk_color = "#d69e2e"
        risk_desc = "风速适中，输沙量较小，风险可控"
    else:
        risk_level = "安全"
        risk_color = "#38a169"
        risk_desc = "风速较低，无明显输沙现象，环境安全"

    # 计算过程展示
    with st.expander("查看详细计算过程", expanded=False):
        st.latex(r"V_{local} = K \times V_{inlet}")
        st.write(f"计算：V_local = {k_factor} × {inlet_velocity} = {local_velocity:.2f} m/s")

        st.markdown("---")
        if local_velocity > V_critical:
            st.write(f"局部风速 {local_velocity:.2f} > {V_critical} m/s，超过起沙临界值，激活输沙量计算")
            st.latex(r"Q = C \times (V_{local} - V_{critical})^3")
            st.write(f"计算：Q = {C} × ({local_velocity:.2f} - {V_critical})³ = {Q:.6f} kg/m/s")
        else:
            st.write(f"局部风速 {local_velocity:.2f} ≤ {V_critical} m/s，低于起沙临界值，输沙量为 0")

    st.markdown("---")
    st.subheader("评估结果输出")

    col_result1, col_result2, col_result3 = st.columns(3)

    with col_result1:
        st.markdown("**局部实际风速**")
        st.metric(label="V_local", value=f"{local_velocity:.2f} m/s", help="经K值修正后的实际风速")
        st.caption("物理意义：目标位置的实际风场强度")

    with col_result2:
        st.markdown("**理论输沙率**")
        st.metric(label="Q (输沙率)", value=f"{Q:.6f} kg/m/s", help="单位时间内通过单位宽度的输沙质量")
        st.caption("物理意义：风沙危害的定量评估指标")

    with col_result3:
        st.markdown("**环境风险等级**")
        st.markdown(f"""
        <div style='background-color: {risk_color}; padding: 15px; border-radius: 6px; color: white; text-align: center;'>
        <strong style='font-size: 18px;'>{risk_level}</strong>
        </div>
        """, unsafe_allow_html=True)
        st.caption(risk_desc)

    st.markdown("---")
    st.subheader("可视化分析")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.patch.set_facecolor('#fafbfc')

    # 风速-输沙率关系曲线
    v_range = np.linspace(0, 20, 100)
    q_range = np.where(v_range > V_critical, C * (v_range - V_critical) ** 3, 0)

    ax1.plot(v_range, q_range, color=COLORS['primary'], linewidth=2, label='Bagnold 输沙公式')
    ax1.axvline(x=V_critical, color=COLORS['train_pred'], linestyle='--', linewidth=1.5,
                label=f'起沙临界风速 ({V_critical} m/s)')
    ax1.axvline(x=local_velocity, color=COLORS['future_pred'], linestyle='-.', linewidth=1.5,
                label=f'当前局部风速 ({local_velocity:.1f} m/s)')
    ax1.scatter(local_velocity, Q, color='#e53e3e', s=80, zorder=5, edgecolors='white', linewidth=1.5)

    ax1.set_xlabel('风速 (m/s)', fontsize=10, fontweight='medium')
    ax1.set_ylabel('输沙率 (kg/m/s)', fontsize=10, fontweight='medium')
    ax1.set_title('风速与输沙率关系', fontsize=11, fontweight='bold')
    ax1.legend(frameon=True, fancybox=False, edgecolor='#e2e8f0', fontsize=9)
    beautify_axis(ax1)

    # 风险等级分布图
    risk_data = {'极高风险': 0.15, '高风险': 0.25, '中等风险': 0.35, '安全': 0.25}
    colors_pie = ['#c53030', '#dd6b20', '#d69e2e', '#38a169']

    wedges, texts, autotexts = ax2.pie(
        risk_data.values(),
        labels=risk_data.keys(),
        colors=colors_pie,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops={'linewidth': 1, 'edgecolor': 'white'}
    )
    ax2.set_title('风险等级分布', fontsize=11, fontweight='bold')

    # 高亮当前风险等级
    risk_mapping = {'极高风险': '极高风险', '高风险': '高风险', '中低风险': '中等风险', '安全': '安全'}
    current_risk_en = risk_mapping.get(risk_level, '安全')
    for wedge, label in zip(wedges, risk_data.keys()):
        if current_risk_en == label:
            wedge.set_edgecolor('#1a365d')
            wedge.set_linewidth(2.5)
        else:
            wedge.set_alpha(0.7)

    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("评估结论与建议")

    st.markdown(f"""
    **评估结论**

    基于气象站预报风速 **{inlet_velocity} m/s** 和风速放大系数 **{k_factor}**，
    目标位置的局部实际风速为 **{local_velocity:.2f} m/s**，
    理论输沙率为 **{Q:.6f} kg/m/s**，
    环境风险等级为 **{risk_level}**。

    **防护建议**

    - 若风险等级为"极高风险"或"高风险"，建议在该区域设置防风栅栏或种植防风林带
    - 若风险等级为"中低风险"，建议定期监测风速变化，必要时采取临时防护措施
    - 若风险等级为"安全"，可维持现有环境条件

    **模型改进方向**

    建议通过现场实测数据验证模型预测结果，进一步优化风速放大系数K和Bagnold经验常数C，提高评估精度。
    """)


# =============================================================================
# 主程序
# =============================================================================
st.sidebar.title("功能导航")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "选择分析模块",
    ["深度学习风速预测", "CFD仿真结果比对", "风沙环境综合评估"],
    help="点击切换不同功能模块"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**系统信息**

西宁地区风环境分析系统

技术架构：VMD-BiLSTM-GBR

数据来源：气象站观测数据
""")

if page == "深度学习风速预测":
    page_prediction()
elif page == "CFD仿真结果比对":
    page_simulation()
else:
    page_wind_sand_evaluation()
