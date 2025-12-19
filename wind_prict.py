import warnings
warnings.filterwarnings('ignore')

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

plt.rcParams.update({
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "STHeiti", "WenQuanYi Micro Hei"],
    'axes.unicode_minus': False,
    'figure.dpi': 500,  # 提高分辨率到300 DPI
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'axes.linewidth': 1.6,
    'text.usetex': False,  # 禁用LaTeX渲染，避免乱码
    'mathtext.fontset': 'stix',  # 使用STIX数学字体，支持中文同时正确显示数学符号
    'mathtext.default': 'regular'  # 设置默认数学字体为常规
})

KNOTS_TO_MS = 0.5144
LOOK_BACK = 7
EPOCHS = 80
BATCH_SIZE = 16
FUTURE_STEPS = 120
TREND_WINDOW = 14
train_ratio = 0.8

MODEL_DIR = Path("saved_models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
AUX_SCALER_PATH = MODEL_DIR / "aux_scaler.pkl"
GB_MODEL_PATH = MODEL_DIR / "residual_gb.pkl"
META_PATH = MODEL_DIR / "model_meta.json"
FORCE_RETRAIN = False  # True 时重新训练

COLORS = {
    'actual': '#073B4C',
    'train_pred': '#FF6B35',
    'test_pred': '#118AB2',
    'future_pred': '#8ECAE6',
    'imf1': '#8D99AE',
    'imf2': '#FFD166',
    'imf3': '#EF476F',
    'scatter_train': '#118AB2',
    'scatter_test': '#FFD166',
    'residual_pos': '#43AA8B',
    'residual_neg': '#E63946'
}

def component_paths(idx):
    return (
        MODEL_DIR / f"imf{idx+1}_bilstm.keras",
        MODEL_DIR / f"imf{idx+1}_scaler.pkl"
    )

def metadata_matches(meta_path, aux_cols, look_back, k, train_ratio_val):
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
        abs(saved.get('train_ratio', train_ratio_val) - train_ratio_val) < 1e-9
    )

def beautify_axis(ax, grid_alpha=0.35):
    ax.grid(True, alpha=grid_alpha, linestyle=':')
    ax.tick_params(axis='both', labelsize=13)

def vmd_decompose(signal, alpha=2000, tau=0, K=3, DC=0, init=1, tol=1e-7, max_iter=500):
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

def smart_read_csv(file_path):
    for sep in [',', '\t', ';', ' ']:
        try:
            df = pd.read_csv(file_path, sep=sep, encoding='utf-8')
            if len(df.columns) > 3:
                return df
        except Exception:
            continue
    return pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8')

def smart_parse_date(date_series):
    formats = ['%Y/%m/%d', '%Y-%m-%d', '%Y%m%d', '%d/%m/%Y', '%m/%d/%Y']
    parsed = pd.to_datetime(date_series, errors='coerce')
    if parsed.isna().sum() > len(parsed) * 0.5:
        for fmt in formats:
            parsed = pd.to_datetime(date_series, format=fmt, errors='coerce')
            if parsed.isna().sum() < len(parsed) * 0.3:
                break
    return parsed

print("正在载入多年份观测数据...")
file_paths = [
    r"C:\Users\Lenovo\Desktop\新建文件夹 (2)\西宁气象数据2021.csv",
    r"C:\Users\Lenovo\Desktop\新建文件夹 (2)\西宁气象数据2022.csv",
    r"C:\Users\Lenovo\Desktop\新建文件夹 (2)\西宁气象数据2023.csv",
    r"C:\Users\Lenovo\Desktop\新建文件夹 (2)\西宁气象数据2024.csv",
    r"C:\Users\Lenovo\Desktop\新建文件夹 (2)\西宁气象数据2025.csv"
]

data_frames = []
for file_path in file_paths:
    try:
        df = smart_read_csv(file_path)
        if 'DATE' in df.columns:
            df['DATE'] = smart_parse_date(df['DATE'])
        data_frames.append(df)
        year = file_path.split('数据')[-1].split('.')[0]
        valid_dates = df['DATE'].notna().sum() if 'DATE' in df.columns else len(df)
        print(f"  {year} 年记录 {len(df)} 条，有效日期 {valid_dates} 条")
    except FileNotFoundError:
        print(f"  文件缺失：{file_path}")
    except Exception as e:
        print(f"  读取异常：{e}")

if not data_frames:
    raise ValueError("未成功读取任何数据文件！")

full_data = pd.concat(data_frames, ignore_index=True).dropna(subset=['DATE'])
full_data = full_data.sort_values('DATE').reset_index(drop=True)
BASE_DATE = full_data['DATE'].min()

print(f"时间范围：{full_data['DATE'].min().date()} 至 {full_data['DATE'].max().date()}")
full_data['Year'] = full_data['DATE'].dt.year
for year, count in full_data.groupby('Year').size().items():
    print(f"  {year} 年：{count} 条")

date_diffs = full_data['DATE'].diff()
gaps = date_diffs[date_diffs > pd.Timedelta(days=7)]
if len(gaps) > 0:
    print(f"注意：发现 {len(gaps)} 处超过 7 天的观测间隔。")

core_features = ['WDSP', 'TEMP', 'SLP', 'DEWP', 'MAX', 'MIN', 'STP']
for col in core_features:
    if col in full_data.columns:
        full_data[col] = pd.to_numeric(full_data[col], errors='coerce').replace(999.9, np.nan)

if 'WDSP' in full_data.columns:
    full_data.loc[(full_data['WDSP'] > 50) | (full_data['WDSP'] < 0), 'WDSP'] = np.nan

for col in ['TEMP', 'MAX', 'MIN', 'DEWP']:
    if col in full_data.columns:
        full_data.loc[(full_data[col] > 150) | (full_data[col] < -60), col] = np.nan

for col in ['SLP', 'STP']:
    if col in full_data.columns:
        full_data.loc[(full_data[col] > 1100) | (full_data[col] < 850), col] = np.nan

for col in core_features:
    if col in full_data.columns:
        full_data[col] = full_data[col].interpolate(method='linear', limit_direction='both')
        full_data[col] = full_data[col].fillna(full_data[col].median())

full_data['WDSP_ms'] = full_data['WDSP'] * KNOTS_TO_MS
print(f"风速范围：{full_data['WDSP'].min():.2f}-{full_data['WDSP'].max():.2f} knots / "
      f"{full_data['WDSP_ms'].min():.2f}-{full_data['WDSP_ms'].max():.2f} m/s")

full_data['Time_Index'] = (full_data['DATE'] - BASE_DATE).dt.days.astype(float)
full_data['DayOfYear'] = full_data['DATE'].dt.dayofyear
full_data['Month'] = full_data['DATE'].dt.month
full_data['Sin_Year'] = np.sin(2 * np.pi * full_data['DayOfYear'] / 365)
full_data['Cos_Year'] = np.cos(2 * np.pi * full_data['DayOfYear'] / 365)
full_data['dSLP_1d'] = full_data['SLP'].diff(1).fillna(0)
full_data['dSLP_2d'] = full_data['SLP'].diff(2).fillna(0)
full_data['dTEMP_1d'] = full_data['TEMP'].diff(1).fillna(0)
full_data['dTEMP_2d'] = full_data['TEMP'].diff(2).fillna(0)
full_data['dWDSP_1d'] = full_data['WDSP_ms'].diff(1).fillna(0)
full_data['WDSP_Trend14'] = (full_data['WDSP_ms'].diff(TREND_WINDOW) / TREND_WINDOW).fillna(0)

for window in [3, 5, 7]:
    full_data[f'WDSP_MA{window}'] = full_data['WDSP_ms'].rolling(window=window, min_periods=1).mean()
    full_data[f'WDSP_STD{window}'] = full_data['WDSP_ms'].rolling(window=window, min_periods=1).std().fillna(0)
    full_data[f'SLP_MA{window}'] = full_data['SLP'].rolling(window=window, min_periods=1).mean()

for lag in [1, 2, 3]:
    full_data[f'WDSP_Lag{lag}'] = full_data['WDSP_ms'].shift(lag).fillna(method='bfill')
    full_data[f'SLP_Lag{lag}'] = full_data['SLP'].shift(lag).fillna(method='bfill')
    full_data[f'TEMP_Lag{lag}'] = full_data['TEMP'].shift(lag).fillna(method='bfill')

full_data['TEMP_Range'] = full_data['MAX'] - full_data['MIN']
full_data['TEMP_DEWP_Diff'] = full_data['TEMP'] - full_data['DEWP']
full_data = full_data.fillna(method='bfill').fillna(method='ffill')

wind_series = full_data['WDSP_ms'].values
dates = full_data['DATE'].values
K = 3
alpha = 2000

try:
    imfs, omega = vmd_decompose(wind_series, alpha=alpha, K=K)
except Exception as e:
    print(f"VMD分解失败，采用滑动均值替代：{e}")
    trend = pd.Series(wind_series).rolling(window=30, center=True, min_periods=1).mean().values
    seasonal = pd.Series(wind_series).rolling(window=7, center=True, min_periods=1).mean().values - trend
    residual = wind_series - trend - seasonal
    imfs = np.array([trend, seasonal, residual])

fig_vmd, axes_vmd = plt.subplots(K + 1, 1, figsize=(14, 3 * (K + 1)))
fig_vmd.suptitle('西宁风速VMD分解结果 (m/s)', fontsize=18, fontweight='bold')

axes_vmd[0].plot(pd.to_datetime(dates), wind_series, color=COLORS['actual'], linewidth=1.3)
axes_vmd[0].set_title('(a) 原始风速序列', fontweight='bold', loc='left')
axes_vmd[0].set_ylabel('风速 (m/s)')
beautify_axis(axes_vmd[0])

imf_titles = ['(b) IMF1 - 趋势', '(c) IMF2 - 周期', '(d) IMF3 - 噪声']
for i in range(K):
    axes_vmd[i + 1].plot(pd.to_datetime(dates), imfs[i],
                         color=[COLORS['imf1'], COLORS['imf2'], COLORS['imf3']][i],
                         linewidth=1.2)
    axes_vmd[i + 1].set_title(imf_titles[i], fontweight='bold', loc='left')
    axes_vmd[i + 1].set_ylabel('幅值 (m/s)')
    axes_vmd[i + 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    beautify_axis(axes_vmd[i + 1])
axes_vmd[-1].set_xlabel('日期')
plt.tight_layout()
plt.show()

aux_feature_cols = [
    'Sin_Year', 'Cos_Year', 'Time_Index', 'dSLP_1d', 'dSLP_2d', 'dTEMP_1d',
    'WDSP_MA3', 'WDSP_MA5', 'WDSP_STD3', 'WDSP_Lag1', 'WDSP_Lag2', 'WDSP_Lag3',
    'SLP_Lag1', 'TEMP_Lag1', 'TEMP_Range', 'dWDSP_1d', 'WDSP_Trend14'
]
aux_feature_cols = [col for col in aux_feature_cols if col in full_data.columns]
aux_features = full_data[aux_feature_cols].values

metadata_ok = metadata_matches(META_PATH, aux_feature_cols, LOOK_BACK, K, train_ratio)
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
reuse_trained_models = required_ready and not FORCE_RETRAIN

if reuse_trained_models:
    print(f"检测到已保存的最优模型，直接载入 {MODEL_DIR.resolve()} 以进行预测。")
    aux_scaler = joblib.load(AUX_SCALER_PATH)
    aux_scaled = aux_scaler.transform(aux_features)
else:
    print(f"模型文件缺失或配置发生变化，重新训练并保存到 {MODEL_DIR.resolve()} 。")
    aux_scaler = StandardScaler()
    aux_scaled = aux_scaler.fit_transform(aux_features)
    joblib.dump(aux_scaler, AUX_SCALER_PATH)

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

all_train_preds, all_test_preds = [], []
component_metrics = []
component_states = []
split_idx = int(len(wind_series) * train_ratio)

for k in range(K):
    imf_k = imfs[k]
    model_path, scaler_path = component_paths(k)

    if reuse_trained_models:
        scaler_imf = joblib.load(scaler_path)
        imf_scaled = scaler_imf.transform(imf_k.reshape(-1, 1)).flatten()
        model = load_model(model_path)
    else:
        scaler_imf = MinMaxScaler(feature_range=(-1, 1))
        imf_scaled = scaler_imf.fit_transform(imf_k.reshape(-1, 1)).flatten()
        model = Sequential([
            Bidirectional(LSTM(32, activation='tanh', return_sequences=True),
                          input_shape=(LOOK_BACK, len(aux_feature_cols) + 1)),
            Dropout(0.2),
            Bidirectional(LSTM(16, activation='tanh')),
            Dropout(0.1),
            Dense(8, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.005), loss='mse')

    X, y = create_multivariate_dataset(imf_scaled, aux_scaled, LOOK_BACK)
    train_size = split_idx - LOOK_BACK
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    if not reuse_trained_models:
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
        ]
        model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=0
        )
        model.save(model_path, include_optimizer=False)
        joblib.dump(scaler_imf, scaler_path)

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
y_train_true = wind_series[LOOK_BACK:split_idx]
y_test_true = wind_series[split_idx:]
train_dates = dates[LOOK_BACK:split_idx]
test_dates = dates[split_idx:]

train_residual = y_train_true - final_train_pred
X_train_xgb = aux_features[LOOK_BACK:split_idx]
X_test_xgb = aux_features[split_idx:]

if reuse_trained_models:
    xgb_model = joblib.load(GB_MODEL_PATH)
else:
    xgb_model = GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=42
    )
    xgb_model.fit(X_train_xgb, train_residual)
    joblib.dump(xgb_model, GB_MODEL_PATH)
    metadata = {
        'LOOK_BACK': LOOK_BACK,
        'K': K,
        'aux_cols': aux_feature_cols,
        'train_ratio': train_ratio
    }
    with open(META_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"模型及缩放器已保存，后续可直接读取 {MODEL_DIR.resolve()} 中的最优结果。")

final_train_pred_corrected = final_train_pred + xgb_model.predict(X_train_xgb)
final_test_pred_corrected = final_test_pred + xgb_model.predict(X_test_xgb)

future_pred, future_dates, future_aux_features = forecast_future(
    component_states, aux_scaler, aux_feature_cols, full_data, wind_series, FUTURE_STEPS, BASE_DATE, TREND_WINDOW
)
future_pred_corrected = future_pred + xgb_model.predict(future_aux_features)

def calculate_metrics(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    print(f"{dataset_name} -> R²:{r2:.4f} | RMSE:{rmse:.4f} m/s | MAE:{mae:.4f} m/s | MAPE:{mape:.2f}% | Corr:{corr:.4f}")
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2, 'Corr': corr}

print("\n模型评价指标：")
train_metrics_base = calculate_metrics(y_train_true, final_train_pred, "训练集(基础)")
test_metrics_base = calculate_metrics(y_test_true, final_test_pred, "测试集(基础)")
train_metrics = calculate_metrics(y_train_true, final_train_pred_corrected, "训练集(修正)")
test_metrics = calculate_metrics(y_test_true, final_test_pred_corrected, "测试集(修正)")
print(f"未来{FUTURE_STEPS}天预测完成，预测范围：{future_pred_corrected.min():.2f} - {future_pred_corrected.max():.2f} m/s")

fig1, ax1 = plt.subplots(figsize=(16, 6))
all_dates = np.concatenate([train_dates, test_dates])
all_actual = np.concatenate([y_train_true, y_test_true])

ax1.plot(pd.to_datetime(all_dates), all_actual,
         color=COLORS['actual'], linewidth=1.9, label='实际值')
ax1.plot(pd.to_datetime(train_dates), final_train_pred_corrected,
         color=COLORS['train_pred'], linewidth=1.5, linestyle='--',
         label=f'训练拟合 ($R²$={train_metrics["R2"]:.3f})')
ax1.plot(pd.to_datetime(test_dates), final_test_pred_corrected,
         color=COLORS['test_pred'], linewidth=1.7, linestyle='-.',
         label=f'测试预测 ($R²$={test_metrics["R2"]:.3f})')
ax1.plot(future_dates, future_pred_corrected,
         color=COLORS['future_pred'], linewidth=1.8, linestyle='--',
         label=f'未来{FUTURE_STEPS}天预测')
split_date = pd.to_datetime(test_dates[0])
ax1.axvline(x=split_date, color='#7F8C8D', linestyle=':', linewidth=2.2, alpha=0.85, label='训练/测试分界')
ax1.axvspan(pd.to_datetime(test_dates[-1]), future_dates[-1], color=COLORS['future_pred'], alpha=0.08, label='未来预测区')
ax1.set_xlabel('日期')
ax1.set_ylabel('风速 (m/s)')
ax1.set_title('西宁地区日平均风速预测与未来展望', fontweight='bold')
ax1.legend(loc='upper right', frameon=True)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.tick_params(axis='x', rotation=20)
beautify_axis(ax1)
plt.tight_layout()
plt.show()

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
fig2.suptitle('西宁地区风速预测详细分析 (m/s)', fontsize=18, fontweight='bold', y=0.98)

ax2_1 = axes2[0, 0]
ax2_1.plot(pd.to_datetime(train_dates), y_train_true, color=COLORS['actual'], linewidth=1.5, label='实际值')
ax2_1.plot(pd.to_datetime(train_dates), final_train_pred_corrected,
           color=COLORS['train_pred'], linewidth=1.3, linestyle='--', label='拟合值')
ax2_1.set_title(f'(a) 训练集拟合 ($R²$={train_metrics["R2"]:.4f})', fontweight='bold', loc='left')
ax2_1.set_xlabel('日期')
ax2_1.set_ylabel('风速 (m/s)')
ax2_1.legend(frameon=True)
ax2_1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2_1.tick_params(axis='x', rotation=20)
beautify_axis(ax2_1)

ax2_2 = axes2[0, 1]
ax2_2.plot(pd.to_datetime(test_dates), y_test_true, color=COLORS['actual'], linewidth=1.7, label='实际值')
ax2_2.plot(pd.to_datetime(test_dates), final_test_pred_corrected,
           color=COLORS['test_pred'], linewidth=1.4, linestyle='--', label='预测值')
ax2_2.set_title(f'(b) 测试集预测 ($R²$={test_metrics["R2"]:.4f})', fontweight='bold', loc='left')
ax2_2.set_xlabel('日期')
ax2_2.set_ylabel('风速 (m/s)')
ax2_2.legend(frameon=True)
ax2_2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2_2.tick_params(axis='x', rotation=20)
beautify_axis(ax2_2)

ax2_3 = axes2[1, 0]
ax2_3.scatter(y_train_true, final_train_pred_corrected,
              c=COLORS['scatter_train'], alpha=0.5, s=35, edgecolors='none', label='训练集')
ax2_3.scatter(y_test_true, final_test_pred_corrected,
              c=COLORS['scatter_test'], alpha=0.6, s=40, edgecolors='none', label='测试集')
min_val = min(all_actual.min(), final_test_pred_corrected.min()) - 0.2
max_val = max(all_actual.max(), final_test_pred_corrected.max()) + 0.2
ax2_3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='理想线')
ax2_3.set_xlabel('实际风速 (m/s)')
ax2_3.set_ylabel('预测风速 (m/s)')
ax2_3.set_title('(c) 实际-预测散点图', fontweight='bold', loc='left')
ax2_3.legend(frameon=True)
ax2_3.set_xlim([min_val, max_val])
ax2_3.set_ylim([min_val, max_val])
ax2_3.set_aspect('equal', adjustable='box')
beautify_axis(ax2_3)

ax2_4 = axes2[1, 1]
display_len = min(80, len(test_dates))
ax2_4.plot(pd.to_datetime(test_dates[:display_len]), y_test_true[:display_len],
           color=COLORS['actual'], linewidth=2, marker='o', markersize=4, label='实际值')
ax2_4.plot(pd.to_datetime(test_dates[:display_len]), final_test_pred_corrected[:display_len],
           color=COLORS['test_pred'], linewidth=1.6, linestyle='--', marker='^', markersize=4, label='预测值')
ax2_4.set_title(f'(d) 测试集局部放大 (前{display_len}天)', fontweight='bold', loc='left')
ax2_4.set_xlabel('日期')
ax2_4.set_ylabel('风速 (m/s)')
ax2_4.legend(frameon=True)
ax2_4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2_4.tick_params(axis='x', rotation=35)
beautify_axis(ax2_4)

plt.tight_layout()
plt.show()

fig3, axes3 = plt.subplots(1, 3, figsize=(16, 4.5))
fig3.suptitle('各IMF分量预测性能', fontsize=17, fontweight='bold')
imf_full_names = ['IMF1 (趋势)', 'IMF2 (周期)', 'IMF3 (噪声)']
bar_colors = [COLORS['imf1'], COLORS['imf2'], COLORS['imf3']]
for i, metrics in enumerate(component_metrics):
    ax = axes3[i]
    bars = ax.bar(['训练集', '测试集'],
                  [metrics['r2_train'], metrics['r2_test']],
                  color=bar_colors[i], alpha=0.85, edgecolor='black', linewidth=1.1)
    ax.set_title(f'{imf_full_names[i]} - $R²$', fontweight='bold')
    ax.set_ylabel('$R²$')
    ax.set_ylim([0, 1.05])
    beautify_axis(ax)
    for bar, val in zip(bars, [metrics['r2_train'], metrics['r2_test']]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()

fig4, axes4 = plt.subplots(1, 2, figsize=(15, 5.5))
fig4.suptitle('预测残差分析 (m/s)', fontsize=17, fontweight='bold')

residuals = y_test_true - final_test_pred_corrected
ax4_1 = axes4[0]
ax4_1.plot(pd.to_datetime(test_dates), residuals, color='#5C6784', linewidth=1.1)
ax4_1.axhline(y=0, color=COLORS['test_pred'], linestyle='--', linewidth=2.2)
ax4_1.fill_between(pd.to_datetime(test_dates), residuals, 0,
                   where=(residuals > 0), color=COLORS['residual_pos'], alpha=0.35, label='正残差')
ax4_1.fill_between(pd.to_datetime(test_dates), residuals, 0,
                   where=(residuals < 0), color=COLORS['residual_neg'], alpha=0.35, label='负残差')
ax4_1.set_title('(a) 测试集残差时序', fontweight='bold', loc='left')
ax4_1.set_xlabel('日期')
ax4_1.set_ylabel('残差 (m/s)')
ax4_1.legend(frameon=True)
ax4_1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax4_1.tick_params(axis='x', rotation=20)
beautify_axis(ax4_1)

ax4_2 = axes4[1]
ax4_2.hist(residuals, bins=30, color=COLORS['actual'], edgecolor='white', alpha=0.82)
ax4_2.axvline(x=0, color=COLORS['test_pred'], linestyle='--', linewidth=2.2, label='零线')
ax4_2.axvline(x=np.mean(residuals), color=COLORS['train_pred'], linestyle='-', linewidth=2,
              label=f'均值: {np.mean(residuals):.3f}')
ax4_2.set_title(f'(b) 残差分布 (σ={np.std(residuals):.3f} m/s)', fontweight='bold', loc='left')
ax4_2.set_xlabel('残差 (m/s)')
ax4_2.set_ylabel('频数')
ax4_2.legend(frameon=True)
beautify_axis(ax4_2)

plt.tight_layout()
plt.show()

fig5, ax5 = plt.subplots(figsize=(16, 5.5))
history_len = min(60, len(test_dates))
if history_len > 0:
    ax5.plot(pd.to_datetime(test_dates[-history_len:]), y_test_true[-history_len:],
             color=COLORS['actual'], linewidth=1.8, marker='o', markersize=3, label='近期实际')
    ax5.plot(pd.to_datetime(test_dates[-history_len:]), final_test_pred_corrected[-history_len:],
             color=COLORS['test_pred'], linewidth=1.4, linestyle='--', marker='^', markersize=3, label='近期预测')
ax5.plot(future_dates, future_pred_corrected,
         color=COLORS['future_pred'], linewidth=2.1, linestyle='-', marker='s', markersize=3, label='未来预测')
ax5.axvspan(future_dates[0], future_dates[-1], color=COLORS['future_pred'], alpha=0.08)
ax5.set_title(f'未来{FUTURE_STEPS}天风速预测', fontweight='bold')
ax5.set_xlabel('日期')
ax5.set_ylabel('风速 (m/s)')
ax5.legend(frameon=True)
ax5.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax5.tick_params(axis='x', rotation=30)
beautify_axis(ax5)
plt.tight_layout()
plt.show()

print("\n分量表现：")
for m in component_metrics:
    print(f"  {m['name']} -> 训练R²={m['r2_train']:.4f} | 测试R²={m['r2_test']:.4f}")

print(f"""
最终指标：
  训练集  R²={train_metrics['R2']:.4f}  RMSE={train_metrics['RMSE']:.4f} m/s  MAE={train_metrics['MAE']:.4f} m/s
  测试集  R²={test_metrics['R2']:.4f}  RMSE={test_metrics['RMSE']:.4f} m/s  MAE={test_metrics['MAE']:.4f} m/s
""")
