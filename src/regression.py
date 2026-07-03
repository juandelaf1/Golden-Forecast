"""
Módulo de regresión para Golden Forecast — Experimentos aislados.

Predice targets continuos con significado económico construidos a partir
de los datos OHLC de Gema (gold-clean.csv), usando features de gold-features.csv.

Targets (forward-looking, sin data leakage):
- realized_vol_20d: volatilidad realizada 20d forward
- future_atr_20d: rango real medio 20d forward (ATR proxy)
- max_drawdown_20d: máximo drawdown en ventana 20d forward
- fair_value_dist: distancia a MA200 (valor relativo)

Todo self-contained: lee CSVs de Gema, construye targets en memoria,
entrena, evalúa, cachea. Sin archivos intermedios en data/processed/.
"""

import os
import pickle
import hashlib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# --------------------------
# Configuración de caché
# --------------------------
CACHE_PATH = 'models/regression_exp_cache.pkl'
CACHE_THRESHOLD_HOURS = 24

TARGET_SPECS = {
    'realized_vol_20d': {
        'horizon': 20,
        'description': 'Volatilidad realizada 20d forward (std retornos)',
        'unit': 'decimal (ej. 0.15 = 15% anual)',
    },
    'future_atr_20d': {
        'horizon': 20,
        'description': 'ATR proxy 20d forward (rango medio normalizado)',
        'unit': 'decimal (fracción del precio)',
    },
    'max_drawdown_20d': {
        'horizon': 20,
        'description': 'Máximo drawdown en ventana 20d forward',
        'unit': 'decimal (pérdida fraccional desde pico)',
    },
    'fair_value_dist': {
        'horizon': 1,
        'description': 'Distancia a MA200 (z-score precio vs media 200d)',
        'unit': 'desviaciones estándar',
    },
}


def _compute_data_hash(file_paths: list[str]) -> str:
    """Hash combinado de múltiples archivos para detectar cambios."""
    combined = b''
    for fp in file_paths:
        try:
            with open(fp, 'rb') as f:
                combined += f.read()
        except Exception:
            pass
    return hashlib.md5(combined).hexdigest()


def _load_cache() -> dict | None:
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        age = datetime.now() - datetime.fromisoformat(cache.get('timestamp', '1970-01-01T00:00:00+00:00'))
        if age < timedelta(hours=CACHE_THRESHOLD_HOURS):
            return cache
    except Exception:
        pass
    return None


def _save_cache(target_name: str, models: dict, scaler, feature_columns, data_hash: str):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    cache = {
        'timestamp': datetime.now().isoformat(),
        'target': target_name,
        'data_hash': data_hash,
        'models': models,
        'scaler': scaler,
        'feature_columns': feature_columns,
    }
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)


# --------------------------
# Construcción de targets (desde gold-clean.csv OHLC)
# --------------------------
def build_regression_targets(clean_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye todos los targets forward-looking desde OHLC limpio.
    Devuelve DataFrame con Date + targets (sin features).
    """
    df = clean_df.copy()
    df = df.sort_values('Date').reset_index(drop=True)

    # Retornos base
    df['ret'] = df['gold_close'].pct_change()

    # 1. Realized Vol 20d forward
    df['realized_vol_20d'] = df['ret'].rolling(20).std().shift(-20) * np.sqrt(252)

    # 2. Future ATR 20d forward (rango medio normalizado)
    daily_range = (df['gold_high'] - df['gold_low']) / df['gold_open']
    df['future_atr_20d'] = daily_range.rolling(20).mean().shift(-20)

    # 3. Max Drawdown 20d forward
    def max_dd_window(prices, window):
        roll_max = prices.rolling(window, min_periods=1).max()
        dd = (prices - roll_max) / roll_max
        return dd.rolling(window, min_periods=1).min()
    df['max_drawdown_20d'] = max_dd_window(df['gold_close'], 20).shift(-20)

    # 4. Fair Value Distance (distancia a MA200 en z-score)
    ma200 = df['gold_close'].rolling(200, min_periods=50).mean()
    std200 = df['gold_close'].rolling(200, min_periods=50).std()
    df['fair_value_dist'] = ((df['gold_close'] - ma200) / std200).shift(-1)

    # Solo targets + Date
    target_cols = ['Date'] + list(TARGET_SPECS.keys())
    return df[target_cols].dropna().reset_index(drop=True)


# --------------------------
# Modelos
# --------------------------
def get_regression_models():
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=15,
            random_state=42, n_jobs=-1
        ),
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'Linear Regression': LinearRegression(),
        'Lasso': Lasso(alpha=0.01, random_state=42, max_iter=5000),
        'ElasticNet': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=42, max_iter=5000),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42
        ),
    }
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            random_state=42, verbosity=0, n_jobs=-1
        )
    return models


def get_best_model_for_target(target_name: str):
    """
    Devuelve el mejor modelo configurado para un target específico.
    Basado en validación cruzada TimeSeriesSplit (gap=20).
    """
    best_configs = {
        'fair_value_dist': RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=15,
            random_state=42, n_jobs=-1
        ),
        'max_drawdown_20d': Ridge(alpha=1.0, random_state=42),
        'realized_vol_20d': Ridge(alpha=1.0, random_state=42),
        'future_atr_20d': Ridge(alpha=1.0, random_state=42),
    }
    return best_configs.get(target_name, RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=15,
        random_state=42, n_jobs=-1
    ))


# --------------------------
# Métricas
# --------------------------
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def information_coefficient(y_true, y_pred):
    """Rank IC (Spearman) + Pearson IC."""
    from scipy.stats import spearmanr, pearsonr
    rho, _ = spearmanr(y_true, y_pred)
    r, _ = pearsonr(y_true, y_pred)
    return float(rho), float(r)


def evaluate_regression(models, X_train, X_test, y_train, y_test):
    results = []
    for name, model in models.items():
        m = model.__class__(**model.get_params())
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        ic_rank, ic_pearson = information_coefficient(y_test, y_pred)

        results.append({
            'Modelo': name,
            'MAE': round(mae, 6),
            'RMSE': round(rmse, 6),
            'R²': round(r2, 4),
            'MAPE (%)': round(mape, 2) if not np.isnan(mape) else np.nan,
            'IC (Pearson)': round(ic_pearson, 4),
            'Rank IC (Spearman)': round(ic_rank, 4),
        })
    return pd.DataFrame(results)


def evaluate_regression_cv(models, X, y, n_splits=5, test_size=250, gap=20):
    """
    Evaluación con TimeSeriesSplit (gap=20 para purgar solapamiento de features rolling).
    Devuelve métricas promedio por modelo across folds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    results = {name: {'R²': [], 'MAE': [], 'RMSE': [], 'IC': [], 'RankIC': []} for name in models}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for name, model in models.items():
            m = model.__class__(**model.get_params())
            m.fit(X_train_s, y_train)
            y_pred = m.predict(X_test_s)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            ic_p, ic_s = information_coefficient(y_test, y_pred)

            results[name]['R²'].append(r2)
            results[name]['MAE'].append(mae)
            results[name]['RMSE'].append(rmse)
            results[name]['IC'].append(ic_p)
            results[name]['RankIC'].append(ic_s)

    # Promediar
    summary = []
    for name, metrics in results.items():
        summary.append({
            'Modelo': name,
            'R²_mean': round(np.mean(metrics['R²']), 4),
            'R²_std': round(np.std(metrics['R²']), 4),
            'MAE_mean': round(np.mean(metrics['MAE']), 6),
            'RMSE_mean': round(np.mean(metrics['RMSE']), 6),
            'IC_mean': round(np.mean(metrics['IC']), 4),
            'IC_std': round(np.std(metrics['IC']), 4),
            'RankIC_mean': round(np.mean(metrics['RankIC']), 4),
            'RankIC_std': round(np.std(metrics['RankIC']), 4),
            'n_folds': len(metrics['R²']),
        })
    return pd.DataFrame(summary)


# --------------------------
# Pipeline por target
# --------------------------
def train_and_evaluate_target(
    target_name: str,
    features_path='data/processed/gold-features.csv',
    clean_path='data/processed/gold-clean.csv',
    test_size=0.2,
) -> dict:
    """
    Entrena y evalúa modelos para UN target específico.
    """
    if target_name not in TARGET_SPECS:
        raise ValueError(f'Target desconocido: {target_name}. Disponibles: {list(TARGET_SPECS.keys())}')

    # Validar archivos
    for p in [features_path, clean_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f'No existe {p}')

    # Hash combinado
    data_hash = _compute_data_hash([features_path, clean_path])

    # Caché
    cached = _load_cache()
    use_cache = cached is not None and cached.get('data_hash') == data_hash and cached.get('target') == target_name

    # Cargar datos base
    features = pd.read_csv(features_path, parse_dates=['Date'])
    clean = pd.read_csv(clean_path, parse_dates=['Date'])

    # Construir targets
    targets_df = build_regression_targets(clean)

    # Merge features + target
    data = features.merge(targets_df, on='Date', how='inner')
    data = data.sort_values('Date').reset_index(drop=True)

    # Preparar X, y
    exclude = ['Date', 'target_binary', 'target_multiclass'] + list(TARGET_SPECS.keys())
    feature_cols = [c for c in data.columns if c not in exclude]
    X = data[feature_cols].values
    y = data[target_name].values

    # Split temporal
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenamiento
    if use_cache:
        models = cached.get('models', {})
        for name in models:
            models[name] = models[name].__class__(**models[name].get_params())
        for model in models.values():
            model.fit(X_train_scaled, y_train)
    else:
        models = get_regression_models()
        for name in models:
            models[name] = models[name].__class__(**models[name].get_params())
        for model in models.values():
            model.fit(X_train_scaled, y_train)
        _save_cache(target_name, models, scaler, feature_cols, data_hash)

    # Evaluación single split (backward compat)
    eval_df = evaluate_regression(models, X_train_scaled, X_test_scaled, y_train, y_test)

    # Evaluación CV robusta (TimeSeriesSplit con gap=20)
    print(f"  Running CV (TimeSeriesSplit, n_splits=5, gap=20)...")
    cv_df = evaluate_regression_cv(models, X, y, n_splits=5, test_size=250, gap=20)

    # Mejor modelo por CV R²_mean
    best_idx = cv_df['R²_mean'].idxmax()
    best_model_name = cv_df.loc[best_idx, 'Modelo']
    best_model = models[best_model_name]
    best_model.fit(X_train_scaled, y_train)

    return {
        'target': target_name,
        'target_info': TARGET_SPECS[target_name],
        'eval_results': eval_df,
        'cv_results': cv_df,
        'best_model_name': best_model_name,
        'best_model': best_model,
        'scaler': scaler,
        'feature_columns': feature_cols,
        'data_hash': data_hash,
        'cached': use_cache,
        'n_train': len(X_train),
        'n_test': len(X_test),
    }


def run_all_targets(
    features_path='data/processed/gold-features.csv',
    clean_path='data/processed/gold-clean.csv',
    test_size=0.2,
) -> dict:
    """Ejecuta pipeline completo para todos los targets."""
    results = {}
    for target in TARGET_SPECS.keys():
        print(f'\n=== Entrenando: {target} ===')
        results[target] = train_and_evaluate_target(target, features_path, clean_path, test_size)
        r = results[target]
        print(f"  Mejor: {r['best_model_name']} | R²={r['eval_results'].loc[r['eval_results']['R²'].idxmax(), 'R²']:.4f} | IC={r['eval_results'].loc[r['eval_results']['R²'].idxmax(), 'IC (Pearson)']:.4f}")
    return results


# --------------------------
# CLI
# --------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        target = sys.argv[1]
        r = train_and_evaluate_target(target)
    else:
        r = run_all_targets()
    print('\n=== RESUMEN ===')
    if isinstance(r, dict) and 'eval_results' in r:
        # single target
        print('--- Single Split ---')
        print(r['eval_results'].to_string(index=False))
        print('\n--- CV (TimeSeriesSplit, gap=20) ---')
        print(r['cv_results'].to_string(index=False))
    else:
        # all targets
        for t, res in r.items():
            best = res['cv_results'].loc[res['cv_results']['R²_mean'].idxmax()]
            print(f"{t:20s} | R²={best['R²_mean']:>7.4f}±{best['R²_std']:.4f} | IC={best['IC_mean']:>6.3f}±{best['IC_std']:.3f} | RankIC={best['RankIC_mean']:>6.3f}±{best['RankIC_std']:.3f} | Best={best['Modelo']}")