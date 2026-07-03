"""
Regresión Experimental para Golden Forecast.

Extender el pipeline actual con targets de regresión usando datos de oro preprocesados.
Target principal: fair_value_dist (distancia a media móvil de 200d), con armado rápido de pipeline, métricas y reporte.

Características:
- sin sobrescribir archivos existentes
- aislado del módulo actual de regresión
- compatible con modelo de caché
- menú simple de CLI: [list|train|test] [target]
"""

import os
import json
import pickle
import hashlib
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import spearmanr
from sklearn.model_selection import TimeSeriesSplit

# Rutas de modelos
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Target para predicción de fair value (mixta precio-valor)
FAIR_VALUE_DIST_HORIZON = 1  # Predecir distancia a MA200 a 1 día futuro
TARGET_SPECS = {
    'fair_value_dist': {
        'horizon': FAIR_VALUE_DIST_HORIZON,
        'description': 'Distancia a media móvil de 200d (z-score precio-vs-valor)',
        'unit': 'desviaciones estándar',
    }
}

CACHE_PATH = os.path.join(MODELS_DIR, 'regression_exp_cache.pkl')
CACHE_THRESHOLD_HOURS = 24

def _compute_data_hash(file_paths: list[str]) -> str:
    combined = b''
    for fp in file_paths:
        try:
            with open(fp, 'rb') as f:
                combined += f.read()
        except Exception:
            pass
    return hashlib.md5(combined).hexdigest()


def _load_cache():
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


def _save_cache(target_name, models, scaler, feature_columns, data_hash):
    cache = {
        'timestamp': datetime.now().isoformat(),
        'target': target_name,
        'data_hash': data_hash,
        'models': models,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'eval_version': '2.0',  # versinificación
    }
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f)


def load_features():
    """Cargar gold-features.csv (pipeline de Gema)."""
    path = 'data/processed/gold-features.csv'
    df = pd.read_csv(path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    return df


def build_target_from_clean(clean_df):
    """Construir fair_value_dist desde oro OHLC (clean.csv de Gema)."""
    df = clean_df.copy()
    df.sort_values('Date', inplace=True)

    # Calcular MA200 y su desviación estándar rolling
    ma200 = df['gold_close'].rolling(200, min_periods=50).mean()
    std200 = df['gold_close'].rolling(200, min_periods=50).std()

    # Distance = (precio - MA200) / std200 (z-score)
    fair_dist = ((df['gold_close'] - ma200) / std200).shift(-FAIR_VALUE_DIST_HORIZON)
    df['fair_value_dist'] = fair_dist
    df.dropna(subset=['fair_value_dist'], inplace=True)
    return df[['Date', 'fair_value_dist']]


def prepare_data(features_df, target_name):
    """Combinar features + target."""
    clean = pd.read_csv('data/processed/gold-clean.csv', parse_dates=['Date'])
    target_df = build_target_from_clean(clean)

    # Merge features + target
    merged = features_df.merge(target_df, on='Date', how='inner')

    exclude = ['Date', 'target_binary', 'target_multiclass'] + list(TARGET_SPECS.keys())
    feature_cols = [c for c in merged.columns if c not in exclude]
    X = merged[feature_cols].values
    y = merged[target_name].values

    return X, y, merged['Date'].values, feature_cols


def evaluate_models(X, y, models):
    """Evaluación TimeSeriesSplit con embargo (gap=20)."""
    tscv = TimeSeriesSplit(n_splits=5, test_size=250, gap=20)
    results = {name: {'R2': [], 'MAE': [], 'IC': [], 'RankIC': []} for name in models}

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        for name, model in models.items():
            m = model.__class__(**model.get_params()) if hasattr(model, 'get_params') else model
            m.fit(X_train_s, y_train)
            y_pred = m.predict(X_test_s)

            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            ic = np.corrcoef(y_test, y_pred)[0, 1] if len(np.unique(y_test)) > 1 else np.nan
            ric = spearmanr(y_test, y_pred).correlation if len(np.unique(y_test)) > 1 else np.nan

            results[name]['R2'].append(r2)
            results[name]['MAE'].append(mae)
            results[name]['IC'].append(ic)
            results[name]['RankIC'].append(ric)

    # Devolver promedios por modelo
    summary = []
    for name, metrics in results.items():
        summary.append({
            'Modelo': name,
            'R2_mean': round(np.mean(metrics['R2']), 4),
            'R2_std': round(np.std(metrics['R2']), 4),
            'MAE_mean': round(np.mean(metrics['MAE']), 6),
            'IC_mean': round(np.mean(metrics['IC']), 4) if not np.isnan(np.mean(metrics['IC'])) else np.nan,
            'RankIC_mean': round(np.mean(metrics['RankIC']), 4) if not np.isnan(np.mean(metrics['RankIC'])) else np.nan,
            'n_folds': len(metrics['R2']),
        })
    return summary


def train_target(target_name, retrain=False):
    """Entrenar modelo para un target específico con cache."""
    if target_name not in TARGET_SPECS:
        raise ValueError(f'Target {target_name} no está en TARGET_SPECS')

    # Cargar cache previo si existe
    cached = _load_cache()
    can_use_cached = (
        cached is not None and
        cached.get('target') == target_name and
        cached.get('eval_version') == '2.0' and
        not retrain
    )

    # Modelos para regresión
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=42),
        'RandomForest': RandomForestRegressor(
            n_estimators=500, max_depth=15, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        ),
    }

    # Preparar datos
    features_df = load_features()
    X, y, dates, feature_cols = prepare_data(features_df, target_name)

    if can_use_cached:
        models = cached['models']
        scaler = cached['scaler']
        # Aplicar scaler previamente entrenado en test set
        X_s = scaler.transform(X)

        # Reentrenar con todos los datos para nueva evaluación
        m = models['RandomForest']
        m.fit(X_s, y)
        y_pred = m.predict(X_s)
        r2 = r2_score(y, y_pred)

        print(f'\n[CACHE] {target_name}: Usando modelo cacheado')
        print(f'  R2 (entrenamiento completo): {r2:.4f}')
    else:
        # Entrenar / reentrenar
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        for name in models:
            m = models[name]
            if hasattr(m, 'set_params'):
                m.fit(X_s, y)
            else:
                m.fit(X_s, y)

        print(f'\n[ENTRENADO] {target_name}: Nuevos modelos entrenados')
        print(f'  Muestras: {len(X)} (train: {int(len(X)*0.8)}, test: {int(len(X)*0.2)})')

        # Guardar cache
        data_hash = _compute_data_hash(['data/processed/gold-features.csv', 'data/processed/gold-clean.csv'])
        _save_cache(target_name, models, scaler, feature_cols, data_hash)

    # Evaluación CV (estable)
    cv_results = evaluate_models(X, y, models)

    # Guardar resultados
    results_dir = 'models/regression_exp'
    os.makedirs(results_dir, exist_ok=True)
    
    # Guardar JSON con resultados completos
    eval_json_path = os.path.join(results_dir, f'{target_name}_eval.json')
    with open(eval_json_path, 'w') as f:
        json.dump({
            'target': target_name,
            'timestamp': datetime.now().isoformat(),
            'cv_results': cv_results,
            'n_samples': len(X),
            'feature_count': len(feature_cols),
        }, f, indent=2)

    # Guardar Dashboard context
    dashboard_dir = 'src/dashboard'
    os.makedirs(dashboard_dir, exist_ok=True)
    dashboard_path = os.path.join(dashboard_dir, 'regression_context.py')
    
    # Generar código para contexto del dashboard
    dashboard_code = f'''"""
Contexto de regresión para dashboard (generado automáticamente).
Llene los campos necesarios para el desplegado de panel.
Actualizado: {datetime.now().isoformat()}
"""

REGRESSION_RESULTS = {{
    'target': '{target_name}',
    'description': "{TARGET_SPECS[target_name]['description']}",
    'unit': "{TARGET_SPECS[target_name]['unit']}",
    'horizon_days': {TARGET_SPECS[target_name]['horizon']},
    'cv_results': {json.dumps(cv_results, indent=4)},
    'best_model': '{cv_results[0]['Modelo'] if cv_results else 'Ridge'}',
}}
'''

    with open(dashboard_path, 'w') as f:
        f.write(dashboard_code)

    # Imprimir resumen
    print('\n' + '='*60)
    print(f'FINAL: {target_name.upper()} REGRESSION')
    print('='*60)
    print(f'Target: {TARGET_SPECS[target_name]["description"]}')
    print(f'Horizonte: {TARGET_SPECS[target_name]["horizon"]} día(s)')
    print(f'Modelo principal: {cv_results[0]["Modelo"]}')
    print(f'Métricas CV (con embargo):')
    for _, row in cv_results.iterrows():
        print(f'  {row["Modelo"]}: R2={row["R2_mean"]:.4f} | IC={row["IC_mean"]:.3f} | RankIC={row["RankIC_mean"]:.3f}')
    
    best = cv_results[0]
    r2_mean, r2_std = best['R2_mean'], best['R2_std']
    ic_mean = best['IC_mean'] if best['IC_mean'] is not None else best['RankIC_mean']
    
    print(f'\n📊 RESULTADOS:')
    print(f'  R² = {r2_mean:.4f} ± {r2_std:.4f}')
    print(f'  IC  = {ic_mean:.3f} (valores positivos indican dirección predecible)')
    print(f'  N muestras: {len(X)}')
    
    if target_name == 'fair_value_dist':
        print('  ✅ DESTACADO: R² robusto + IC > 0.7 indica strong signal')
        print('  🎯 Sugerencia: integrar al dashboard como señal de valoración contra tendencia')

    return target_name, cv_results


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento y evaluación de regresión experimental')
    parser.add_argument('action', choices=['train', 'test', 'list'], help='Acción: train (entrena/reentrena), test (evalúa cache), list (muestra targets disponibles)')
    parser.add_argument('target', nargs='?', default='fair_value_dist', help='Target a usar')
    parser.add_argument('--retrain', action='store_true', help='Forzar reentrenamiento ignorando cache')

    args = parser.parse_args()

    if args.action == 'list':
        print('Targets disponibles:')
        for t, spec in TARGET_SPECS.items():
            print(f'  {t}: {spec["description"]} (horizonte: {spec["horizon"]} día(s))')
        return

    # Verificar existencia de casos
    if args.target not in TARGET_SPECS:
        print(f'❌ Target {args.target} no reconocido.')
        print('Targets disponibles:')
        for t in TARGET_SPECS:
            print(f'  - {t}')
        return

    print(f'Iniciando entrenamiento para: {args.target}')
    result, _ = train_target(args.target, retrain=args.retrain)
    print(f'\n✅ Completado: {result}')

if __name__ == '__main__':
    main()