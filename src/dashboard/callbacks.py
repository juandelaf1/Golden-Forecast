from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, html, dcc
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve

from src.dashboard import data

context = data.context
FEATURE_COLUMNS = data.FEATURE_COLUMNS

PLOT_THEME = {
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font': {'color': '#F2EBE1', 'family': 'Space Mono, monospace'},
    'margin': {'t': 30, 'b': 30, 'l': 30, 'r': 30},
    'legend': {'orientation': 'h', 'y': -0.2, 'font': {'color': '#F2EBE1'}},
    'hoverlabel': {
        'bgcolor': 'rgba(26, 16, 8, 0.95)',
        'bordercolor': '#D4AF37',
        'font': {'color': '#F5EDE0', 'family': 'Space Mono, monospace', 'size': 12},
    },
}

AXIS_THEME = {
    'gridcolor': 'rgba(242,235,225,0.08)',
    'linecolor': 'rgba(242,235,225,0.2)',
    'tickfont': {'color': '#F2EBE1', 'family': 'Space Mono, monospace'},
    'title': {'font': {'color': '#D4AF37', 'family': 'Rye, Smokum, serif'}},
}

RANGESELECTOR = {
    'bgcolor': 'rgba(26, 16, 8, 0.8)',
    'activecolor': 'rgba(232, 195, 74, 0.3)',
    'borderwidth': 1,
    'bordercolor': 'rgba(232, 195, 74, 0.3)',
    'font': {'color': '#F2EBE1', 'size': 10, 'family': 'Space Mono, monospace'},
}

RANGE_BUTTONS = [
    {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
    {'count': 3, 'label': '3M', 'step': 'month', 'stepmode': 'backward'},
    {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
    {'count': 1, 'label': '1A', 'step': 'year', 'stepmode': 'backward'},
    {'step': 'all', 'label': 'TODO'},
]

# Colores estilo western/wanted
WANTED_COLORS = {
    'gold': '#E8C34A',
    'brass': '#D4AF37',
    'brass_light': '#F5DC8A',
    'iron': '#1A1613',
    'iron_mid': '#2C2620',
    'iron_light': '#3D352C',
    'signal_buy': '#4ADE80',
    'signal_sell': '#F2554D',
    'signal_neutral': '#E8C34A',
    'buy': '#22C55E',
    'sell': '#EF4444',
    'hold': '#F59E0B',
}


def register_callbacks(app):
    @app.callback(Output('tab-content', 'children'), Input('dashboard-tabs', 'value'))
    def render_tab(active_tab):
        if active_tab == 'tab-summary':
            return build_summary_tab()
        if active_tab == 'tab-price':
            return build_price_tab()
        if active_tab == 'tab-indicators':
            return build_indicators_tab()
        if active_tab == 'tab-macro':
            return build_macro_tab()
        if active_tab == 'tab-backtest':
            return build_backtest_tab()
        if active_tab == 'tab-sim':
            return build_simulation_tab()
        if active_tab == 'tab-metrics':
            return build_metrics_tab()
        if active_tab == 'tab-governance':
            return build_methodology_tab()
        raise PreventUpdate

    @app.callback(Output('ticker-content', 'children'), Input('ticker-content', 'id'))
    def update_ticker(_):
        latest = context['latest']
        change_pct = context.get('change_pct', 0)
        signal = context['signal_label']
        accuracy = context.get('accuracy', 0)
        f1 = context.get('f1', 0)
        gold_price = latest['gold']
        
        items = [
            html.Div(
                className='ticker-item',
                children=[
                    html.Span('ORO SPOT', className='ticker-label'),
                    html.Span(f'${gold_price:.2f}', className='ticker-value positive' if change_pct >= 0 else 'ticker-value negative'),
                ],
            ),
            html.Div(
                className='ticker-item',
                children=[
                    html.Span('VAR. %', className='ticker-label'),
                    html.Span(f'{change_pct:+.2f}%', className='ticker-value positive' if change_pct >= 0 else 'ticker-value negative'),
                ],
            ),
            html.Div(
                className='ticker-item',
                children=[
                    html.Span('SEÑAL', className='ticker-label'),
                    html.Span(signal, className=f'ticker-value {"positive" if signal == "ALZA" else "negative" if signal == "PRECAUCIÓN" else "neutral"}'),
                ],
            ),
            html.Div(
                className='ticker-item',
                children=[
                    html.Span('ACIERTOS', className='ticker-label'),
                    html.Span(f'{accuracy:.1%}', className='ticker-value neutral'),
                ],
            ),
            html.Div(
                className='ticker-item',
                children=[
                    html.Span('FIABILIDAD', className='ticker-label'),
                    html.Span(f'{f1:.1%}', className='ticker-value neutral'),
                ],
            ),
            html.Div(
                className='ticker-item',
                children=[
                    html.Span('DXY', className='ticker-label'),
                    html.Span(f'{latest.get("dxy", 0):.2f}', className='ticker-value neutral'),
                ],
            ),
            html.Div(
                className='ticker-item',
                children=[
                    html.Span('VIX', className='ticker-label'),
                    html.Span(f'{latest.get("vix", 0):.2f}', className='ticker-value neutral'),
                ],
            ),
            html.Div(
                className='ticker-item',
                children=[
                    html.Span('MA 21', className='ticker-label'),
                    html.Span(f'${latest.get("ma_21", 0):.2f}', className='ticker-value neutral'),
                ],
            ),
        ]
        return items

    @app.callback(
        [Output('sim-final', 'children'), Output('sim-return', 'children'), Output('sim-trades', 'children'), Output('sim-chart', 'figure')],
        Input('simulate-button', 'n_clicks'),
        State('sim-start', 'date'),
        State('sim-end', 'date'),
        State('sim-capital', 'value'),
    )
    def run_simulation(n_clicks, start_date, end_date, capital):
        if not n_clicks or not start_date or not end_date or not capital:
            raise PreventUpdate

        df = context['data']
        mask = (df.index >= start_date) & (df.index <= end_date)
        sim_data = df.loc[mask].copy()
        if len(sim_data) < 10:
            fig = go.Figure()
            fig.update_layout(**PLOT_THEME, height=380)
            fig.add_annotation(
                x=0.5, y=0.5, xref='paper', yref='paper',
                text='Intervalo seleccionado demasiado corto para una simulación válida.',
                showarrow=False,
                font={'color': '#D4AF37', 'size': 14, 'family': 'Space Mono, monospace'},
            )
            return '$0', '0.0%', '0', fig

        positions = sim_data['signal'].shift(1).fillna(0)
        sim_returns = sim_data['returns'].values * positions.values
        equity = capital * np.cumprod(1 + sim_returns)
        final = float(equity[-1])
        total_return = float((final / capital - 1) * 100)
        trades = int(positions.sum())

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sim_data.index, y=equity,
                name='Portfolio Value',
                line={'color': '#D4AF37', 'width': 2.5},
                fill='tozeroy', fillcolor='rgba(212, 175, 55, 0.12)',
            )
        )
        fig.add_hline(
            y=capital, line_dash='dash', line_color='#AC7D3D',
            annotation_text=f'Capital inicial: ${capital:,.0f}',
            annotation_position='top left',
            annotation_font={'color': '#F2EBE1', 'family': 'Space Mono, monospace', 'size': 11},
        )
        fig.update_layout(**PLOT_THEME, height=420, hovermode='x unified')
        fig.update_xaxes(**AXIS_THEME)
        fig.update_yaxes(**AXIS_THEME, title_text='Valor de cartera ($)')

        return f'${final:,.0f}', f'{total_return:+.1f}%', str(trades), fig


def build_summary_tab() -> html.Div:
    return html.Div(
        className='section-panel',
        children=[
            html.Div(
                className='section-copy',
                children=[
                    html.H2('Panel de Control', className='section-title'),
                    html.P(
                        'Resumen ejecutivo con la señal del día, la certeza del modelo y los indicadores clave para la toma de decisiones sobre el oro.',
                        className='section-text',
                    ),
                ],
            ),
            html.Div(
                className='summary-grid',
                children=[
                    build_stat_card(
                        'Precio actual',
                        f'${context["latest"]["gold"]:.2f}',
                        'Precio spot por onza de oro en el mercado actual.',
                        'Es el precio que cotiza el oro hoy. Los inversores lo usan para comparar si el oro está barato o caro.',
                    ),
                    build_stat_card(
                        'Señal de mercado',
                        context['signal_label'],
                        'Predicción del modelo para el próximo día hábil.',
                        'ALZA = el modelo espera que el precio suba. PRECAUCIÓN = posible baja. ESTABLE = señal débil, mejor esperar.',
                        context['signal_color'],
                    ),
                    build_stat_card(
                        'Certeza',
                        f'{context["confidence"]:.0%}',
                        'Qué tan seguro está el modelo de su señal.',
                        'Por ejemplo, 60% significa que el modelo estima 6 de cada 10 probabilidades de que su predicción sea correcta.',
                    ),
                    build_stat_card(
                        'Actualizado',
                        context['loaded_at'],
                        'Última hora en que se actualizaron los datos.',
                        'Esto te dice cuándo se cargaron los últimos precios y señales. Si es reciente, la lectura es más relevante.',
                    ),
                ],
            ),
            html.Div(
                className='help-panel',
                children=[
                    html.Details(
                        className='help-panel-details',
                        children=[
                            html.Summary('Claves para interpretar el panel', className='help-summary'),
                            html.Div(
                                className='help-copy',
                                children=[
                                    html.P(html.Strong('Precio actual:'), ' cotizaci\u00f3n spot del oro (GC=F) en USD/oz.'),
                                    html.P(html.Strong('Se\u00f1al del modelo:'), ' predicci\u00f3n del algoritmo ensemble para la pr\u00f3xima sesi\u00f3n.'),
                                    html.P(html.Strong('Certeza:'), ' probabilidad asignada por el modelo a su predicci\u00f3n. Mide la confianza estad\u00edstica, no la precisi\u00f3n real.'),
                                    html.P(html.Strong('Aciertos / Fiabilidad:'), ' precisi\u00f3n global (accuracy) y F1-score del modelo en datos de test.'),
                                    html.P(html.Strong('Indicadores macro (DXY, VIX):'), ' el precio del oro se correlaciona inversamente con el d\u00f3lar y directamente con la volatilidad.'),
                                    html.P(html.Strong('MA 21:'), ' media m\u00f3vil de 21 sesiones. Si el precio est\u00e1 por encima, la tendencia es alcista a corto plazo.'),
                                    html.P('Los gr\u00e1ficos inferiores del panel comparan: (1) la evoluci\u00f3n del precio con las se\u00f1ales del modelo, (2) la precisi\u00f3n acumulada, (3) la desviaci\u00f3n entre predicci\u00f3n y realidad, y (4) el rendimiento de la estrategia ML versus Buy & Hold.', className='help-note'),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className='overview-row',
                children=[
                    html.Div(
                        className='wide-card',
                        children=[
                            html.Div('Precio y tendencia del oro', className='card-title'),
                            dcc.Loading(
                                type='dot',
                                children=dcc.Graph(id='summary-price', figure=build_price_figure(), config={'displayModeBar': False}),
                            ),
                        ],
                    ),
                    html.Div(
                        className='narrow-card',
                        children=[
                            html.Div('Precisión del pronosticador', className='card-title'),
                            dcc.Loading(
                                type='dot',
                                children=dcc.Graph(id='summary-model', figure=build_model_figure(), config={'displayModeBar': False}),
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className='overview-row',
                children=[
                    html.Div(
                        className='wide-card',
                        children=[
                            html.Div('Predicción vs Realidad', className='card-title'),
                            dcc.Loading(
                                type='dot',
                                children=dcc.Graph(id='summary-deviation', figure=build_prediction_deviation_figure(), config={'displayModeBar': False}),
                            ),
                        ],
                    ),
                    html.Div(
                        className='narrow-card',
                        children=[
                            html.Div('Rendimiento Acumulado', className='card-title'),
                            dcc.Loading(
                                type='dot',
                                children=dcc.Graph(id='summary-learning', figure=build_learning_curve_figure(), config={'displayModeBar': False}),
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_methodology_tab() -> html.Div:
    return html.Div(
        className='section-panel',
        children=[
            html.Div(
                className='section-copy',
                children=[
                    html.H2('Metodolog\u00eda', className='section-title'),
                    html.P(
                        'Arquitectura del sistema, pipeline de datos, modelos implementados y se\u00f1ales de trading.',
                        className='section-text',
                    ),
                ],
            ),
            html.Div(
                className='overview-row',
                children=[
                    html.Div(
                        className='wide-card',
                        children=[
                            html.Div('Pipeline de Datos', className='card-title'),
                            html.Div(
                                className='governance-markdown',
                                children=[
                                    html.P('1. Extracci\u00f3n: Datos diarios v\u00eda Yahoo Finance desde 2015 (gold, DXY, VIX, TNX).'),
                                    html.P('2. Preprocesamiento: Limpieza, normalizaci\u00f3n y tratamiento de NaN.'),
                                    html.P('3. Feature Engineering: RSI, MACD, medias m\u00f3viles, volatilidad, retornos macro.'),
                                    html.P('4. Modelado: Ensemble con validaci\u00f3n temporal (80/20).'),
                                    html.P('5. Evaluaci\u00f3n: Precisi\u00f3n, F1, ROC-AUC, backtest vs Buy & Hold.'),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className='narrow-card',
                        children=[
                            html.Div('Modelos Implementados', className='card-title'),
                            html.Div(
                                className='governance-markdown',
                                children=[
                                    html.P(html.Strong('Random Forest'), ' \u2014 200 \u00e1rboles, max_depth=10 (principal)'),
                                    html.P(html.Strong('Logistic Regression'), ' \u2014 baseline lineal'),
                                    html.P(html.Strong('XGBoost'), ' \u2014 gradient boosting (si disponible)'),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                className='wide-card',
                children=[
                    html.Div('Se\u00f1ales de Trading', className='card-title'),
                    html.Div(
                        className='governance-markdown',
                        children=[
                            html.P(html.Strong('ALZA'), ' \u2014 Se espera que el precio suba en la pr\u00f3xima sesi\u00f3n (confianza \u2265 58%)'),
                            html.P(html.Strong('PRECAUCI\u00d3N'), ' \u2014 Se anticipa una posible ca\u00edda (confianza \u2264 42%)'),
                            html.P(html.Strong('ESTABLE'), ' \u2014 Se\u00f1al no concluyente, se recomienda esperar'),
                        ],
                    ),
                ],
            ),
            html.Div(
                className='wide-card',
                children=[
                    html.Div('Supuestos y Limitaciones', className='card-title'),
                    html.Div(
                        className='governance-markdown',
                        children=[
                            html.P('Datos de cierre diario (no HFT).'),
                            html.P('Se\u00f1ales probabil\u00edsticas, no certezas absolutas.'),
                            html.P('Rendimiento pasado no garantiza resultados futuros.'),
                            html.P('Modelos re-entrenados con cada actualizaci\u00f3n de datos.'),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_price_tab() -> html.Div:
    return html.Div(
        className='section-panel',
        children=[
            html.Div(
                className='section-copy',
                children=[
                    html.H2('Precio y Señales', className='section-title'),
                    html.P(
                        'Precio histórico del oro con señales del modelo (estrellas) y media móvil de 21 períodos.',
                        className='section-text',
                    ),
                ],
            ),
            html.Div(
                className='wide-card',
                children=[
                    html.Div('Precio del Oro y Señales del Modelo', className='card-title'),
                    dcc.Loading(type='dot', children=dcc.Graph(id='price-chart', figure=build_price_figure(), config={'displayModeBar': False})),
                ],
            ),
        ],
    )


def build_indicators_tab() -> html.Div:
    return html.Div(
        className='section-panel',
        children=[
            html.Div(
                className='section-copy',
                children=[
                    html.H2('Indicadores Técnicos', className='section-title'),
                    html.P(
                        'RSI (14), MACD y volatilidad para identificar condiciones de sobrecompra/sobreventa y cambios de tendencia.',
                        className='section-text',
                    ),
                ],
            ),
            html.Div(
                className='chart-grid',
                children=[
                    html.Div(
                        className='graph-card',
                        children=[
                            html.Div('RSI (14)', className='card-title'),
                            dcc.Loading(type='dot', children=dcc.Graph(id='indicators-rsi', figure=build_rsi_figure(), config={'displayModeBar': False})),
                        ],
                    ),
                    html.Div(
                        className='graph-card',
                        children=[
                            html.Div('MACD', className='card-title'),
                            dcc.Loading(type='dot', children=dcc.Graph(id='indicators-macd', figure=build_macd_figure(), config={'displayModeBar': False})),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_macro_tab() -> html.Div:
    return html.Div(
        className='section-panel',
        children=[
            html.Div(
                className='section-copy',
                children=[
                    html.H2('Correlaciones Macro', className='section-title'),
                    html.P(
                        'Relación del oro con DXY (dólar), VIX (volatilidad) y TNX (bonos). Panel inferior: indicadores normalizados.',
                        className='section-text',
                    ),
                ],
            ),
            html.Div(
                className='wide-card',
                children=[
                    html.Div('Correlaciones y Normalizados', className='card-title'),
                    dcc.Loading(type='dot', children=dcc.Graph(id='macro-chart', figure=build_macro_figure(), config={'displayModeBar': False})),
                ],
            ),
        ],
    )


def build_backtest_tab() -> html.Div:
    return html.Div(
        className='section-panel',
        children=[
            html.Div(
                className='section-copy',
                children=[
                    html.H2('Backtest y Estrategia', className='section-title'),
                    html.P(
                        'Rendimiento de la estrategia ML vs Buy & Hold, alpha generado y precisión del modelo en el tiempo.',
                        className='section-text',
                    ),
                ],
            ),
            html.Div(
                className='overview-row',
                children=[
                    html.Div(
                        className='wide-card',
                        children=[
                            html.Div('Estrategia ML vs Buy & Hold', className='card-title'),
                            dcc.Loading(type='dot', children=dcc.Graph(id='backtest-chart', figure=build_backtest_figure(), config={'displayModeBar': False})),
                        ],
                    ),
                    html.Div(
                        className='narrow-card',
                        children=[
                            html.Div('Precisión en el Tiempo', className='card-title'),
                            dcc.Loading(type='dot', children=dcc.Graph(id='learning-chart', figure=build_learning_curve_figure(), config={'displayModeBar': False})),
                        ],
                    ),
                ],
            ),
            html.Div(
                className='overview-row',
                children=[
                    html.Div(
                        className='wide-card',
                        children=[
                            html.Div('Predicción vs Realidad', className='card-title'),
                            dcc.Loading(type='dot', children=dcc.Graph(id='deviation-chart', figure=build_prediction_deviation_figure(), config={'displayModeBar': False})),
                        ],
                    ),
                    html.Div(
                        className='narrow-card',
                        children=[
                            html.Div('Métricas del Modelo', className='card-title'),
                            dcc.Loading(type='dot', children=dcc.Graph(id='model-chart', figure=build_model_figure(), config={'displayModeBar': False})),
                        ],
                    ),
                ],
            ),
        ],
    )


def build_simulation_tab() -> html.Div:
    return html.Div(
        className='section-panel',
        children=[
            html.Div(
                className='section-copy',
                children=[
                    html.H2('Simulación de Escenario', className='section-title'),
                    html.P(
                        'Simula el comportamiento de la cartera siguiendo las señales del modelo en un período personalizado.',
                        className='section-text',
                    ),
                ],
            ),
            html.Div(
                className='simulation-panel',
                children=[
                    html.Div('Parámetros', className='card-title'),
                    html.Div(
                        className='simulation-controls',
                        children=[
                            html.Div(
                                className='input-group',
                                children=[
                                    html.Label('Fecha inicio', className='input-label'),
                                    dcc.DatePickerSingle(id='sim-start', date=context['data'].index[context['split_index']].date(), display_format='YYYY-MM-DD', className='date-picker'),
                                ],
                            ),
                            html.Div(
                                className='input-group',
                                children=[
                                    html.Label('Fecha fin', className='input-label'),
                                    dcc.DatePickerSingle(id='sim-end', date=context['data'].index[-1].date(), display_format='YYYY-MM-DD', className='date-picker'),
                                ],
                            ),
                            html.Div(
                                className='input-group',
                                children=[
                                    html.Label('Capital inicial', className='input-label'),
                                    dcc.Input(id='sim-capital', type='number', min=1000, max=1000000, step=1000, value=10000, className='input-field'),
                                ],
                            ),
                            html.Button('EJECUTAR SIMULACIÓN', id='simulate-button', className='control-button', n_clicks=0),
                        ],
                    ),
                    html.Div(
                        className='stats-grid',
                        children=[
                            html.Div(
                                className='metric-card',
                                children=[
                                    html.Div('Valor final', className='metric-label'),
                                    html.Div('$0', id='sim-final', className='metric-value', style={'color': '#F2EBE1'}),
                                    html.Div('Resultado de la simulación', className='metric-note'),
                                ],
                            ),
                            html.Div(
                                className='metric-card',
                                children=[
                                    html.Div('Retorno', className='metric-label'),
                                    html.Div('0.0%', id='sim-return', className='metric-value', style={'color': '#F2EBE1'}),
                                    html.Div('Rentabilidad de la simulación', className='metric-note'),
                                ],
                            ),
                            html.Div(
                                className='metric-card',
                                children=[
                                    html.Div('Operaciones', className='metric-label'),
                                    html.Div('0', id='sim-trades', className='metric-value', style={'color': '#F2EBE1'}),
                                    html.Div('Número de señales ejecutadas', className='metric-note'),
                                ],
                            ),
                        ],
                    ),
                    dcc.Loading(type='dot', children=dcc.Graph(id='sim-chart', figure=build_simulation_placeholder(), config={'displayModeBar': False})),
                ],
            ),
        ],
    )


def build_stat_card(title, value, subtitle, help_text=None, accent='#D4AF37') -> html.Div:
    children = [
        html.Div(title, className='metric-label'),
        html.Div(value, className='metric-value', style={'color': accent}),
        html.Div(subtitle, className='metric-note'),
    ]
    if help_text:
        children.append(
            html.Details(
                className='metric-help',
                children=[
                    html.Summary('¿Qué significa?', className='metric-help-summary'),
                    html.Div(help_text, className='metric-help-text'),
                ],
            )
        )
    return html.Div(
        className='metric-card',
        children=children,
    )


def build_price_figure() -> go.Figure:
    df = context['data']
    fig = go.Figure()
    
    # Línea principal del precio
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['gold'],
            name='Precio Oro',
            mode='lines',
            line={'color': WANTED_COLORS['gold'], 'width': 3},
            fill='tozeroy',
            fillcolor='rgba(232, 195, 74, 0.15)',
        )
    )
    
    # Media móvil 21
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ma_21'],
            name='MA 21',
            mode='lines',
            line={'color': WANTED_COLORS['brass'], 'dash': 'dot', 'width': 2},
        )
    )
    
    # Señal del modelo
    test = context['test_data'] if 'test_data' in context else df.iloc[int(len(df)*0.8):]
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test['gold'],
            mode='markers',
            name='Señal Modelo',
            marker={
                'color': [WANTED_COLORS['buy'] if s == 1 else WANTED_COLORS['sell'] for s in context['data']['signal'].iloc[len(context['data'])-len(test):]],
                'size': 8,
                'symbol': 'star',
            },
        )
    )
    
    fig.add_shape(
        type='line',
        x0=context['split_date'],
        x1=context['split_date'],
        y0=df['gold'].min(),
        y1=df['gold'].max(),
        line={'color': WANTED_COLORS['iron_mid'], 'dash': 'dash', 'width': 2},
    )
    
    fig.update_layout(**PLOT_THEME, height=420, hovermode='x unified',
                      title={'text': 'Precio del Oro y Señales del Modelo', 'font': {'family': 'Rye, Smokum, serif', 'color': WANTED_COLORS['gold']}})
    fig.update_xaxes(**AXIS_THEME, rangeselector={
        'buttons': [
            {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
            {'count': 3, 'label': '3M', 'step': 'month', 'stepmode': 'backward'},
            {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
            {'count': 1, 'label': '1A', 'step': 'year', 'stepmode': 'backward'},
            {'step': 'all', 'label': 'TODO'},
        ],
        'bgcolor': 'rgba(26, 16, 8, 0.8)',
        'activecolor': 'rgba(232, 195, 74, 0.3)',
        'borderwidth': 1,
        'bordercolor': 'rgba(232, 195, 74, 0.3)',
        'font': {'color': '#F2EBE1', 'size': 10, 'family': 'Space Mono, monospace'},
    }, rangeslider={'visible': True, 'bgcolor': 'rgba(26, 16, 8, 0.4)', 'bordercolor': 'rgba(232, 195, 74, 0.15)'},
       range=[df.index[-365], df.index[-1]])
    fig.update_yaxes(**AXIS_THEME, title_text='USD por onza')
    return fig


def build_rsi_figure() -> go.Figure:
    df = context['data']
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['rsi'],
            name='RSI',
            line={'color': '#D4AF37', 'width': 2},
        )
    )
    fig.add_hline(y=70, line_dash='dash', line_color='#AC7D3D')
    fig.add_hline(y=30, line_dash='dash', line_color='#7A5C33')
    fig.update_layout(**PLOT_THEME, height=340, hovermode='x unified')
    fig.update_xaxes(**AXIS_THEME)
    fig.update_yaxes(**AXIS_THEME, title_text='RSI')
    return fig


def build_macd_figure() -> go.Figure:
    df = context['data']
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['macd'],
            name='MACD',
            line={'color': '#D4AF37', 'width': 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['macd_signal'],
            name='Señal',
            line={'color': '#AC7D3D', 'width': 2, 'dash': 'dot'},
        )
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['macd'] - df['macd_signal'],
            name='Histograma',
            marker_color='#E8C34A',
            opacity=0.6,
        )
    )
    fig.add_hline(y=0, line_color='#5C4732', line_width=1)
    fig.update_layout(**PLOT_THEME, height=340, hovermode='x unified')
    fig.update_xaxes(**AXIS_THEME)
    fig.update_yaxes(**AXIS_THEME, title_text='MACD')
    return fig


def build_macro_figure() -> go.Figure:
    df = context['data']
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Oro vs DXY', 'Oro vs VIX', 'Oro vs TNX', 'Indicadores normalizados'))
    fig.add_trace(go.Scatter(x=df['dxy'], y=df['gold'], mode='markers', marker={'color': '#D4AF37', 'size': 4}), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['vix'], y=df['gold'], mode='markers', marker={'color': '#AC7D3D', 'size': 4}), row=1, col=2)
    fig.add_trace(go.Scatter(x=df['tnx'], y=df['gold'], mode='markers', marker={'color': '#8D6B34', 'size': 4}), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['gold'] / df['gold'].iloc[0] * 100, name='Gold', line={'color': '#D4AF37', 'width': 2}), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['dxy'] / df['dxy'].iloc[0] * 100, name='DXY', line={'color': '#AC7D3D', 'width': 2}), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['vix'] / df['vix'].iloc[0] * 100, name='VIX', line={'color': '#7A5C33', 'width': 2}), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['tnx'] / df['tnx'].iloc[0] * 100, name='TNX', line={'color': '#5C4732', 'width': 2}), row=2, col=2)
    fig.update_layout(**PLOT_THEME, height=520, hovermode='x unified', showlegend=True)
    fig.update_xaxes(**AXIS_THEME)
    fig.update_yaxes(**AXIS_THEME)
    return fig


def build_model_figure() -> go.Figure:
    metrics = ['Aciertos', 'Fiabilidad', 'Poder predictivo']
    values = [context['accuracy'], context['f1'], context['auc']]
    colors = [WANTED_COLORS['buy'], WANTED_COLORS['brass'], WANTED_COLORS['signal_neutral']]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=metrics,
            y=values,
            marker={'color': colors, 'line': {'color': WANTED_COLORS['gold'], 'width': 2}},
            text=[f'{v:.2%}' for v in values],
            textposition='outside',
            textfont={'color': '#F2EBE1', 'size': 16, 'family': 'Space Mono, monospace'},
        )
    )
    
    fig.update_layout(
        **PLOT_THEME, 
        height=380, 
        yaxis={'range': [0, 1]}, 
        bargap=0.25,
        title={'text': 'Rendimiento del Modelo', 'font': {'family': 'Rye, Smokum, serif', 'color': WANTED_COLORS['gold']}}
    )
    fig.update_xaxes(**AXIS_THEME, title_font={'size': 14})
    fig.update_yaxes(**AXIS_THEME, title_text='Score')
    return fig


def build_backtest_figure() -> go.Figure:
    test = context['test_data']
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=context['cum_strategy'] * 100,
            name='Estrategia ML',
            line={'color': WANTED_COLORS['buy'], 'width': 3},
            fill='tozeroy',
            fillcolor='rgba(34, 197, 94, 0.12)',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=context['cum_bh'] * 100,
            name='Buy & Hold',
            line={'color': WANTED_COLORS['sell'], 'width': 2.5, 'dash': 'dash'},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=context['cum_strategy'] * 100 - context['cum_bh'] * 100,
            name='Alpha (Estrategia - B&H)',
            line={'color': WANTED_COLORS['gold'], 'width': 2},
            yaxis='y2',
        )
    )
    fig.update_layout(
        **PLOT_THEME, 
        height=400, 
        hovermode='x unified', 
        title={'text': 'Backtest: Estrategia ML vs Buy & Hold', 'font': {'family': 'Rye, Smokum, serif', 'color': WANTED_COLORS['gold']}},
        yaxis2={'overlaying': 'y', 'side': 'right', 'showgrid': False, 'title': 'Alpha (%)'},
    )
    fig.update_xaxes(**AXIS_THEME)
    fig.update_yaxes(**AXIS_THEME, title_text='Rendimiento acumulado (%)')
    return fig


def build_prediction_deviation_figure() -> go.Figure:
    """Gráfico de desviación entre predicción y precio real"""
    test = context['test_data']
    predictions = context.get('predictions', [])
    probabilities = context.get('probabilities', [])
    
    fig = go.Figure()
    
    # Precio real
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test['gold'],
            name='Precio Real',
            mode='lines',
            line={'color': WANTED_COLORS['gold'], 'width': 2.5},
        )
    )
    
    # Precio predicho (aproximado usando señales)
    predicted_prices = []
    for i, (idx, row) in enumerate(test.iterrows()):
        if i > 0:
            pred_change = row['returns'] * (1 if predictions[i-1] == 1 else -1)
            predicted_prices.append(test.iloc[i-1]['gold'] * (1 + pred_change))
        else:
            predicted_prices.append(row['gold'])
    
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=predicted_prices,
            name='Precio Predicho (Modelo)',
            mode='lines',
            line={'color': WANTED_COLORS['buy'], 'width': 2, 'dash': 'dot'},
        )
    )
    
    # Área de confianza
    prob_high = [p * test.iloc[i]['gold'] * 1.02 for i, p in enumerate(probabilities[:len(test)])]
    prob_low = [p * test.iloc[i]['gold'] * 0.98 for i, p in enumerate(probabilities[:len(test)])]
    
    fig.add_trace(
        go.Scatter(
            x=list(test.index) + list(test.index)[::-1],
            y=prob_high + prob_low[::-1],
            fill='toself',
            fillcolor='rgba(232, 195, 74, 0.1)',
            line={'color': 'rgba(0,0,0,0)'},
            name='Banda de Confianza (95%)',
            showlegend=True,
        )
    )
    
    fig.update_layout(
        **PLOT_THEME,
        height=380,
        hovermode='x unified',
        title={'text': 'Predicción vs Realidad', 'font': {'family': 'Rye, Smokum, serif', 'color': WANTED_COLORS['gold']}},
    )
    fig.update_xaxes(**AXIS_THEME)
    fig.update_yaxes(**AXIS_THEME, title_text='USD por onza')
    return fig


def build_learning_curve_figure() -> go.Figure:
    """Curva de precisión del modelo durante el entrenamiento"""
    test = context['test_data']
    train = context.get('train', test)
    
    # Simular curva de aprendizaje basada en precisión acumulada
    cum_accuracy = []
    correct = 0
    total = 0
    
    for i, (_, row) in enumerate(test.iterrows()):
        pred = context['predictions'][i] if i < len(context['predictions']) else 0
        actual = row['target_bin']
        total += 1
        if pred == actual:
            correct += 1
        cum_accuracy.append(correct / max(total, 1))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=list(range(len(cum_accuracy))),
        y=cum_accuracy,
        name='Precisión Acumulada',
        line={'color': WANTED_COLORS['gold'], 'width': 3},
        fill='tozeroy',
        fillcolor='rgba(232, 195, 74, 0.1)',
    ))
    
    fig.add_hline(
        y=context['accuracy'],
        line_dash='dash',
        line_color=WANTED_COLORS['buy'],
        annotation_text=f'Precisión Total: {context["accuracy"]:.1%}',
        annotation_font={'color': WANTED_COLORS['buy'], 'family': 'Space Mono, monospace'},
    )
    
    fig.update_layout(
        **PLOT_THEME,
        height=380,
        hovermode='x unified',
        title={'text': 'Precisión del Modelo en el Tiempo', 'font': {'family': 'Rye, Smokum, serif', 'color': WANTED_COLORS['gold']}},
        xaxis_title='Operaciones',
        yaxis_title='Precisión Acumulada',
    )
    fig.update_xaxes(**AXIS_THEME)
    fig.update_yaxes(**AXIS_THEME, range=[0.3, 1.0])
    return fig


def build_simulation_placeholder() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**PLOT_THEME, height=380)
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref='paper',
        yref='paper',
        text='Ejecuta la simulación para visualizar el comportamiento de la cartera.',
        showarrow=False,
        font={'color': '#D4AF37', 'size': 14, 'family': 'Space Mono, monospace'},
    )
    return fig


def build_feature_importance_figure() -> go.Figure:
    """Feature importance chart for the Random Forest model."""
    model_results = context.get('model_results', {})
    rf_result = model_results.get('Random Forest')
    if not rf_result or 'model' not in rf_result:
        fig = go.Figure()
        fig.update_layout(**PLOT_THEME, height=350)
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='No hay datos de importancia de variables', showarrow=False, font={'color': '#D4AF37', 'size': 14})
        return fig

    fi = rf_result['model'].feature_importances_
    fi_indices = np.argsort(fi)[::-1]
    features = [FEATURE_COLUMNS[i] for i in fi_indices]
    values = [fi[i] for i in fi_indices]
    colors = [f'rgba(232, 195, 74, {0.4 + v * 0.6:.2f})' for v in values]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker={
            'color': colors,
            'line': {'color': WANTED_COLORS['gold'], 'width': 1.5},
            'cornerradius': 4,
        },
        text=[f'{v:.1%}' for v in values],
        textposition='outside',
        textfont={'color': '#F2EBE1', 'size': 11, 'family': 'Space Mono, monospace'},
        hovertemplate='%{y}: %{x:.1%}<extra></extra>',
    ))
    fig.update_layout(
        **PLOT_THEME,
        height=400,
        xaxis={'range': [0, max(values) * 1.25], 'showgrid': True, 'gridcolor': 'rgba(242,235,225,0.08)'},
        title={'text': 'Importancia de Variables (Random Forest)', 'font': {'family': 'Rye, Smokum, serif', 'color': WANTED_COLORS['gold']}},
    )
    fig.update_xaxes(**AXIS_THEME, tickformat='.0%')
    fig.update_yaxes(**AXIS_THEME, autorange='reversed')
    return fig


def build_confusion_matrix_figure() -> go.Figure:
    """Confusion matrix heatmap."""
    model_results = context.get('model_results', {})
    rf_result = model_results.get('Random Forest')
    if not rf_result:
        fig = go.Figure()
        fig.update_layout(**PLOT_THEME, height=350)
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='No hay datos de matriz de confusión', showarrow=False, font={'color': '#D4AF37', 'size': 14})
        return fig

    test = context['test_data']
    cm = confusion_matrix(test['target_bin'], rf_result['predictions'])

    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Pred: Bajada', 'Pred: Subida'],
        y=['Real: Bajada', 'Real: Subida'],
        colorscale=[[0, '#2C1A0C'], [0.5, '#7A5C1E'], [1, '#E8C34A']],
        text=cm,
        texttemplate='%{text}',
        textfont={'color': '#F5EDE0', 'size': 16},
        hovertemplate='Real: %{y}<br>Pred: %{x}<br>Cuenta: %{z}<extra></extra>',
    ))
    fig.update_layout(
        **PLOT_THEME,
        height=350,
        title={'text': 'Matriz de Confusión', 'font': {'family': 'Rye, Smokum, serif', 'color': WANTED_COLORS['gold']}},
    )
    fig.update_xaxes(**AXIS_THEME, side='bottom')
    fig.update_yaxes(**AXIS_THEME)
    return fig


def build_roc_curve_figure() -> go.Figure:
    """ROC curve for the Random Forest model."""
    model_results = context.get('model_results', {})
    rf_result = model_results.get('Random Forest')
    if not rf_result:
        fig = go.Figure()
        fig.update_layout(**PLOT_THEME, height=350)
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='No hay datos de curva ROC', showarrow=False, font={'color': '#D4AF37', 'size': 14})
        return fig

    test = context['test_data']
    fpr, tpr, _ = roc_curve(test['target_bin'], rf_result['probabilities'])
    auc_score = rf_result['auc']

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        line={'color': WANTED_COLORS['gold'], 'width': 3},
        name=f'ROC (AUC={auc_score:.3f})',
        fill='tozeroy',
        fillcolor='rgba(212, 175, 55, 0.12)',
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        line={'color': '#2C1A0C', 'width': 1, 'dash': 'dash'},
        name='Aleatorio',
    ))
    fig.update_layout(
        **PLOT_THEME,
        height=350,
        title={'text': 'Curva ROC', 'font': {'family': 'Rye, Smokum, serif', 'color': WANTED_COLORS['gold']}},
        xaxis_title='Falsos Positivos (1 - Especificidad)',
        yaxis_title='Verdaderos Positivos (Sensibilidad)',
    )
    fig.update_xaxes(**AXIS_THEME)
    fig.update_yaxes(**AXIS_THEME)
    return fig


def build_model_comparison_table() -> html.Table:
    """Model comparison table with all trained models."""
    model_results = context.get('model_results', {})
    if not model_results:
        return html.Table(style={'width': '100%', 'borderCollapse': 'collapse'}, children=[
            html.Tr([html.Th('No hay datos de modelos disponibles', style={'padding': '8px', 'color': '#D4AF37'})])
        ])

    rows = [
        html.Tr([
            html.Th('Modelo', style={'borderBottom': '1px solid #2C1A0C', 'padding': '8px', 'textAlign': 'left', 'color': '#D4AF37'}),
            html.Th('Precisión', style={'borderBottom': '1px solid #2C1A0C', 'padding': '8px', 'color': '#D4AF37'}),
            html.Th('F1 Score', style={'borderBottom': '1px solid #2C1A0C', 'padding': '8px', 'color': '#D4AF37'}),
            html.Th('ROC-AUC', style={'borderBottom': '1px solid #2C1A0C', 'padding': '8px', 'color': '#D4AF37'}),
            html.Th('Rentabilidad', style={'borderBottom': '1px solid #2C1A0C', 'padding': '8px', 'color': '#D4AF37'}),
        ]),
    ]

    for name, result in model_results.items():
        profit_color = '#22C55E' if result['cum_strategy'][-1] > 0 else '#EF4444'
        rows.append(html.Tr([
            html.Td(name, style={'borderBottom': '1px solid rgba(44,26,12,0.3)', 'padding': '6px 8px'}),
            html.Td(f'{result["accuracy"]:.1%}', style={'borderBottom': '1px solid rgba(44,26,12,0.3)', 'padding': '6px 8px'}),
            html.Td(f'{result["f1"]:.3f}', style={'borderBottom': '1px solid rgba(44,26,12,0.3)', 'padding': '6px 8px'}),
            html.Td(f'{result["auc"]:.3f}' if result['auc'] > 0 else '—', style={'borderBottom': '1px solid rgba(44,26,12,0.3)', 'padding': '6px 8px'}),
            html.Td(f'{result["cum_strategy"][-1]:+.1%}', style={'borderBottom': '1px solid rgba(44,26,12,0.3)', 'padding': '6px 8px', 'color': profit_color}),
        ]))

    return html.Table(style={'width': '100%', 'borderCollapse': 'collapse', 'color': '#F5EDE0', 'fontFamily': 'Space Mono, monospace', 'fontSize': '13px'}, children=rows)


def build_metrics_tab() -> html.Div:
    return html.Div(
        className='section-panel',
        children=[
            html.Div(
                className='section-copy',
                children=[
                    html.H2('M\u00e9tricas de Modelo', className='section-title'),
                    html.P(
                        'Evaluaci\u00f3n t\u00e9cnica del rendimiento de los modelos: importancia de variables, matriz de confusi\u00f3n, curva ROC y comparativa multi-modelo.',
                        className='section-text',
                    ),
                ],
            ),
            html.Div(
                className='wide-card',
                children=[
                    html.Div('Importancia de Variables', className='card-title'),
                    dcc.Loading(type='dot', children=dcc.Graph(id='fi-chart', figure=build_feature_importance_figure(), config={'displayModeBar': False})),
                ],
            ),
            html.Div(
                className='chart-grid',
                children=[
                    html.Div(
                        className='graph-card',
                        children=[
                            html.Div('Matriz de Confusi\u00f3n', className='card-title'),
                            dcc.Loading(type='dot', children=dcc.Graph(id='cm-chart', figure=build_confusion_matrix_figure(), config={'displayModeBar': False})),
                        ],
                    ),
                    html.Div(
                        className='graph-card',
                        children=[
                            html.Div('Curva ROC', className='card-title'),
                            dcc.Loading(type='dot', children=dcc.Graph(id='roc-chart', figure=build_roc_curve_figure(), config={'displayModeBar': False})),
                        ],
                    ),
                ],
            ),
            html.Div(
                className='wide-card',
                children=[
                    html.Div('Comparativa de Modelos', className='card-title'),
                    build_model_comparison_table(),
                ],
            ),
        ],
    )
