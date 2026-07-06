from datetime import datetime, timedelta

import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, html, dcc
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve

import pandas as pd

from src.dashboard import data

context = data.context
FEATURE_COLUMNS = data.FEATURE_COLUMNS

FEATURE_LABELS = {
    'gold_volume': 'Volumen Operaciones',
    'gold_return': 'Rendimiento Diario Oro',
    'dxy_return': 'Retorno DXY (D\u00f3lar)',
    'vix_return': 'Retorno VIX (Volatilidad)',
    'tnx_return': 'Retorno TNX (Bonos)',
    'gold_daily_range': 'Rango Diario Oro',
    'dxy_daily_range': 'Rango Diario DXY',
    'vix_daily_range': 'Rango Diario VIX',
    'tnx_daily_range': 'Rango Diario TNX',
    'gold_open_close_return': 'Retorno Intra-d\u00eda Oro',
    'dxy_open_close_return': 'Retorno Intra-d\u00eda DXY',
    'vix_open_close_return': 'Retorno Intra-d\u00eda VIX',
    'tnx_open_close_return': 'Retorno Intra-d\u00eda TNX',
    'gold_ma_5': 'Media M\u00f3vil 5 d\u00edas',
    'gold_ma_20': 'Media M\u00f3vil 20 d\u00edas',
    'gold_close_vs_ma_5': 'Precio vs MA5',
    'gold_close_vs_ma_20': 'Precio vs MA20',
    'gold_rsi_14': 'RSI 14 d\u00edas',
    'gold_macd': 'MACD',
    'gold_macd_signal': 'Se\u00f1al MACD',
    'gold_volatility_14': 'Volatilidad 14 d\u00edas',
    'gold_return_lag_1': 'Retorno d\u00eda anterior',
    'gold_return_lag_2': 'Retorno hace 2 d\u00edas',
}

FEATURE_CATEGORIES = {
    'gold_volume': 'T\u00e9cnico',
    'gold_return': 'Precio',
    'dxy_return': 'Macro',
    'vix_return': 'Macro',
    'tnx_return': 'Macro',
    'gold_daily_range': 'T\u00e9cnico',
    'dxy_daily_range': 'T\u00e9cnico',
    'vix_daily_range': 'T\u00e9cnico',
    'tnx_daily_range': 'T\u00e9cnico',
    'gold_open_close_return': 'T\u00e9cnico',
    'dxy_open_close_return': 'T\u00e9cnico',
    'vix_open_close_return': 'T\u00e9cnico',
    'tnx_open_close_return': 'T\u00e9cnico',
    'gold_ma_5': 'T\u00e9cnico',
    'gold_ma_20': 'T\u00e9cnico',
    'gold_close_vs_ma_5': 'T\u00e9cnico',
    'gold_close_vs_ma_20': 'T\u00e9cnico',
    'gold_rsi_14': 'T\u00e9cnico',
    'gold_macd': 'T\u00e9cnico',
    'gold_macd_signal': 'T\u00e9cnico',
    'gold_volatility_14': 'T\u00e9cnico',
    'gold_return_lag_1': 'Precio',
    'gold_return_lag_2': 'Precio',
}

CATEGORY_COLORS = {
    'Precio': '#00E676',
    'T\u00e9cnico': '#42A5F5',
    'Macro': '#AB47BC',
}

FEATURE_DESCRIPTIONS = {
    'gold_volume': 'N\u00famero de contratos operados en la sesi\u00f3n',
    'gold_return': 'Cambio porcentual del precio del oro respecto al d\u00eda anterior',
    'dxy_return': 'Retorno diario del \u00edndice del d\u00f3lar (correlaci\u00f3n inversa con el oro)',
    'vix_return': 'Retorno diario del \u00edndice de volatilidad (miedo del mercado)',
    'tnx_return': 'Retorno diario de los bonos del tesoro a 10 a\u00f1os',
    'gold_daily_range': '(M\u00e1ximo - M\u00ednimo) / Apertura del oro en el d\u00eda',
    'dxy_daily_range': '(M\u00e1ximo - M\u00ednimo) / Apertura del DXY en el d\u00eda',
    'vix_daily_range': '(M\u00e1ximo - M\u00ednimo) / Apertura del VIX en el d\u00eda',
    'tnx_daily_range': '(M\u00e1ximo - M\u00ednimo) / Apertura del TNX en el d\u00eda',
    'gold_open_close_return': '(Cierre - Apertura) / Apertura del oro, movimiento intra-d\u00eda',
    'dxy_open_close_return': '(Cierre - Apertura) / Apertura del DXY, movimiento intra-d\u00eda',
    'vix_open_close_return': '(Cierre - Apertura) / Apertura del VIX, movimiento intra-d\u00eda',
    'tnx_open_close_return': '(Cierre - Apertura) / Apertura del TNX, movimiento intra-d\u00eda',
    'gold_ma_5': 'Precio promedio del oro en los \u00faltimos 5 d\u00edas h\u00e1biles',
    'gold_ma_20': 'Precio promedio del oro en los \u00faltimos 20 d\u00edas h\u00e1biles',
    'gold_close_vs_ma_5': 'Distancia porcentual del precio actual respecto a su MA5',
    'gold_close_vs_ma_20': 'Distancia porcentual del precio actual respecto a su MA20',
    'gold_rsi_14': 'Indicador de sobrecompra (>70) o sobreventa (<30)',
    'gold_macd': 'Diferencia entre medias m\u00f3viles exponenciales de 12 y 26 d\u00edas',
    'gold_macd_signal': 'Media m\u00f3vil de 9 d\u00edas de la l\u00ednea MACD',
    'gold_volatility_14': 'Desviaci\u00f3n t\u00edpica de los retornos en 14 d\u00edas',
    'gold_return_lag_1': 'Retorno del oro del d\u00eda anterior',
    'gold_return_lag_2': 'Retorno del oro de hace dos d\u00edas',
}

PLOT_THEME = {
    'plot_bgcolor': 'rgba(0,0,0,0)',
    'paper_bgcolor': 'rgba(0,0,0,0)',
    'font': {'color': '#F2EBE1', 'family': 'Space Mono, monospace'},
    'margin': {'t': 40, 'b': 50, 'l': 70, 'r': 50},
    'legend': {'orientation': 'h', 'y': -0.25, 'font': {'color': '#F2EBE1'}},
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
    'font': {'color': '#F2EBE1', 'size': 9, 'family': 'Space Mono, monospace'},
    'yanchor': 'top',
    'x': 1.02,
    'xanchor': 'right',
}

RANGE_BUTTONS = [
    {'count': 1, 'label': '1D', 'step': 'day', 'stepmode': 'backward'},
    {'count': 5, 'label': '5D', 'step': 'day', 'stepmode': 'backward'},
    {'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
    {'count': 3, 'label': '3M', 'step': 'month', 'stepmode': 'backward'},
    {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
    {'count': 1, 'label': '1A', 'step': 'year', 'stepmode': 'backward'},
    {'step': 'all', 'label': 'HIST'},
]


def _apply_rangeselector(fig, df):
    """Apply date range selector and slider to a figure's x-axis (default: last 90 days)."""
    last_idx = min(len(df) - 1, 90)
    fig.update_xaxes(
        rangeselector={'buttons': RANGE_BUTTONS, **RANGESELECTOR},
        rangeslider={'visible': True, 'bgcolor': 'rgba(26, 16, 8, 0.4)', 'bordercolor': 'rgba(232, 195, 74, 0.15)'},
        range=[df.index[-last_idx], df.index[-1]],
    )


# Colores por tipo de serie
WANTED_COLORS = {
    'gold': '#FFD700',
    'ma': '#29B6F6',
    'ma_short': '#4FC3F7',
    'signal_buy': '#00C853',
    'signal_sell': '#FF5252',
    'signal_neutral': '#FFD700',
    'buy': '#00C853',
    'sell': '#FF5252',
    'hold': '#FFB74D',
    'prediction': '#00E676',
    'brass': '#D4AF37',
    'brass_light': '#F5DC8A',
    'iron': '#1A1613',
    'iron_mid': '#2C2620',
    'iron_light': '#3D352C',
    'macro_dxy': '#42A5F5',
    'macro_vix': '#AB47BC',
    'macro_tnx': '#FFA726',
    'rsi_overbought': '#FF5252',
    'rsi_oversold': '#4CAF50',
}


def register_callbacks(app):
    _tab_cache: dict[str, html.Div] = {}

    @app.callback(Output('tab-content', 'children'), Input('dashboard-tabs', 'value'))
    def render_tab(active_tab):
        if active_tab not in _tab_cache:
            if active_tab == 'tab-summary':
                _tab_cache[active_tab] = build_summary_tab()
            elif active_tab == 'tab-price':
                _tab_cache[active_tab] = build_price_tab()
            elif active_tab == 'tab-indicators':
                _tab_cache[active_tab] = build_indicators_tab()
            elif active_tab == 'tab-macro':
                _tab_cache[active_tab] = build_macro_tab()
            elif active_tab == 'tab-backtest':
                _tab_cache[active_tab] = build_backtest_tab()
            elif active_tab == 'tab-sim':
                _tab_cache[active_tab] = build_simulation_tab()
            elif active_tab == 'tab-metrics':
                _tab_cache[active_tab] = build_metrics_tab()
            elif active_tab == 'tab-regression':
                _tab_cache[active_tab] = build_regression_tab()
            elif active_tab == 'tab-governance':
                _tab_cache[active_tab] = build_methodology_tab()
            else:
                raise PreventUpdate
        return _tab_cache[active_tab]

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
        fig.update_xaxes(**AXIS_THEME, title_text='Fecha')
        fig.update_yaxes(**AXIS_THEME, title_text='Valor de cartera ($)')

        return f'${final:,.0f}', f'{total_return:+.1f}%', str(trades), fig

    @app.callback(Output('fi-chart', 'figure'), Input('fi-category-filter', 'value'))
    def update_feature_importance(category):
        return build_feature_importance_figure(category)

    @app.callback(Output('fi-detail', 'children'), Input('fi-chart', 'clickData'))
    def show_feature_detail(click_data):
        if not click_data:
            return 'Haz clic en una barra para ver m\u00e1s detalle sobre esa variable.'
        point = click_data['points'][0]
        label = point['y']
        value = point['x']
        desc = point['customdata'][0] if point.get('customdata') else ''
        cat = point['customdata'][1] if point.get('customdata') else ''
        return html.Span([
            html.Strong(f'{label}', style={'color': '#D4AF37'}),
            f' \u2014 Peso: {value:.1%}  |  Categor\u00eda: {cat}  |  {desc}',
        ])

    app.clientside_callback(
        '''
        function(volume) {
            const audio = document.getElementById('ambient-audio');
            if (audio) {
                audio.muted = false;
                audio.volume = volume / 100;
                if (audio.paused) {
                    audio.play().catch(function() {});
                }
            }
            return window.dash_clientside.no_update;
        }
        ''',
        Output('ambient-audio', 'id'),
        Input('vol-slider', 'value'),
    )

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
                    html.Div(
                        className='help-copy',
                        style={'marginTop': '12px', 'padding': '16px 18px', 'background': 'rgba(232,195,74,0.05)', 'borderRadius': '14px', 'border': '1px solid rgba(232,195,74,0.12)'},
                        children=[
                            html.P([
                                'Esta herramienta utiliza inteligencia artificial para analizar datos históricos del oro (', html.Code('GC=F', style={'color': '#D4AF37'}), 
                                ') y predecir su dirección a corto plazo. ',
                                'Combina 12 modelos de machine learning (Random Forest, XGBoost, Regresión Logística) ',
                                'con indicadores técnicos (RSI, MACD, Bandas de Bollinger) y variables macroeconómicas (DXY, VIX, TNX). ',
                                'El resultado es una señal de trading diaria con nivel de confianza asociado.'
                            ], style={'color': '#d4c4a8', 'lineHeight': '1.7', 'fontSize': '0.92rem', 'margin': '0 0 8px'}),
                            html.P([
                                html.Strong('¿Para qué sirve? ', style={'color': '#D4AF37'}),
                                'Ayuda a inversores y traders a complementar su análisis con una señal objetiva basada en datos. ',
                                'No reemplaza el juicio humano ni constituye asesoría financiera. Los resultados pasados no garantizan rendimientos futuros.'
                            ], style={'color': '#a89070', 'lineHeight': '1.7', 'fontSize': '0.85rem', 'margin': '0'}),
                        ],
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
                            html.Summary('Gu\u00eda r\u00e1pida del panel', className='help-summary'),
                            html.Div(
                                className='help-copy',
                                children=[
                                    html.H4('Indicadores principales', style={'color': '#D4AF37', 'margin': '10px 0 4px', 'fontFamily': 'Rye, Smokum, serif', 'fontSize': '0.95rem'}),
                                    html.Ul(style={'paddingLeft': '18px', 'margin': '4px 0'}, children=[
                                        html.Li([html.Strong('Precio actual'), ' \u2014 Cotizaci\u00f3n spot del oro (GC=F) en USD/oz. Fuente: Yahoo Finance \u00faltimo cierre.']),
                                        html.Li([html.Strong('Se\u00f1al de mercado'), ' \u2014 Predicci\u00f3n del ensemble (RF+LR+XGB). ALZA (\u226558% confianza), PRECAUCI\u00d3N (\u226442%), ESTABLE (entre 42-58%).']),
                                        html.Li([html.Strong('Certeza'), ' \u2014 Probabilidad que el modelo asigna a su predicci\u00f3n. Ejemplo: 65% = 6.5/10 de acertar. No es la precisi\u00f3n hist\u00f3rica.']),
                                        html.Li([html.Strong('Aciertos (Accuracy)'), ' \u2014 Porcentaje de predicciones correctas en el per\u00edodo de test (20% final de los datos).']),
                                        html.Li([html.Strong('Fiabilidad (F1)'), ' \u2014 Media arm\u00f3nica entre precisi\u00f3n y sensibilidad. Valores >0.5 indican buen equilibrio.']),
                                    ]),
                                    html.H4('Indicadores de contexto', style={'color': '#D4AF37', 'margin': '10px 0 4px', 'fontFamily': 'Rye, Smokum, serif', 'fontSize': '0.95rem'}),
                                    html.Ul(style={'paddingLeft': '18px', 'margin': '4px 0'}, children=[
                                        html.Li([html.Strong('DXY (\u00cdndice del D\u00f3lar)'), ' \u2014 Se correlaciona inversamente con el oro. DXY sube \u2192 oro baja (generalmente).']),
                                        html.Li([html.Strong('VIX (\u00cdndice de Volatilidad)'), ' \u2014 Mide el miedo del mercado. VIX alto \u2192 incertidumbre \u2192 el oro como refugio puede subir.']),
                                        html.Li([html.Strong('MA 21 (Media M\u00f3vil 21d)'), ' \u2014 Tendencia a corto plazo. Precio > MA21 \u2192 tendencia alcista; Precio < MA21 \u2192 tendencia bajista.']),
                                        html.Li([html.Strong('TNX (Bono USA 10a)'), ' \u2014 Rendimiento del tesoro. TNX alto \u2192 menor atractivo del oro (que no paga intereses).']),
                                    ]),
                                    html.H4('Gr\u00e1ficos del panel', style={'color': '#D4AF37', 'margin': '10px 0 4px', 'fontFamily': 'Rye, Smokum, serif', 'fontSize': '0.95rem'}),
                                    html.Ul(style={'paddingLeft': '18px', 'margin': '4px 0'}, children=[
                                        html.Li([html.Strong('Aciertos y Confianza'), ' \u2014 Barras de retorno diario en verde (aciertos) y rojo (fallos), con l\u00ednea de confianza del modelo.']),
                                        html.Li([html.Strong('Rendimiento del Modelo'), ' \u2014 Accuracy, Precisi\u00f3n, Recall, F1 y ROC-AUC en barras con color seg\u00fan valor.']),
                                        html.Li([html.Strong('Precisi\u00f3n acumulada'), ' \u2014 Evoluci\u00f3n del acierto del modelo operaci\u00f3n a operaci\u00f3n.']),
                                        html.Li([html.Strong('Rendimiento ML vs B&H'), ' \u2014 Backtest detallado en la pesta\u00f1a correspondiente.']),
                                    ]),
                                    html.H4('C\u00f3mo usar el panel', style={'color': '#D4AF37', 'margin': '10px 0 4px', 'fontFamily': 'Rye, Smokum, serif', 'fontSize': '0.95rem'}),
                                    html.P('1. Revisa la se\u00f1al del d\u00eda y la certeza en las tarjetas superiores.', style={'margin': '2px 0'}),
                                    html.P('2. Los gr\u00e1ficos inferiores muestran el rendimiento del modelo, no el precio del oro.', style={'margin': '2px 0'}),
                                    html.P('3. Navega por las pesta\u00f1as superiores para ver precio, indicadores t\u00e9cnicos, macro, backtest y simulaci\u00f3n.', style={'margin': '2px 0'}),
                                    html.P(html.Em('Nota: Los datos se actualizan con cada cierre diario. El modelo se re-entrena autom\u00e1ticamente. No es asesor\u00eda financiera.'), style={'marginTop': '8px', 'fontSize': '0.8rem', 'color': '#8D6B34'}),
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
                            html.Div('Aciertos y Confianza del Modelo', className='card-title'),
                            dcc.Loading(
                                type='dot',
                                children=dcc.Graph(id='summary-deviation', figure=build_prediction_deviation_figure(), config={'displayModeBar': False}),
                            ),
                        ],
                    ),
                    html.Div(
                        className='narrow-card',
                        children=[
                            html.Div('Rendimiento del Modelo', className='card-title'),
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
                            html.Div('Rendimiento Acumulado', className='card-title'),
                            dcc.Loading(
                                type='dot',
                                children=dcc.Graph(id='summary-learning', figure=build_learning_curve_figure(), config={'displayModeBar': False}),
                            ),
                        ],
                    ),
                    html.Div(
                        className='narrow-card',
                        children=[
                            html.Div('Resumen del Modelo', className='card-title'),
                            html.Div(
                                className='summary-grid',
                                style={'gridTemplateColumns': '1fr 1fr', 'gap': '10px'},
                                children=[
                                    build_stat_card('Accuracy', f'{context["accuracy"]:.1%}', 'Aciertos totales', ''),
                                    build_stat_card('Precision', f'{context["precision"]:.1%}', 'Calidad de aciertos ALZA', ''),
                                    build_stat_card('Recall', f'{context["recall"]:.1%}', 'Captura de ALZAS reales', ''),
                                    build_stat_card('F1 Score', f'{context["f1"]:.1%}', 'Equilibrio precision-recall', ''),
                                ],
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
                                    html.P('1. Extracci\u00f3n: Datos diarios v\u00eda Yahoo Finance desde 2015 (GC=F, DXY, VIX, TNX).'),
                                    html.P('2. Preprocesamiento: Limpieza, normalizaci\u00f3n y tratamiento de valores faltantes.'),
                                    html.P('3. Feature Engineering: RSI, MACD, medias m\u00f3viles, volatilidad, retornos macro.'),
                                    html.P('4. Modelado: Ensemble con validaci\u00f3n temporal (80% train / 20% test).'),
                                    html.P('5. Evaluaci\u00f3n: Precisi\u00f3n, F1-Score, ROC-AUC, backtest vs Buy & Hold.'),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className='narrow-card',
                        children=[
                            html.Div('Stack Tecnol\u00f3gico', className='card-title'),
                            html.Div(
                                className='governance-markdown',
                                children=[
                                    html.P(html.Strong('Python'), ' 3.10+ \u2014 pandas, numpy, scikit-learn'),
                                    html.P(html.Strong('Modelos'), ' \u2014 Random Forest, Logistic Regression, XGBoost'),
                                    html.P(html.Strong('Frontend'), ' \u2014 Plotly Dash, CSS Grid, Plotly.js'),
                                    html.P(html.Strong('Datos'), ' \u2014 Yahoo Finance (yfinance)'),
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
                            html.Div('Modelos Implementados', className='card-title'),
                            html.Div(
                                className='governance-markdown',
                                children=[
                                    html.P(html.Strong('lr_strong_reg_binary'), ' \u2014 Logistic Regression (C=0.1), modelo principal con mejor F1 (0.70)'),
                                    html.P(html.Strong('rf_binary'), ' \u2014 Random Forest, 100 \u00e1rboles, max_depth=5'),
                                    html.P(html.Strong('xgb_binary'), ' \u2014 XGBoost, 100 estimadores, max_depth=3'),
                                    html.P(html.Strong('lr_multiclass'), ' \u2014 Logistic Regression multiclase (compra/mant\u00e9n/vende)'),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className='narrow-card',
                        children=[
                            html.Div('Se\u00f1ales de Trading', className='card-title'),
                            html.Div(
                                className='governance-markdown',
                                children=[
                                    html.P(html.Strong('ALZA'), ' \u2014 Subida esperada (confianza \u2265 58%)'),
                                    html.P(html.Strong('PRECAUCI\u00d3N'), ' \u2014 Ca\u00edda anticipada (confianza \u2264 42%)'),
                                    html.P(html.Strong('ESTABLE'), ' \u2014 Se\u00f1al no concluyente (42-58%)'),
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
                            html.Div('Equipo DataScope Solutions', className='card-title'),
                            html.Div(
                                className='governance-markdown',
                                children=[
                                    html.P([html.Strong('Maria'), ' (PO)']),
                                    html.P([html.Strong('Juan'), ' (SM)']),
                                    html.P([html.Strong('Jose'), ' (Dev)']),
                                    html.P([html.Strong('Gema'), ' (Dev)']),
                                    html.P([html.Strong('Joel'), ' (Dev)']),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className='narrow-card',
                        style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'center'},
                        children=[
                            html.Div('Repositorio', className='card-title'),
                            html.Img(
                                src='https://api.qrserver.com/v1/create-qr-code/?size=120x120&data=https://github.com/juandelaf1/Golden-Forecast',
                                style={'width': '120px', 'height': '120px', 'borderRadius': '8px', 'border': '2px solid rgba(232,195,74,0.3)'},
                                alt='QR GitHub',
                            ),
                            html.P('github.com/juandelaf1/Golden-Forecast', style={'color': 'var(--muted)', 'fontSize': '0.75rem', 'marginTop': '8px', 'textAlign': 'center'}),
                        ],
                    ),
                ],
            ),
            html.Div(
                className='wide-card',
                children=[
                    html.Div('Limitaciones y Buenas Pr\u00e1cticas', className='card-title'),
                    html.Div(
                        className='governance-markdown',
                        children=[
                            html.P('Datos de cierre diario \u2014 no apto para trading de alta frecuencia.'),
                            html.P('Se\u00f1ales probabil\u00edsticas \u2014 no certezas absolutas. El modelo asigna una confianza, no una garant\u00eda.'),
                            html.P('Rendimiento pasado no garantiza resultados futuros.'),
                            html.P('Modelos re-entrenados autom\u00e1ticamente con cada actualizaci\u00f3n de datos.'),
                            html.P('Gobernanza Scrum \u2014 ramas feature/, PRs con revisi\u00f3n por pares, main protegido.'),
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
    latest = context['latest']
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
                className='help-panel',
                children=[
                    html.Details(
                        className='help-panel-details',
                        children=[
                            html.Summary('\u00bfC\u00f3mo interpretar los indicadores macro?', className='help-summary'),
                            html.Div(
                                className='help-copy',
                                children=[
                                    html.H4(f'DXY (\u00cdndice del D\u00f3lar) \u2014 Actual: {latest.get("dxy", 0):.2f}', style={'color': '#AC7D3D', 'margin': '10px 0 4px', 'fontFamily': 'Rye, Smokum, serif', 'fontSize': '0.9rem'}),
                                    html.Ul(style={'paddingLeft': '18px', 'margin': '4px 0'}, children=[
                                        html.Li('Mide el valor del d\u00f3lar frente a una cesta de 6 divisas (EUR, JPY, GBP, CAD, SEK, CHF).'),
                                        html.Li('Relaci\u00f3n inversa con el oro: cuando el d\u00f3lar se fortalece (DXY sube), el oro tiende a bajar porque se encarece en otras divisas.'),
                                        html.Li(f'Rango t\u00edpico: 90-110. Por encima de 105 se considera d\u00f3lar fuerte; por debajo de 95, d\u00f3lar d\u00e9bil.'),
                                    ]),
                                    html.H4(f'VIX (\u00cdndice de Volatilidad CBOE) \u2014 Actual: {latest.get("vix", 0):.2f}', style={'color': '#7A5C33', 'margin': '10px 0 4px', 'fontFamily': 'Rye, Smokum, serif', 'fontSize': '0.9rem'}),
                                    html.Ul(style={'paddingLeft': '18px', 'margin': '4px 0'}, children=[
                                        html.Li('Conocido como el "\u00edndice del miedo". Mide la volatilidad esperada del S&P 500 para los pr\u00f3ximos 30 d\u00edas.'),
                                        html.Li('Relaci\u00f3n directa con el oro: VIX alto \u2192 incertidumbre en los mercados \u2192 inversores buscan refugio en el oro.'),
                                        html.Li(f'Rango t\u00edpico: 12-20 en mercados estables. >30 indica p\u00e1nico; <12 indica complacencia.'),
                                    ]),
                                    html.H4(f'TNX (Bono del Tesoro USA a 10 a\u00f1os) \u2014 Actual: {latest.get("tnx", 0):.2f}%', style={'color': '#5C4732', 'margin': '10px 0 4px', 'fontFamily': 'Rye, Smokum, serif', 'fontSize': '0.9rem'}),
                                    html.Ul(style={'paddingLeft': '18px', 'margin': '4px 0'}, children=[
                                        html.Li('Rendimiento del bono del gobierno estadounidense a 10 a\u00f1os. Referencia global de tipos de inter\u00e9s.'),
                                        html.Li('Relaci\u00f3n inversa con el oro: TNX alto \u2192 los bonos pagan m\u00e1s intereses \u2192 el oro (que no paga intereses) pierde atractivo.'),
                                        html.Li(f'Rango t\u00edpico: 1.5-5.0%. Por encima de 4.5% presiona a la baja el oro; por debajo de 2.5% lo favorece.'),
                                    ]),
                                    html.P(html.Em('Interpretaci\u00f3n combinada: El movimiento del oro rara vez depende de un solo factor. Lo ideal es observar los tres indicadores juntos para detectar se\u00f1ales de consenso o divergencia.'), style={'marginTop': '10px', 'fontSize': '0.8rem', 'color': '#8D6B34'}),
                                ],
                            ),
                        ],
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
                            html.Div('Aciertos y Confianza del Modelo', className='card-title'),
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
    base = df['gold'].iloc[0]
    scale = 100.0 / base
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['gold'] * scale,
            name='Precio Oro',
            mode='lines',
            line={'color': WANTED_COLORS['gold'], 'width': 3},
        )
    )
    
    # Media móvil 21
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['ma_21'] * scale,
            name='MA 21',
            mode='lines',
            line={'color': WANTED_COLORS['ma'], 'dash': 'dot', 'width': 2},
        )
    )
    
    # Señal del modelo
    test = context['test_data'] if 'test_data' in context else df.iloc[int(len(df)*0.8):]
    fig.add_trace(
        go.Scatter(
            x=test.index,
            y=test['gold'] * scale,
            mode='markers',
            name='Señal Modelo',
            marker={
                'color': [WANTED_COLORS['buy'] if s == 1 else WANTED_COLORS['sell'] for s in context['data']['signal'].iloc[len(context['data'])-len(test):]],
                'size': 12,
                'symbol': 'star',
                'line': {'width': 1, 'color': '#FFFFFF'},
            },
        )
    )
    
    split_y = df['gold'].max() * scale
    fig.add_shape(
        type='line',
        x0=context['split_date'],
        x1=context['split_date'],
        y0=df['gold'].min() * scale,
        y1=split_y,
        line={'color': WANTED_COLORS['iron_mid'], 'dash': 'dash', 'width': 2},
    )
    fig.add_annotation(
        x=context['split_date'],
        y=split_y,
        text='Inicio test set',
        font={'color': '#a89070', 'size': 10, 'family': 'Space Mono, monospace'},
        showarrow=False,
        yshift=10,
    )
    
    fig.update_layout(**PLOT_THEME, height=420, hovermode='x unified')
    fig.update_xaxes(**AXIS_THEME, title_text='Fecha')
    _apply_rangeselector(fig, df)
    fig.update_yaxes(**AXIS_THEME, title_text='Precio indexado (base 100)', autorange=True)
    return fig


def build_rsi_figure() -> go.Figure:
    df = context['data']
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
                    y=df['gold_rsi_14'],
            name='RSI',
            line={'color': WANTED_COLORS['prediction'], 'width': 2},
        )
    )
    fig.add_hline(y=70, line_dash='dash', line_color=WANTED_COLORS['rsi_overbought'], annotation_text='Sobrecompra (70)', annotation_position='top left')
    fig.add_hline(y=30, line_dash='dash', line_color=WANTED_COLORS['rsi_oversold'], annotation_text='Sobreventa (30)', annotation_position='bottom left')
    fig.update_layout(**PLOT_THEME, height=340, hovermode='x unified')
    fig.update_xaxes(**AXIS_THEME, title_text='Fecha')
    _apply_rangeselector(fig, df)
    fig.update_yaxes(**AXIS_THEME, title_text='RSI', range=[15, 85])
    return fig


def build_macd_figure() -> go.Figure:
    df = context['data']
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['gold_macd'],
            name='MACD',
            line={'color': WANTED_COLORS['prediction'], 'width': 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['gold_macd_signal'],
            name='Señal',
            line={'color': WANTED_COLORS['sell'], 'width': 2, 'dash': 'dot'},
        )
    )
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['gold_macd'] - df['gold_macd_signal'],
            name='Histograma',
            marker_color=WANTED_COLORS['ma'],
            opacity=0.8,
        )
    )
    fig.add_hline(y=0, line_color='rgba(255,255,255,0.3)', line_width=1)
    fig.update_layout(**PLOT_THEME, height=340, hovermode='x unified')
    fig.update_xaxes(**AXIS_THEME, title_text='Fecha')
    _apply_rangeselector(fig, df)
    fig.update_yaxes(**AXIS_THEME, title_text='MACD', autorange=True)
    return fig


def build_macro_figure() -> go.Figure:
    df = context['data']
    from numpy import polyfit, polyval
    fig = make_subplots(rows=2, cols=2, subplot_titles=('Oro vs DXY', 'Oro vs VIX', 'Oro vs TNX', 'Indicadores normalizados (base 100)'))

    def scatter_with_trend(x, y, color, row, col, x_label):
        valid = x.notna() & y.notna()
        xv, yv = x[valid].values, y[valid].values
        corr = float(pd.Series(xv).corr(pd.Series(yv)))
        fig.add_trace(go.Scatter(x=xv, y=yv, mode='markers', name=f'ρ={corr:.2f}',
            marker={'color': color, 'size': 4, 'opacity': 0.25}), row=row, col=col)
        if len(xv) > 2:
            slope, intercept = polyfit(xv, yv, 1)
            x_sorted = np.sort(xv)
            fig.add_trace(go.Scatter(x=x_sorted, y=polyval([slope, intercept], x_sorted),
                mode='lines', name='Tendencia', line={'color': color, 'width': 2, 'dash': 'dash'}), row=row, col=col)
        fig.update_xaxes(**AXIS_THEME, title_text=x_label, row=row, col=col)
        fig.update_yaxes(**AXIS_THEME, title_text='Oro (USD/oz)', row=row, col=col)

    scatter_with_trend(df['dxy'], df['gold'], WANTED_COLORS['macro_dxy'], 1, 1, 'DXY (Dólar)')
    scatter_with_trend(df['vix'], df['gold'], WANTED_COLORS['macro_vix'], 1, 2, 'VIX (Volatilidad)')
    scatter_with_trend(df['tnx'], df['gold'], WANTED_COLORS['macro_tnx'], 2, 1, 'TNX (%)')

    fig.add_trace(go.Scatter(x=df.index, y=df['gold'] / df['gold'].iloc[0] * 100,
        name='Oro', line={'color': WANTED_COLORS['gold'], 'width': 2}), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['dxy'] / df['dxy'].iloc[0] * 100,
        name='DXY', line={'color': WANTED_COLORS['macro_dxy'], 'width': 2}), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['vix'] / df['vix'].iloc[0] * 100,
        name='VIX', line={'color': WANTED_COLORS['macro_vix'], 'width': 2}), row=2, col=2)
    fig.add_trace(go.Scatter(x=df.index, y=df['tnx'] / df['tnx'].iloc[0] * 100,
        name='TNX', line={'color': WANTED_COLORS['macro_tnx'], 'width': 2}), row=2, col=2)
    fig.update_xaxes(**AXIS_THEME, title_text='Fecha', row=2, col=2)
    fig.update_yaxes(**AXIS_THEME, title_text='Índice (base 100)', row=2, col=2)
    fig.update_layout(**PLOT_THEME, height=520, hovermode='x unified', showlegend=True)
    return fig


def _metric_color(v: float) -> str:
    if v < 0.35:
        return '#f2554d'
    if v < 0.50:
        return '#f5a623'
    if v < 0.65:
        return '#e8c34a'
    if v < 0.80:
        return '#7bc96a'
    return '#4ade80'

def build_model_figure() -> go.Figure:
    metrics = ['Precisión', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    values = [context['accuracy'], context['precision'], context['recall'], context['f1'], context['auc']]
    colors = [_metric_color(v) for v in values]
    
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
        yaxis={'range': [0, 1.15]}, 
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
    final_alpha = (context['cum_strategy'][-1] - context['cum_bh'][-1]) * 100
    if final_alpha < 0:
        fig.add_annotation(
            x=0.5, y=0.95, xref='paper', yref='paper',
            text=f'⚠ Alpha negativa ({final_alpha:.1f}%) — La estrategia no supera al benchmark',
            showarrow=False,
            font={'color': WANTED_COLORS['sell'], 'size': 11, 'family': 'Space Mono, monospace'},
            bgcolor='rgba(255,82,82,0.15)', bordercolor=WANTED_COLORS['sell'], borderwidth=1,
        )
    fig.update_layout(
        **{**PLOT_THEME, 'margin': {'t': 50, 'b': 50, 'l': 70, 'r': 70}},
        height=400,
        hovermode='x unified',
        title={'text': 'Backtest: Estrategia ML vs Buy & Hold', 'font': {'family': 'Rye, Smokum, serif', 'color': WANTED_COLORS['gold']}},
        yaxis2={'overlaying': 'y', 'side': 'right', 'showgrid': False, 'title': 'Alpha (%)'},
    )
    fig.update_xaxes(**AXIS_THEME, title_text='Fecha')
    fig.update_yaxes(**AXIS_THEME, title_text='Rendimiento acumulado (%)', autorange=True)
    return fig


def build_prediction_deviation_figure() -> go.Figure:
    """Aciertos del modelo: barras de retorno diario coloreadas según acierto/fallo + línea de confianza"""
    test = context['test_data']
    predictions = context.get('predictions', [])
    probabilities = context.get('probabilities', [])
    
    max_points = 90
    n = min(len(predictions), max_points, len(test))
    returns = test['returns'].values[-n:]
    idx = test.index[-n:]
    actuals = test['target_binary'].values[-n:] if 'target_binary' in test.columns else (returns > 0).astype(int)
    preds = np.array(predictions[-n:])
    confs = np.array(probabilities[-n:])
    correct = (preds == actuals)
    
    colors = np.where(correct, '#4ade80', '#f2554d')
    
    returns_pct = returns * 100
    rng = np.max(np.abs(returns_pct)) if len(returns) > 0 else 1
    rng = max(rng, 2)
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=idx,
            y=returns_pct,
            marker_color=colors.tolist(),
            name='Retorno diario',
            opacity=0.7,
            hovertemplate='%{x|%d %b %Y}<br>Retorno: %{y:.2f}%<br>%{text}<extra></extra>',
            text=['✅ Acierto' if c else '❌ Error' for c in correct],
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=confs * 100,
            name='Confianza del Modelo',
            mode='lines',
            line={'color': '#e8c34a', 'width': 2},
            yaxis='y2',
            hovertemplate='%{x|%d %b %Y}<br>Confianza: %{y:.0f}%<extra></extra>',
        )
    )
    
    fig.add_hline(y=0, line_color='rgba(242,235,225,0.2)', line_width=1)
    
    theme_no_legend = {k: v for k, v in PLOT_THEME.items() if k != 'legend'}
    margin_r = rng * 3 + 15
    fig.update_layout(
        **theme_no_legend,
        height=380,
        hovermode='x unified',
        yaxis={'title': 'Retorno diario (%)', 'gridcolor': 'rgba(242,235,225,0.08)', 'range': [-rng * 1.5, rng * 1.5]},
        yaxis2={'title': 'Confianza del modelo (%)', 'overlaying': 'y', 'side': 'right', 'range': [0, 105], 'gridcolor': 'rgba(0,0,0,0)'},
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.02, 'x': 0.5, 'xanchor': 'center', 'font': {'color': '#F2EBE1'}},
    )
    fig.update_xaxes(**AXIS_THEME, title_text='Fecha')
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
        actual = row.get('target_binary', row.get('target_bin', 0))
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
    fig.update_yaxes(**AXIS_THEME, range=[0.3, 1.05])
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


def build_feature_importance_figure(category='all') -> go.Figure:
    """Feature importance chart from Random Forest."""
    model_results = context.get('model_results', {})
    rf_result = model_results.get('rf_binary') or model_results.get(data.PRIMARY_MODEL)
    if not rf_result or 'model' not in rf_result or rf_result['model'] is None:
        fig = go.Figure()
        fig.update_layout(**PLOT_THEME, height=350)
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='No hay datos de importancia de variables', showarrow=False, font={'color': '#D4AF37', 'size': 14})
        return fig

    if not hasattr(rf_result['model'], 'feature_importances_'):
        from sklearn.linear_model import LogisticRegression
        fig = go.Figure()
        fig.update_layout(**PLOT_THEME, height=350)
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='Importancia no disponible para este tipo de modelo', showarrow=False, font={'color': '#D4AF37', 'size': 14})
        return fig

    fi = rf_result['model'].feature_importances_
    fi_indices = np.argsort(fi)[::-1]
    raw_features = [FEATURE_COLUMNS[i] for i in fi_indices]
    labels = [FEATURE_LABELS.get(f, f) for f in raw_features]
    descs = [FEATURE_DESCRIPTIONS.get(f, '') for f in raw_features]
    cats = [FEATURE_CATEGORIES.get(f, 'Otra') for f in raw_features]
    values = [fi[i] for i in fi_indices]

    # Filter by category
    if category != 'all':
        pairs = [(l, v, d, c) for l, v, d, c in zip(labels, values, descs, cats) if c == category]
        if pairs:
            labels, values, descs, cats = zip(*pairs)
        labels = list(labels)
        values = list(values)
        descs = list(descs)
        cats = list(cats)

    bar_colors = [CATEGORY_COLORS.get(c, '#E8C34A') for c in cats]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker={
            'color': bar_colors,
            'line': {'color': WANTED_COLORS['gold'], 'width': 1},
            'cornerradius': 4,
        },
        text=[f'{v:.1%}' for v in values],
        textposition='outside',
        textfont={'color': '#F2EBE1', 'size': 10, 'family': 'Space Mono, monospace'},
        cliponaxis=False,
        hovertemplate='<b>%{y}</b><br>Peso: %{x:.1%}<br>Categor\u00eda: %{customdata[1]}<br>%{customdata[0]}<extra></extra>',
        customdata=list(zip(descs, cats)),
        selected={'marker': {'color': '#F5EDE0', 'opacity': 1}},
        unselected={'marker': {'opacity': 0.4}},
    ))
    max_val = max(values) if values else 0.1
    fig.update_layout(
        **{**PLOT_THEME, 'margin': {'t': 30, 'b': 30, 'l': 180, 'r': 60}},
        height=400,
        clickmode='event+select',
        xaxis={'range': [0, max_val * 1.5], 'showgrid': True, 'gridcolor': 'rgba(242,235,225,0.08)'},
        title={'text': 'Factores que M\u00e1s Influyen en la Predicci\u00f3n', 'font': {'family': 'Rye, Smokum, serif', 'color': WANTED_COLORS['gold']}},
    )
    fig.update_xaxes(**AXIS_THEME, tickformat='.0%', title_text='Peso relativo en la decisi\u00f3n del modelo')
    fig.update_yaxes(**AXIS_THEME, autorange='reversed')
    return fig


def build_confusion_matrix_figure() -> go.Figure:
    """Confusion matrix heatmap."""
    model_results = context.get('model_results', {})
    rf_result = model_results.get(data.PRIMARY_MODEL)
    if not rf_result:
        fig = go.Figure()
        fig.update_layout(**PLOT_THEME, height=350)
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='No hay datos de matriz de confusi\u00f3n', showarrow=False, font={'color': '#D4AF37', 'size': 14})
        return fig

    y_true = context['test_data'].get('target_binary') if hasattr(context['test_data'], 'get') else None
    if y_true is None:
        fig = go.Figure()
        fig.update_layout(**PLOT_THEME, height=350)
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='Columna target_binary no disponible', showarrow=False, font={'color': '#D4AF37', 'size': 14})
        return fig
    cm = confusion_matrix(y_true, rf_result['predictions'])

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
    """ROC curve for the primary model."""
    model_results = context.get('model_results', {})
    rf_result = model_results.get(data.PRIMARY_MODEL)
    if not rf_result:
        fig = go.Figure()
        fig.update_layout(**PLOT_THEME, height=350)
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='No hay datos de curva ROC', showarrow=False, font={'color': '#D4AF37', 'size': 14})
        return fig

    if len(np.unique(rf_result['probabilities'])) < 2:
        fig = go.Figure()
        fig.update_layout(**PLOT_THEME, height=350)
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='ROC no disponible (probabilidades constantes)', showarrow=False, font={'color': '#D4AF37', 'size': 14})
        return fig

    y_true = context['test_data'].get('target_binary') if hasattr(context['test_data'], 'get') else None
    if y_true is None:
        fig = go.Figure()
        fig.update_layout(**PLOT_THEME, height=350)
        fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='target_binary no disponible para ROC', showarrow=False, font={'color': '#D4AF37', 'size': 14})
        return fig
    fpr, tpr, _ = roc_curve(y_true, rf_result['probabilities'])
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


def build_model_comparison_table() -> html.Div:
    """Model comparison tables for classification and regression."""
    model_results = context.get('model_results', {})
    reg_raw = context.get('regression')
    regression_results = None
    if reg_raw:
        rows = []
        for t_name, t_res in reg_raw.items():
            ev = t_res.get('eval_results')
            if ev is not None:
                for _, r in ev.iterrows():
                    rows.append({**r.to_dict(), 'Target': t_name})
        if rows:
            from pandas import DataFrame
            regression_results = DataFrame(rows)
    
    clf_models = {n: r for n, r in model_results.items() if r.get('type') == 'classification'}

    # Classification table
    clf_rows = [
        html.Tr([
            html.Th('Modelo', style=TH),
            html.Th('Precisión', style=TH),
            html.Th('Precision', style=TH),
            html.Th('Recall', style=TH),
            html.Th('F1', style=TH),
            html.Th('ROC-AUC', style=TH),
            html.Th('Rentab.', style=TH),
        ]),
    ]
    for name, r in clf_models.items():
        profit_color = '#22C55E' if r['cum_strategy'][-1] > 0 else '#EF4444'
        clf_rows.append(html.Tr([
            html.Td(name, style=TD),
            html.Td(f'{r["accuracy"]:.1%}', style=TD),
            html.Td(f'{r["precision"]:.1%}', style=TD),
            html.Td(f'{r["recall"]:.1%}', style=TD),
            html.Td(f'{r["f1"]:.3f}', style=TD),
            html.Td(f'{r["auc"]:.3f}' if r['auc'] > 0 else '—', style=TD),
            html.Td(f'{r["cum_strategy"][-1]:+.1%}', style={**TD, 'color': profit_color}),
        ]))

    # Regression table
    reg_rows = [
        html.Tr([
            html.Th('Target', style=TH),
            html.Th('Modelo', style=TH),
            html.Th('MAE', style=TH),
            html.Th('RMSE', style=TH),
            html.Th('R²', style=TH),
        ]),
    ]
    if regression_results is not None:
        for _, row in regression_results.iterrows():
            reg_rows.append(html.Tr([
                html.Td(row['Target'], style=TD),
                html.Td(row['Modelo'], style=TD),
                html.Td(f'{row["MAE"]:.4f}', style=TD),
                html.Td(f'{row["RMSE"]:.4f}', style=TD),
                html.Td(f'{row["R²"]:.4f}', style=TD),
            ]))

    children = []
    if clf_rows:
        children.append(html.Div('Clasificación (sube/baja)', style={'color': '#D4AF37', 'fontFamily': 'Rye, Smokum, serif', 'fontSize': '0.9rem', 'marginBottom': '6px'}))
        children.append(html.Table(style=TABLE, children=clf_rows))
    if reg_rows and len(reg_rows) > 1:
        children.append(html.Div('Regresión (retorno esperado)', style={'color': '#D4AF37', 'fontFamily': 'Rye, Smokum, serif', 'fontSize': '0.9rem', 'margin': '16px 0 6px'}))
        children.append(html.Table(style=TABLE, children=reg_rows))

    return html.Div(children=children)


TH = {'borderBottom': '1px solid #2C1A0C', 'padding': '8px', 'color': '#D4AF37', 'fontFamily': 'Space Mono, monospace', 'fontSize': '12px', 'textAlign': 'center'}
TD = {'borderBottom': '1px solid rgba(44,26,12,0.3)', 'padding': '6px 8px', 'color': '#F5EDE0', 'fontFamily': 'Space Mono, monospace', 'fontSize': '12px', 'textAlign': 'center'}
TD_LEFT = {**TD, 'textAlign': 'left'}
TABLE = {'width': '100%', 'borderCollapse': 'collapse'}


def build_experimental_section() -> html.Div:
    """B2B-friendly horizon comparison table."""
    exp = context.get('experimental')
    if not exp or not exp.get('horizons'):
        return html.Div()

    rows = [
        html.Tr([
            html.Th('Ventana', style=TH),
            html.Th('Precisión', style=TH),
            html.Th('Precision', style=TH),
            html.Th('Recall', style=TH),
            html.Th('F1', style=TH),
            html.Th('ROC-AUC', style=TH),
        ]),
    ]

    for h in exp['horizons']:
        label = f'{h["horizon"]} día' + ('s' if h['horizon'] > 1 else '')
        rows.append(html.Tr([
            html.Td(label, style=TD),
            html.Td(f'{h["accuracy"]:.1%}', style=TD),
            html.Td(f'{h["precision"]:.1%}', style=TD),
            html.Td(f'{h["recall"]:.1%}', style=TD),
            html.Td(f'{h["f1"]:.3f}', style=TD),
            html.Td(f'{h["auc"]:.3f}', style=TD),
        ]))

    children = [
        html.Div('Evaluación Multiciclo', className='card-title'),
        html.P(
            'Análisis del rendimiento del modelo Logistic Regression aplicado a distintos horizontes temporales. '
            'Permite identificar a qué plazo la señal predictiva es más consistente.',
            style={'color': '#a89070', 'fontSize': '0.8rem', 'margin': '4px 0 12px', 'lineHeight': '1.4'},
        ),
        html.Table(style=TABLE, children=rows),
    ]

    # Voting classifier row
    voting = exp.get('voting')
    if voting:
        v_row = html.Tr([
            html.Td('Voting (LR+RF+XGB) — 1 día', style={**TD, 'color': '#D4AF37'}),
            html.Td(f'{voting["accuracy"]:.1%}', style=TD),
            html.Td(f'{voting["precision"]:.1%}', style=TD),
            html.Td(f'{voting["recall"]:.1%}', style=TD),
            html.Td(f'{voting["f1"]:.3f}', style=TD),
            html.Td('—', style=TD),
        ])
        children.append(html.Div(
            'Ensemble: promedio de 3 modelos',
            style={'color': '#D4AF37', 'fontSize': '0.75rem', 'margin': '12px 0 4px', 'fontFamily': 'Rye, Smokum, serif'},
        ))
        children.append(html.Table(style=TABLE, children=[v_row]))

    # Threshold optimization
    best_th = exp.get('best_threshold_1d')
    best_f1 = exp.get('best_f1_at_threshold')
    if best_th is not None:
        children.append(html.P(
            f'Umbral óptimo calculado: {best_th:.2f} (F1: {best_f1:.3f}). '
            f'El umbral por defecto (0.50) se ajusta automáticamente para maximizar la relación acierto/error.',
            style={'color': '#a89070', 'fontSize': '0.75rem', 'margin': '8px 0 0', 'lineHeight': '1.4'},
        ))

    return html.Div(className='wide-card', style={'marginTop': '16px'}, children=children)


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
                    html.Div([
                        html.Span('Factores que Influyen en la Predicci\u00f3n', className='card-title'),
                        dcc.Dropdown(
                            id='fi-category-filter',
                            options=[
                                {'label': 'Todas las variables', 'value': 'all'},
                                {'label': 'T\u00e9cnicas (RSI, MACD, Medias...)', 'value': 'T\u00e9cnico'},
                                {'label': 'Macro (DXY, VIX, TNX...)', 'value': 'Macro'},
                                {'label': 'Precio (Rendimiento Diario)', 'value': 'Precio'},
                            ],
                            value='all',
                            clearable=False,
                            style={
                                'width': '280px', 'marginLeft': 'auto', 'background': '#2C2620', 'color': '#F2EBE1',
                                'border': '1px solid rgba(232,195,74,0.3)', 'borderRadius': '6px', 'fontSize': '0.75rem',
                            },
                        ),
                    ], style={'display': 'flex', 'alignItems': 'center', 'gap': '12px', 'flexWrap': 'wrap'}),
                    dcc.Loading(type='dot', children=dcc.Graph(id='fi-chart', figure=build_feature_importance_figure(), config={'displayModeBar': False})),
                    html.Div(id='fi-detail', style={'color': '#8D6B34', 'fontSize': '0.8rem', 'marginTop': '6px', 'fontFamily': 'Space Mono, monospace', 'minHeight': '20px'}),
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
            build_experimental_section(),
        ],
    )


def build_regression_tab() -> html.Div:
    reg = context.get('regression')

    targets = [
        {
            'title': 'Valor Justo',
            'key': 'fair_value_dist',
            'metric': 'Distancia a media 200d',
            'unit': 'desv. estándar',
            'value_prefix': '',
            'explanation': (
                'Compara el precio actual del oro con su media de 200 días. '
                'Valor positivo = el oro está por encima de su media histórica (más caro de lo habitual). '
                'Negativo = está por debajo (más barato). Ayuda a decidir si es buen momento de compra.'
            ),
        },
        {
            'title': 'Volatilidad Esperada',
            'key': 'realized_vol_20d',
            'metric': 'Volatilidad 20d forward',
            'unit': '% anualizado',
            'value_prefix': '',
            'explanation': (
                'Mide la volatilidad esperada del oro a 20 días vista. '
                'Alta = movimientos bruscos (oportunidad y riesgo). '
                'Baja = mercado estable. Expresada en términos anualizados para comparar con otros activos.'
            ),
        },
        {
            'title': 'Rango de Precio (ATR)',
            'key': 'future_atr_20d',
            'metric': 'Rango medio 20d forward',
            'unit': 'fracción del precio',
            'value_prefix': '',
            'explanation': (
                'Estima cuánto puede moverse el oro cada día de media en las próximas semanas. '
                'Útil para calcular stops de protección y objetivos de precio con fundamento estadístico.'
            ),
        },
        {
            'title': 'Riesgo de Caída',
            'key': 'max_drawdown_20d',
            'metric': 'Máximo drawdown 20d forward',
            'unit': '% desde pico',
            'value_prefix': '-',
            'explanation': (
                'Calcula la mayor caída esperada desde un pico en las próximas 4 semanas. '
                'Fundamental para entender el riesgo real de una inversión en oro y dimensionar posiciones.'
            ),
        },
    ]

    cards = []
    for t in targets:
        best = None
        if reg and t['key'] in reg:
            r = reg[t['key']]
            cv = r.get('cv_results')
            if cv is not None and len(cv) > 0:
                best_idx = cv['R²_mean'].idxmax()
                best = cv.loc[best_idx]

        if best is not None and best['R²_mean'] > 0:
            value_text = f'{t["value_prefix"]}{best["MAE_mean"]:.4f} {t["unit"]}'
            detail = f'Modelo: {best["Modelo"]} — R²: {best["R²_mean"]:.2f} — IC: {best["IC_mean"]:.2f}'
        elif best is not None:
            continue
        else:
            continue

        cards.append(
            html.Div(
                className='metric-card',
                children=[
                    html.Div(t['title'], style={'color': WANTED_COLORS['gold'], 'fontFamily': 'Rye, Smokum, serif', 'fontSize': '1rem'}),
                    html.Div(t['metric'], className='metric-note'),
                    html.Div(value_text, className='metric-value', style={'color': WANTED_COLORS['gold']}),
                    html.Div(detail, style={'color': '#a89070', 'fontSize': '0.75rem', 'marginBottom': '8px'}),
                ],
            )
        )

    if not cards:
        cards = [html.Div('No hay datos disponibles. Ejecuta el pipeline completo para ver el análisis de Valor y Riesgo.',
            style={'color': '#a89070', 'fontSize': '0.9rem', 'textAlign': 'center', 'padding': '40px', 'gridColumn': '1 / -1'})]

    return html.Div(
        className='section-panel',
        children=[
            html.Div(
                className='section-copy',
                children=[
                    html.H2('Valor y Riesgo', className='section-title'),
                    html.P(
                        'An\u00e1lisis complementario de valoraci\u00f3n y riesgo del oro. '
                        'Mientras el modelo principal de clasificaci\u00f3n predice si el precio subir\u00e1 o bajar\u00e1, '
                        'este m\u00f3dulo responde a preguntas diferentes: \u00bfest\u00e1 caro o barato? '
                        '\u00bfcu\u00e1nto puede moverse? \u00bfcu\u00e1l es el riesgo real de ca\u00edda?',
                        className='section-text',
                    ),
                ],
            ),
            html.Div(className='summary-grid', children=cards),
            html.Div(
                className='help-panel',
                children=[
                    html.Details(
                        className='help-panel-details',
                        children=[
                            html.Summary('C\u00f3mo usar esta informaci\u00f3n', className='help-summary'),
                            html.Div(
                                className='help-copy',
                                children=[
                                    html.H4('Modelos de regresi\u00f3n: valoraci\u00f3n de activos y medici\u00f3n de riesgo', style={'color': WANTED_COLORS['gold'], 'margin': '10px 0 4px', 'fontFamily': 'Rye, Smokum, serif'}),
                                    html.P('El modelo de clasificaci\u00f3n predice la direcci\u00f3n del precio (sube/baja). Este an\u00e1lisis de Valor y Riesgo responde a:'),
                                    html.Ul(children=[
                                        html.Li('\u00bfEst\u00e1 el oro en zona de sobrevaloraci\u00f3n o infravaloraci\u00f3n hist\u00f3rica?'),
                                        html.Li('\u00bfCu\u00e1nto riesgo de ca\u00edda existe en las pr\u00f3ximas semanas?'),
                                        html.Li('\u00bfCu\u00e1l es el rango de movimiento esperado?'),
                                    ]),
                                    html.P('Juntos ofrecen una visi\u00f3n completa: saber hacia d\u00f3nde va el precio y entender el contexto de valor y riesgo.'),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
