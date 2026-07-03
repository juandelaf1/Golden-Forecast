from dash import dcc, html
from src.dashboard import data

context = data.context


def build_layout() -> html.Div:
    return html.Div(
        className='scene-shell',
        children=[
            html.Div(
                className='scene-background',
                style={
                    'backgroundImage': 'url(/assets/background.png)',
                    'backgroundSize': 'cover',
                    'backgroundPosition': 'center',
                },
            ),
            html.Div(className='scene-vignette'),
            html.Div(className='scene-lamp-flicker'),
            html.Div(className='scene-dust', id='scene-dust'),

            html.Div(
                className='machine-shell',
                children=[
                    html.Div(
                        className='machine-top',
                        style={'position': 'relative'},
                        children=[
                            html.Div(
                                'Intelligence \u00b7 Insight \u00b7 Trust',
                                className='machine-subtitle',
                            ),
                        ],
                    ),

                    html.Div(
                        className='wanted-ticker',
                        children=[
                            html.Div(
                                className='ticker-track',
                                children=html.Div(
                                    id='ticker-content',
                                    className='ticker-content',
                                ),
                            ),
                        ],
                    ),

                    html.Div(
                        className='machine-screen',
                        children=[
                            html.Div(className='machine-glow'),
                            html.Div(
                                className='page-shell',
                                children=[
                                    html.Header(
                                        className='hero-panel',
                                        children=[
                                            html.Div(
                                                className='hero-copy',
                                                children=[
                                                    html.Div(
                                                        className='hero-text',
                                                        children=[
                                                            html.Div('GOLDEN FORECAST', className='hero-title'),
                                                            html.Div(
                                                                'Pron\u00f3stico diario del precio del oro impulsado por Machine Learning.',
                                                                className='hero-subtitle',
                                                            ),
                                                        ],
                                                    ),
                                                    html.Img(src='/assets/logo.png', className='hero-logo', style={'height': '80px'}),
                                                ],
                                            ),
                                        ],
                                    ),

                                    dcc.Tabs(
                                        id='dashboard-tabs',
                                        value='tab-summary',
                                        className='dashboard-tabs',
                                        children=[
                                            dcc.Tab(label='Panel de Control', value='tab-summary', className='tab-button', selected_className='tab-button--selected'),
                                            dcc.Tab(label='Precio y Se\u00f1ales', value='tab-price', className='tab-button', selected_className='tab-button--selected'),
                                            dcc.Tab(label='Indicadores T\u00e9cnicos', value='tab-indicators', className='tab-button', selected_className='tab-button--selected'),
                                            dcc.Tab(label='Correlaciones Macro', value='tab-macro', className='tab-button', selected_className='tab-button--selected'),
                                            dcc.Tab(label='Backtest y Estrategia', value='tab-backtest', className='tab-button', selected_className='tab-button--selected'),
                                            dcc.Tab(label='Simulaci\u00f3n', value='tab-sim', className='tab-button', selected_className='tab-button--selected'),
                                            dcc.Tab(label='M\u00e9tricas', value='tab-metrics', className='tab-button', selected_className='tab-button--selected'),
                                            dcc.Tab(label='Metodolog\u00eda', value='tab-governance', className='tab-button', selected_className='tab-button--selected'),
                                        ],
                                    ),

                                    html.Div(id='tab-content', className='tab-content'),

                                    html.Footer(
                                        className='footer-panel',
                                        children=[
                                            html.Div(
                                                className='footer-status',
                                                children=[
                                                    html.Div('Estado del sistema', className='footer-title'),
                                                    html.Div(
                                                        f'{context["load_source"].upper()} \u00b7 \u00daltima actualizaci\u00f3n {context["loaded_at"]}',
                                                        className='footer-detail',
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className='footer-audio',
                                                children=[
                                                    html.Div('Ambientaci\u00f3n sonora', className='audio-label'),
                                                    html.Audio(
                                                        id='ambient-audio',
                                                        src='/assets/saloon.mp3',
                                                        controls=True,
                                                        autoPlay=True,
                                                        loop=True,
                                                        style={'width': '100%', 'maxWidth': '320px'},
                                                    ),
                                                ],
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),

                ],
            ),
        ],
    )
