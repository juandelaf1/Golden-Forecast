from dash import dcc, html
from src.dashboard import data

context = data.context


def build_layout() -> html.Div:
    return html.Div(
        className='scene-shell',
        children=[
            # ── Capa 1: Fondo hiperrealista (local de compra-venta de oro) ──
            html.Div(
                className='scene-background',
                style={
                    'backgroundImage': 'url(/assets/background.png)',
                    'backgroundSize': 'cover',
                    'backgroundPosition': 'center',
                    'transform': 'scale(1.03)',
                },
            ),
            # ── Capa 2: Viñeta oscura ──
            html.Div(className='scene-vignette'),
            # ── Capa 3: Luz cálida de lámpara con parpadeo ──
            html.Div(className='scene-lamp-flicker'),
            # ── Capa 4: Polvo ambiental (inyectado por JS) ──
            html.Div(className='scene-dust', id='scene-dust'),

            # ── MÁQUINA TRAGAMONEDAS ──
            html.Div(
                className='machine-shell',
                children=[
# ── Tapa superior con remaches ──
                      html.Div(
                          className='machine-top',
                          style={'position': 'relative'},
                          children=[
                              html.Div(
                                  'Intelligence · Insight · Trust',
                                  className='machine-subtitle',
                              ),
                          ],
                      ),

                    # ── Cinta móvil Wanted Ticker ──
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

                    # ── Pantalla interior ──
                    html.Div(
                        className='machine-screen',
                        children=[
                            html.Div(className='machine-glow'),
                            html.Div(
                                className='page-shell',
                                children=[
# ── Hero / cabecera informativa ──
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
                                                                 'Pronóstico diario del precio del oro impulsado por Machine Learning.',
                                                                 className='hero-subtitle',
                                                                 ),
                                                         ],
                                                     ),
                                                     html.Img(src='/assets/logo.png', className='hero-logo', style={'height': '80px'}),
                                                 ],
                                             ),
                                         ],
                                     ),

# ── Tabs con lenguaje B2B ──
                                     dcc.Tabs(
                                         id='dashboard-tabs',
                                         value='tab-summary',
                                         className='dashboard-tabs',
                                         children=[
                                             dcc.Tab(
                                                 label='Panel de Control',
                                                 value='tab-summary',
                                                 className='tab-button',
                                                 selected_className='tab-button--selected',
                                             ),
                                             dcc.Tab(
                                                 label='Precio y Señales',
                                                 value='tab-price',
                                                 className='tab-button',
                                                 selected_className='tab-button--selected',
                                             ),
                                             dcc.Tab(
                                                 label='Indicadores Técnicos',
                                                 value='tab-indicators',
                                                 className='tab-button',
                                                 selected_className='tab-button--selected',
                                             ),
                                             dcc.Tab(
                                                 label='Correlaciones Macro',
                                                 value='tab-macro',
                                                 className='tab-button',
                                                 selected_className='tab-button--selected',
                                             ),
                                             dcc.Tab(
                                                 label='Backtest y Estrategia',
                                                 value='tab-backtest',
                                                 className='tab-button',
                                                 selected_className='tab-button--selected',
                                             ),
                                             dcc.Tab(
                                                 label='Simulación',
                                                 value='tab-sim',
                                                 className='tab-button',
                                                 selected_className='tab-button--selected',
                                             ),
                                             dcc.Tab(
                                                 label='Métricas',
                                                 value='tab-metrics',
                                                 className='tab-button',
                                                 selected_className='tab-button--selected',
                                             ),
                                             dcc.Tab(
                                                 label='Metodología',
                                                 value='tab-governance',
                                                 className='tab-button',
                                                 selected_className='tab-button--selected',
                                             ),
                                         ],
                                     ),

                                    html.Div(id='tab-content', className='tab-content'),

                                    # ── Footer ──
                                    html.Footer(
                                        className='footer-panel',
                                        children=[
                                            html.Div(
                                                className='footer-status',
                                                children=[
                                                    html.Div('Estado del sistema', className='footer-title'),
                                                    html.Div(
                                                        f'{context["load_source"].upper()} · Última actualización {context["loaded_at"]}',
                                                        className='footer-detail',
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                className='footer-audio',
                                                children=[
                                                    html.Div('Ambientación sonora', className='audio-label'),
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

                    # ── Base de la máquina con palanca ──
                    html.Div(
                        className='machine-base',
                        children=[
                            html.Div(
                                className='machine-lever',
                                children=html.Div(className='machine-handle'),
                            ),
                            html.Div(
                                className='gen-button',
                                children=[
                                    html.Button(
                                        'GENERAR PRONÓSTICO',
                                        id='gen-button',
                                        className='button-wanted',
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
