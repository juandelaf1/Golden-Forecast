import dash
from dash import html

from src.dashboard import data, layout, callbacks

app = dash.Dash(__name__, title='Golden Forecast', suppress_callback_exceptions=True, update_title=None)
server = app.server

app.layout = layout.build_layout()
callbacks.register_callbacks(app)

if __name__ == '__main__':
    app.run(debug=False, port=8050)
