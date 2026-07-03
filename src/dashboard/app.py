import os

import dash

from src.dashboard import data, layout, callbacks

app = dash.Dash(__name__, title='Golden Forecast', suppress_callback_exceptions=True, update_title=None)
server = app.server

app.layout = layout.build_layout()
callbacks.register_callbacks(app)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)
