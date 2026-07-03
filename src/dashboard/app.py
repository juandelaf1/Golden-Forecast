import os

import dash
from flask import jsonify

from src.dashboard import data, layout, callbacks

app = dash.Dash(__name__, title='Golden Forecast', suppress_callback_exceptions=True, update_title=None)
server = app.server

# Health endpoint for Render monitoring
@server.route('/health')
def health():
    return jsonify({'status': 'healthy', 'service': 'golden-forecast'})

app.layout = layout.build_layout()
callbacks.register_callbacks(app)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(debug=False, host='0.0.0.0', port=port)
