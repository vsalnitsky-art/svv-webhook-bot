"""Public, cache-free client build version used by both web interfaces."""

import json
import os
from pathlib import Path

from flask import jsonify, request


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _file_version():
    try:
        payload = json.loads((_PROJECT_ROOT / 'version.json').read_text(encoding='utf-8'))
        return str(payload.get('version') or '').strip()
    except (OSError, ValueError, TypeError):
        return ''


def client_version():
    """Git/deploy id is automatic; version.json is the portable fallback."""
    for name in ('VSV_APP_VERSION', 'RENDER_GIT_COMMIT', 'SOURCE_VERSION', 'APP_VERSION'):
        value = str(os.getenv(name) or '').strip()
        if value:
            return value
    return _file_version() or 'unknown'


def register_versioning(app):
    @app.route('/api/version', methods=['GET', 'HEAD'])
    def api_client_version():
        return jsonify({'ok': True, 'version': client_version()})

    @app.after_request
    def _fresh_html_and_version(resp):
        # HTML must be revalidated so new versioned asset URLs reach the browser.
        # Static CSS/JS/images keep their normal cache policy.
        if request.path == '/api/version' or resp.mimetype == 'text/html':
            resp.headers['Cache-Control'] = 'no-store, no-cache, max-age=0, must-revalidate'
            resp.headers['Pragma'] = 'no-cache'
            resp.headers['Expires'] = '0'
        return resp
