"""
Web module - Flask application
"""

from .flask_app import create_app, get_app, app

__all__ = [
    'create_app',
    'get_app',
    'app'
]
