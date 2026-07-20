import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'Mandol'
copyright = '2024-2026, Mandol Contributors'
author = 'Mandol Contributors'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.graphviz',
    'myst_parser',
    'sphinxcontrib.mermaid',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
html_static_path = ['_static']
html_theme_options = {
    'light_logo': 'logo-light.png',
    'dark_logo': 'logo-dark.png',
    'sidebar_hide_name': False,
    'navigation_with_keys': True,
}

todo_include_todos = True

myst_enable_extensions = [
    'colon_fence',
    'deflist',
]

language = 'en'
