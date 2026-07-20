# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

project = 'Mandol'
copyright = '2024-2026, Mandol Contributors'
author = 'Mandol Contributors'
release = '0.1.0a1'

language = 'en'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.graphviz',
    'myst_parser',
    'sphinxcontrib.mermaid',
]

# Archived pre-refactor documents are retained for provenance, not publication.
exclude_patterns = [
    '_build',
    'Thumbs.db',
    '.DS_Store',
    'archive/**',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_theme_options = {
    'sidebar_hide_name': False,
    'navigation_with_keys': True,
}

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html

todo_include_todos = False

# -- Options for MyST markdown -----------------------------------------------
myst_enable_extensions = [
    'colon_fence',
    'deflist',
]
