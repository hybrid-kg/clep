# -*- coding: utf-8 -*-

import os
import re
import sys

sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'sphinx_click.ext',
]
templates_path = ['_templates']
source_suffix = '.rst'
master_doc = 'index'
project = 'CLEP'
copyright = '2019-2020, Vinay Bharadhwaj, Daniel Domingo-Fernández and Charles Tapley Hoyt'
author = 'Vinay Bharadhwaj, Daniel Domingo-Fernández and Charles Tapley Hoyt'

release = '0.0.5'

parsed_version = re.match(
    '(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:-(?P<release>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+(?P<build>[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?',
    release
)
version = parsed_version.expand('\g<major>.\g<minor>.\g<patch>')

tags = set()

if parsed_version.group('release'):
    tags.add('prerelease')

language = None
exclude_patterns = []
pygments_style = 'sphinx'
todo_include_todos = True
html_theme = 'sphinx_rtd_theme'
html_static_path = []
htmlhelp_basename = 'cleppdoc'
html_logo = 'docs_logo.png'
html_favicon = 'favicon.ico'
latex_elements = {}
latex_documents = [
    (master_doc, 'clep.tex', 'CLEP Documentation',
     'Vinay Bharadhwaj, Daniel Domingo-Fernández and Charles Tapley Hoyt', 'manual'),
]
man_pages = [
    (master_doc, 'clep', 'CLEP Documentation', [author], 1)
]
texinfo_documents = [
    (master_doc, 'clep', 'CLEP Documentation', author, 'CLEP'),
]

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
}

autodoc_member_order = 'bysource'
autoclass_content = 'both'

autodoc_mock_imports = ["r2py"]

if os.environ.get('READTHEDOCS'):
    tags.add('readthedocs')
