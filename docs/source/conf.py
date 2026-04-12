# Configuration file for the Sphinx documentation builder.

import os
import sys

DOCS_SOURCE_DIR = os.path.abspath(os.path.dirname(__file__))
DOCS_DIR = os.path.abspath(os.path.join(DOCS_SOURCE_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(DOCS_SOURCE_DIR, "..", ".."))
SRC_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, "src"))

sys.path.insert(0, DOCS_SOURCE_DIR)
sys.path.insert(0, DOCS_DIR)
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SRC_DIR)

project = "QuVINE"
copyright = "2026 QuVINE"
author = "QuVINE Team"
release = "0.1.0"

autodoc_mock_imports = [
    "torch",
    "node2vec",
    "gensim",
    "hiperwalk",
    "hydra",
    "hydra_core",
    "omegaconf",
    "skdim",
    "hfda",
    "matplotlib",
    "sklearn",
    "scipy",
    "pandas",
    "numpy",
]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

autosummary_generate = True
add_module_names = False
napoleon_google_docstring = True
napoleon_include_init_with_doc = True
todo_include_todos = True

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True
html_title = "QuVINE Documentation"
html_logo = "_static/quvine.png"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/IBM/QuVINE",
            "icon": "fab fa-github",
            "type": "fontawesome",
        }
    ],
    "show_prev_next": True,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "header_links_before_dropdown": 8,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "collapse_navigation": False,
    "navigation_with_keys": True,
}

html_context = {
    "default_mode": "light",
}

myst_enable_extensions = [
    "colon_fence",
    "strikethrough",
    "tasklist",
]

rst_epilog = """
.. |ai_note| replace:: *Portions of this documentation were generated with AI assistance.*
"""

html_sidebars = {
    "**": ["sidebar-nav-bs"],
}

pygments_style = "sphinx"

# Made with Bob
