project = "OpenQuantumSim"
author = "Mohammad Jafari"
copyright = "2026, Mohammad Jafari"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "publishing.rst",
    "quickstart_validation.rst",
    "release_checklist.md",
]
html_theme = "furo"
html_title = "OpenQuantumSim"
html_baseurl = "https://mohammadjafariph.github.io/OpenQuantumSimulation/"

html_theme_options = {
    "sidebar_hide_name": True,
    "light_css_variables": {
        "color-brand-primary": "#0e7490",
        "color-brand-content": "#0e7490",
    },
    "dark_css_variables": {
        "color-brand-primary": "#22d3ee",
        "color-brand-content": "#22d3ee",
    },
}

autodoc_member_order = "bysource"
autodoc_typehints = "description"
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath"]
