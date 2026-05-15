project = "OpenQuantumSim"
author = "Mohammad Jafari"
extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]
templates_path = ["_templates"]
exclude_patterns = [
    "publishing.rst",
    "quickstart_validation.rst",
    "release_checklist.md",
]
html_theme = "alabaster"

autodoc_member_order = "bysource"
autodoc_typehints = "description"
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "dollarmath"]
