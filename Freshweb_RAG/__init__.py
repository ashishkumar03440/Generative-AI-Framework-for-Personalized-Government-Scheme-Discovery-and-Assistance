"""FreshWeb_RAG package initialization."""
import importlib, sys
# Load the original capitalized package
module = importlib.import_module('Freshweb_RAG')
# Register it under the lowercase name
sys.modules[__name__] = module