__version__ = "0.0.0"

from experimentation import compute_results
from experimentation import visualization

# Only print in interactive mode
import __main__ as main
if not hasattr(main, '__file__'):
    print("""Importing the gpforecaster module. L. Roque. 
    Algorithm to forecast Hierarchical Time Series providing point forecast and uncertainty intervals.\n""")
