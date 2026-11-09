"""English result summary; training and validation are implemented by the original runner."""
from pathlib import Path
import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from summarize_research import summarize

def report(final=False):
    return summarize(HERE, "architecture", final=final)
