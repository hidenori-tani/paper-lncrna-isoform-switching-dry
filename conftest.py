"""Put the code/ directory on sys.path so tests and scripts import modules by bare name
(from isoform_metrics import ...). The directory is intentionally NOT a package named `code`,
which would shadow Python's stdlib `code` module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "code"))
