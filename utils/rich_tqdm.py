import warnings

from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)
