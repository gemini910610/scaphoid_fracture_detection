import warnings

from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm, trange

warnings.filterwarnings('ignore', category=TqdmExperimentalWarning)
