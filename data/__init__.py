from .gpqa import GPQAGenerator
from .gsm8k import GSM8KGenerator
from .mmlu import MMLUMathGenerator
from .mip import MIPGenerator
from .hle import HLEGenerator
from .umwp import UMWPGenerator
from .mc import MCGenerator
from .base import BaseDatasetGenerator

def get_dataset_generator(name: str, **kwargs):
    """
    Returns the dataset generator class based on the provided name.
    """
    if name == "gpqa":
        return GPQAGenerator(**kwargs)
    elif name == "gsm8k":
        return GSM8KGenerator(**kwargs)
    elif name == "mmlu":
        return MMLUMathGenerator(**kwargs)
    elif name == "mip":
        return MIPGenerator(**kwargs)
    elif name == 'hle':
        return HLEGenerator(**kwargs)
    elif name == 'umwp':
        return UMWPGenerator(**kwargs)        
    elif name == 'mc':
        return MCGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown dataset generator: {name}")