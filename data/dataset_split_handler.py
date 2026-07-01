
""" Convenient dataset splitting. Add new split functions here.  """


from data.supported_datasets import SupportedDatasets
from data.cortex import Cortex

# Mapping supported datasets to split functions
dataset_split_handler = {
    SupportedDatasets.DATASET_NAME.name: Cortex.split,
}

