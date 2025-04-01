from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.dataset import ModalityConfig
from gr00t.experiment.data_config import DATA_CONFIG_MAP

# get the data config
data_config = DATA_CONFIG_MAP["custom_coin"]

# get the modality configs and transforms
modality_config = data_config.modality_config()
transforms = data_config.transform()

# This is a LeRobotSingleDataset object that loads the data from the given dataset path.
dataset = LeRobotSingleDataset(
    dataset_path="/home/wangxianhao/data/project/reasoning/openpi/gr00t_dataset/tabletop_dataset",
    modality_configs=modality_config,
    transforms=None,  # we can choose to not apply any transforms
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, # the embodiment to use
)

# This is an example of how to access the data.
print(dataset[0])