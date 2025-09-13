import os
from transformer_lens.utils import get_device
from dataset import CustomDataset
from model import get_model
import pickle


def create_dataset(dir, name, input_csv):
    #  Set device
    device = get_device()
    print(f"Device: {device}")

    #  Initialize model
    model = get_model()

    #  Initialize dataset
    dataset = CustomDataset(
        model=model,
        data_path=input_csv,
    )

    #  Save dataset to a file
    dataset_path = os.path.join(dir, name)  
    with open(dataset_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {dataset_path}")


if __name__ == "__main__":
    dir = "/teamspace/studios/this_studio/mech_interp_exploration/data"
    name = "dataset_v2.pkl"
    input_csv = "/teamspace/studios/this_studio/mech_interp_exploration/data/1000_convs_gpt5_mini.csv"
    os.makedirs(dir, exist_ok=True)
    create_dataset(dir, name, input_csv)
