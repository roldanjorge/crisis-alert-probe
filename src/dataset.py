import torch as t
from torch.utils.data import Dataset
import pandas as pd
from tqdm import tqdm
import circuitsvis as cv
from IPython.display import display
from transformer_lens import HookedTransformer
from utils import llama_v2_prompt


class CustomDataset(Dataset):
    def __init__(
        self,
        model,
        data_path="/teamspace/studios/this_studio/mech_interp_exploration/data/1000_convs_gpt5_v2.csv",
    ):
        model: HookedTransformer = model
        self.label2id = {
            "very_happy": 0,
            "happy": 1,
            "slightly_positive": 2,
            "neutral": 3,
            "slightly_negative": 4,
            "sad": 5,
            "very_sad": 6,
        }
        self.labels = []
        self.resids = []
        self.texts = []
        self._length = None
        self.data = pd.read_csv(data_path, header=0)
        self._load_in_data(model=model)

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get item at index idx."""
        return {
            "text": self.texts[idx],
            "resids": self.resids[idx],
            "label": self.labels[idx],
        }

    def _load_in_data(self, model):
        """Load in data from CSV and process with model to get residuals."""
        self._length = len(self.data)
        for idx in tqdm(range(len(self.data))):
            messages = [{"content": self.data.iloc[idx]["message"], "role": "user"}]
            text = llama_v2_prompt(messages)
            text += f" I think the emotional state of this user is"
            with t.no_grad():
                tokens = model.to_tokens(text)
                # Ensure sequence length constraint similar to previous max_length=2048
                if tokens.shape[-1] > 2048:
                    tokens = tokens[:, -2048:]

                device = model.cfg.device
                _, cache = model.run_with_cache(
                    tokens.to(device), remove_batch_dim=False
                )

                resid_posts = []
                for layer in range(model.cfg.n_layers):
                    resid_post = (
                        cache["resid_post", layer][:, -1].detach().cpu().to(t.float)
                    )
                    resid_posts.append(resid_post)

                resid_posts = t.cat(resid_posts, dim=0)
                self.texts.append(text)
                self.resids.append(resid_posts)
                self.labels.append(self.label2id[self.data.iloc[idx]["label"]])

                # Clean up to save memory
                del tokens, cache
                t.cuda.empty_cache()

        print("Data loading complete.")
