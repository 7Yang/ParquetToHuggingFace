import os
from datasets import load_dataset
from config_loader import load_config  

class HuggingFaceDownloader:
    def __init__(self, config):
        self.config  = load_config(config)
        self.repo_id = self.config['huggingface']['repo_id']
        self.token   = os.getenv(self.config['huggingface']['token_env_var'])

    def map_function(dataset_info, features):
        return {
            "sample_audio"      : features["audio"],
            "sample_text"       : features["text"],
            "sample_translation": features["translation"],
            "sample_speaker"    : features["translation"],
            "sample_pitch_mean" : features["utterance_pitch_mean"],
            "sample_pitch_std"  : features["utterance_pitch_mean"],
        }

    def download(self, map_func=None):
        # Load the dataset with both train and test splits
        datasets = load_dataset(path=self.repo_id, token=self.token, split=["train", "test"], trust_remote_code=True)

        for dataset in datasets:
            map_func = map_func if map_func else self.map_function
            mapped_dataset = dataset.map(map_func, batched=True)

        return mapped_dataset



