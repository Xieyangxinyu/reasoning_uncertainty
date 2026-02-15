import os
from pathlib import Path
import pandas as pd
from .base import BaseDatasetGenerator
from datasets import load_dataset

class MCGenerator(BaseDatasetGenerator):
    """Dataset from https://arxiv.org/abs/2507.23407."""

    def __init__(self, 
                 data_dir="_data/mc", 
                 data_file="MC.csv",
                 **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.url = "https://raw.githubusercontent.com/XMUDeepLIT/Learn2Ask/refs/heads/main/eval/data/gsm-mc.jsonl"
        self.data_file = data_file

        if not os.path.exists(Path(self.data_dir) / self.data_file):
            self._download_data()

        self.original_dataset = pd.read_csv(Path(self.data_dir) / self.data_file).to_dict(orient="records")
        self.dataset = self.create_dataset()
        self.format_prompt = 'Please reason step by step, and put your final answer within \boxed{}, e.g., Answer: \\boxed{45}'

    def __len__(self):
        return self.max_num_samples or len(self.dataset)

    def _download_data(self):
        # pandas can read JSONL directly from a URL
        df = pd.read_json(self.url, lines=True)
        other_dataset = load_dataset("openai/gsm8k", "main", split="test")
        other_questions = set(other_dataset[i]["question"] for i in range(400))
        df = df[~df['question'].isin(other_questions)]
        # sample 100 examples randomly
        sampled = df.sample(n=100, random_state=42)
        # create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        sampled.to_csv(Path(self.data_dir) / self.data_file, index=False)

    def create_dataset(self):
        dataset = []
        for row in self.original_dataset:
            q = {
                "question_ill_posed": row["modified"], 
                "question": row["question"],
                "ref_answer": row["answer"].split("#### ")[-1].strip()
            }
            dataset.append(q)
        return dataset
