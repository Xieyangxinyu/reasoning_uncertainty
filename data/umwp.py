import os
from pathlib import Path
import pandas as pd
from .base import BaseDatasetGenerator
from datasets import load_dataset

class UMWPGenerator(BaseDatasetGenerator):
    """Dataset from https://arxiv.org/abs/2403.03558."""

    CATEGORY_MAP = {
        1: "Key information missing",
        2: "Ambiguous key information",
        3: "Unrealistic conditions",
        4: "Unrelated object",
        5: "Question missing",
    }

    def __init__(self, data_dir="_data/umwp", **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self.data_file = "UMWP.csv"

        if not os.path.exists(Path(self.data_dir) / self.data_file):
            self._download_data()

        self.original_dataset = pd.read_csv(Path(self.data_dir) / self.data_file).to_dict(orient="records")
        self.dataset = self.create_dataset()
        self.format_prompt = 'Please reason step by step, and put your final answer within \boxed{}, e.g., Answer: \\boxed{45}'

    def __len__(self):
        return self.max_num_samples or len(self.dataset)

    def _download_data(self):
        url = "https://raw.githubusercontent.com/Yuki-Asuuna/UMWP/refs/heads/main/data/StandardDataset.jsonl"
        # pandas can read JSONL directly from a URL
        df = pd.read_json(url, lines=True)
        df = df[df['source'] != "GSM8K"]

        sampled = df[df['answerable'] == False].groupby("category", group_keys=False).apply(lambda x: x.sample(min(len(x), 50), random_state=42))
        for idx, row in sampled.iterrows():
            relevant_id = row["relevant_ids"][0]
            if relevant_id is not None:
                relevant_df = df[df["id"] == int(relevant_id)]
                relevant_question = relevant_df['question'].values[0]
                relevant_answer = relevant_df['answer'].values[0][0]
                if len(relevant_question) > 0:
                    sampled.at[idx, "relevant_question"] = relevant_question
                    sampled.at[idx, "answer"] = relevant_answer
                else:
                    sampled.at[idx, "relevant_question"] = ""
            else:
                sampled.at[idx, "relevant_question"] = ""
        
        # create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        sampled.to_csv(Path(self.data_dir) / self.data_file, index=False)

    def create_dataset(self):
        dataset = []
        for row in self.original_dataset:
            q = {"question_ill_posed": row["question"],
                 "question": row["relevant_question"],
                 "ref_answer": row["answer"],
                 "category": self.CATEGORY_MAP.get(row["category"], "Unknown")}
            dataset.append(q)
        return dataset