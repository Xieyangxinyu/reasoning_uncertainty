import re
from .base import BaseDatasetGenerator
import os
import pickle
from datasets import load_dataset
from tqdm import tqdm

class GPQAGenerator(BaseDatasetGenerator):
    """
    Multiple choice graduate level science questions
    that are not googleable.
    
    We combine main, extended and diamond.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cache_dir = "_data/gpqa/"
        os.makedirs(cache_dir, exist_ok=True)
        # Try to load cached dataset list
        try:
            self.original_dataset = [
                load_dataset("csv", data_files=f"_data/gpqa/{n}.csv", split='train') 
                for n in ["gpqa_diamond", "gpqa_main", "gpqa_extended"]
            ]
        except (FileNotFoundError, pickle.PickleError, EOFError):
            self.original_dataset = [load_dataset("Idavidrein/gpqa", n, split="train") for n in ["gpqa_diamond", "gpqa_main", "gpqa_extended"]]
        
        for dataset in self.original_dataset:
            self.dataset += self.create_dataset(dataset)

    def create_dataset(self, original_dataset):
        dataset = []
        for q in original_dataset:
            if re.search(self.context_regex_pattern, q["Question"]):
                question = q["Question"]
                question_ill_posed = self.remove_context(question)
                choices = [
                    self._preprocess(q["Incorrect Answer 1"]),
                    self._preprocess(q["Incorrect Answer 2"]),
                    self._preprocess(q["Incorrect Answer 3"]),
                    self._preprocess(q["Correct Answer"]),
                ]
                choices_text, correct_answer_index = self.shuffle_choices(
                    choices, self._preprocess(q["Correct Answer"])
                )
                dataset.append({
                        "question": question + "\n" + choices_text,
                        "question_ill_posed": question_ill_posed + "\n" + choices_text,
                        'answer': q["Correct Answer"],
                        'ref_answer': correct_answer_index
                    }
                )
        return dataset

    def _preprocess(self, text):
        if text is None:
            return " "
        text = text.strip()
        text = text.replace(" [title]", ". ")
        text = re.sub("\\[.*?\\]", "", text)
        text = text.replace("  ", " ")
        return text

if __name__ == "__main__":
    gpqa = GPQAGenerator()
    # print length of dataset with and without context
    print(f"Dataset with context: {len(gpqa.dataset)}")

    for i in range(3):
        print(gpqa.dataset[i])
    