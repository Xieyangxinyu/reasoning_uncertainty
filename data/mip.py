from .base import BaseDatasetGenerator
import requests
import datasets
import random

class MIPGenerator(BaseDatasetGenerator):
    """
    This is not a multiple choice dataset.
    Answers are numeric
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.dataset_name = "mip"
        url, data_dir = "https://raw.githubusercontent.com/tianyi-lab/MiP-Overthinking/refs/heads/main/data/math.json", "_data/mip/math500"
        try:
            self.original_dataset = datasets.Dataset.load_from_disk(data_dir)
        except:
            response = requests.get(url)
            data = response.json()
            self.original_dataset = datasets.Dataset.from_list(data)

        self.dataset = self.create_dataset()
        self.format_prompt = 'Please reason step by step, and put your final answer within \\boxed{}, e.g., Answer: \\boxed{45}'

    def remove_context(self, question: str) -> str:
        # remove the first half of all sentences in the question
        sentences = question.split('. ')
        question_ill_posed = '. '.join(sentences[len(sentences) // 2:]).strip()
        return question_ill_posed
    
    def create_dataset(self):
        dataset = []
        for row in self.original_dataset:
            q = {
                "question_ill_posed": row["insufficient_question"],
                "question": row["question"],
                "ref_answer": row["answer"].strip()
            }
            dataset.append(q)
        return dataset

if __name__ == "__main__":
    # Example usage
    mip = MIPGenerator()
    print(f"MIP dataset size: {len(mip.dataset)}")

    for i in range(3):
        print(mip.dataset[i])
    