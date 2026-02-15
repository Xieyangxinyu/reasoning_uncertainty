from datasets import load_dataset
from .base import BaseDatasetGenerator
import re

class HLEGenerator(BaseDatasetGenerator):
    """
    This is not a multiple choice dataset.
    Answers are numeric
    """

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.original_dataset = load_dataset("cais/hle", "default", split='test')
        # keep only examples with category != 'Math'
        self.original_dataset = self.original_dataset.filter(lambda ex: ex.get("category") != "Math")
        # remove examples with images
        self.original_dataset = self.original_dataset.filter(lambda ex: ex.get("image") != "None")
        self.original_dataset = self.original_dataset.filter(lambda ex: ex.get('answer_type') == "multipleChoice")
        self.original_dataset = self.original_dataset.remove_columns(
            [col for col in self.original_dataset.column_names if col not in ["question", "answer"]]
        )
        # shuffle the dataset
        self.original_dataset = self.original_dataset.shuffle(seed=42)
        self.dataset = self.create_dataset()
        self.format_prompt = 'Please reason step by step, and put your final answer within \\boxed{}, e.g., Answer: \\boxed{A}'
    
    def create_dataset(self):
        dataset = []
        for item in self.original_dataset:
            
            question, _, answers_text = item['question'].partition("Answer Choices:")
            question = question.strip()
            if re.search(self.context_regex_pattern, question):
                question_ill_posed = self.remove_context(question)
                if question_ill_posed.strip() != question.strip():
                    dataset.append({
                            "question": f"Question: {question}\nChoices:\n{answers_text}\nAnswer: ",
                            "question_ill_posed": f"Question: {question_ill_posed}\nChoices:\n{answers_text}\nAnswer: ",
                            "answer": item["answer"],
                            "ref_answer": item["answer"],
                        })
        return dataset
        