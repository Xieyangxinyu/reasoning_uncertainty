import re
import random
from tqdm import tqdm
import json

def is_number(s):
    return bool(re.match(r'^-?\d+(\.\d+)?$', s))

class BaseDatasetGenerator:
    def __init__(self, **kwargs):
        """
        Each dataset generator should contain the following attributes:
        - format_prompt: A string that describes how the answer should be formatted.
        - context_regex_pattern: Optional. A regex pattern to identify the context in the question.
        Each dataset generator should implement the following methods:
        - create_dataset: Create the dataset with the required attributes.
        """
        # regex identifies sentences that precede the question
        # [context. ][question?]
        self.context_regex_pattern = r"(?<=\. )[^\.\?\!]*\?$"
        self.format_prompt = 'Please reason step by step, and put your final answer within \boxed{}, e.g., Answer: \\boxed{C}'
        if 'additional_format_prompt' in kwargs:
            self.format_prompt += f"\n\n{kwargs['additional_format_prompt']}"
        
        self.dataset = []

    def create_dataset(self):
        '''
        The dataset should contain at least four attributes:
        - question: The question text. If multiple-choice, it should include the choices.
        - question_ill_posed: The question text without necessary context needed to answer it.
        - answer: The correct answer to the question in text format.
        - ref_answer: The short answer to the question, typically a single letter (A, B, C, D) for multiple-choice questions or a number for numerical answers.
        '''
        raise NotImplementedError

    # below are a few utility methods that can be used by the dataset generators
    def remove_context(self, question: str) -> str:
        res = re.search(self.context_regex_pattern, question)
        if res:
            question_ill_posed = res.group().strip()
            return question_ill_posed
        else:
            raise ValueError(
                "The question does not contain a context that can be removed. "
                "Please check the regex pattern or the question format."
            )

    def shuffle_choices(self, choices, correct_answer):
        """
        Shuffle the answer choices and return them as a formatted string.
        """
        random.shuffle(choices)
        choices_text = "\n".join(
            [f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)]
        )
        # find out which choice is the correct answer
        correct_answer_index = chr(65 + choices.index(correct_answer))
        return choices_text, correct_answer_index

    def extract_answer(self, content:str):
        """
        Extract the correct answer from the answer string.
        """

        # truncate the content to where the last boxed{ occurs
        if "boxed{" in content:
            content = content[content.rindex("boxed{"):]
        
        if "boxed{\\frac" in content or "\\dfrac" in content:
            # if the answer is a fraction, we need to extract the numerator and denominator
            if "\\dfrac" in content:
                # if the answer is a LaTeX fraction, we need to extract it
                start = content.index("\\dfrac{") + len("\\dfrac{")
            else:
                start = content.index("\\frac{") + len("\\frac{")
            end = content.index("}", start)
            numerator = content[start:end]
            try:
                start = content.index("{", end) + 1
                end = content.index("}", start)
                denominator = content[start:end]
                return f"{numerator}/{denominator}"
            except:
                return None
        elif "boxed{" in content:
            start = content.index("boxed{") + len("boxed{")
            try:
                end = content.index("}", start)
            except ValueError:
                end = -1
            if '\\text' in content[start:end]:
                # if the answer contains LaTeX, we need to remove it
                # e.g., "boxed{\\text{answer}}"
                start = content.index("{", start) + 1
                try:
                    end = content.index("}", start)
                except ValueError:
                    end = -1
            answer = content[start:end]

            if answer in ["A", "B", "C", "D"]:
                # if the answer is a single letter, we return it
                return answer
            elif ',' in answer:
                answer = answer.replace(',', '')
            if is_number(answer):
                return answer
            if '.' in answer:
                answer = answer[:answer.index(".")].strip()
        return None
    
    def evaluate(self, content, ref_answer):
        answer = self.extract_answer(content)
        if answer:
            ref_answer = str(ref_answer).strip()
            if ',' in answer or ',' in ref_answer:
                # remove commas from the digits for numerical comparison
                answer = answer.replace(',', '')
                ref_answer = ref_answer.replace(',', '')
            if answer in ["A", "B", "C", "D"]:
                return answer == ref_answer
            elif is_number(answer):
                if '$' in ref_answer:
                    ref_answer = ref_answer.replace('$', '').strip()
                if is_number(ref_answer):
                    return float(answer) == float(ref_answer)
                #else:
                #    print(f"answer: {answer}, ref_answer: {ref_answer}")
            else:
                # try exact match
                return answer == ref_answer
        return False
    
    def apply_chat_template(self, tokenizer, question: str, thinking: str = None) -> str:
        messages = [
            {"role": "user", "content": question + "\n\n" + self.format_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
           messages, tokenize=False, add_generation_prompt=True
        )
        if "<think>" not in prompt:
            prompt = prompt + "<think>"

        if "<think>" in thinking:
            thinking = thinking.split("<think>")[1]
        if "</think>" in thinking:
            thinking = thinking.split("</think>")[0]
        prompt = prompt + thinking
        
        assert "<think>" in prompt
        return prompt

    def run_inference(self, inference_instance, output_file, begin_index=0, batch_size=32):
        n = max(len(self.dataset), 0)

        # detect if "question" exists in the dataset at all
        has_question = "question" in self.dataset[0]

        for i in tqdm(range(begin_index, n, batch_size)):
            end_index = min(i + batch_size, n)
            batch = self.dataset[i:end_index]

            if has_question:
                prompts = [row["question"] + "\n\n" + self.format_prompt for row in batch]
                responses = inference_instance.get_response(prompts)

            prompts_ill_posed = [row["question_ill_posed"] + "\n\n" + self.format_prompt for row in batch]
            responses_ill_posed = inference_instance.get_response(prompts_ill_posed)

            for j, row in enumerate(batch):
                row['index'] = i + j
                if has_question:
                    row["reasoning_well_posed"] = responses[j][0]
                    row["response_well_posed"] = responses[j][1]
                row["reasoning_ill_posed"] = responses_ill_posed[j][0]
                row["response_ill_posed"] = responses_ill_posed[j][1]

                with open(output_file, "a") as f:
                    f.write(json.dumps(row) + "\n")

        print("Done")
        