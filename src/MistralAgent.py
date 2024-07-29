from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from transformers import pipeline

from config import config
from utils.dataset_collection_utils import get_step_score


class MistralAgent(LLM):
    model: Any
    tokenizer: Any
    mode: Any
    train_df: Any
    temperature: Any

    @property
    def _llm_type(self) -> str:
        return "custom_mistral_agent"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        messages = [
            {"role": "user", "content": prompt},
        ]

        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)[
            :40000
        ]

        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
        )
        result = pipe(f"{input_text}")

        output = (
            result[0]["generated_text"].split("[/INST]")[1].replace("</s>", "").strip()
        )

        if stop is not None:
            for word in stop:
                output = output.split(word)[0].strip()

        while not output.endswith("```"):
            output += "`"

        if self.mode == "dataset_collection":
            self.train_df = get_step_score(self.train_df, input_text, output)

        return output
