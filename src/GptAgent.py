# в конфиг
# 'gpt-3.5-turbo-0125'
# 'gpt-4o'

from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

from config import config
from utils.dataset_collection_utils import get_step_score


class GptAgent(LLM):
    OPENAI_MODEL_NAME: Any
    OPENAI_CLIENT: Any
    tokenizer: Any
    mode: Any
    train_df: Any
    temperature: Any

    @property
    def _llm_type(self) -> str:
        return "custom_gpt_agent"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        messages = [
            {"role": "user", "content": prompt},
        ]

        instruction = "You are instruction model. Follow the instractions exactly."
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)

        d = {
            "messages": [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{input_text}"},
            ]
        }

        output = (
            self.OPENAI_CLIENT.chat.completions.create(
                model=self.OPENAI_MODEL_NAME,
                messages=d["messages"],
                max_tokens=2048,
                temperature=self.temperature,
            )
            .choices[0]
            .message.content
        )

        if stop is not None:
            for word in stop:
                output = output.split(word)[0].strip()

        while not output.endswith("```"):
            output += "`"

        if self.mode == "dataset_collection":
            self.train_df = get_step_score(self.train_df, input_text, output)

        return output
