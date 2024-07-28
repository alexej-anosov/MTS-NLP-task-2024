from typing import Any, List, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM

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

        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        # encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        # model_inputs = encodeds.to(self.model.device)

        # if self.temperature == 0:
        #   generated_ids = self.model.generate(model_inputs, max_new_tokens=512, do_sample=False, pad_token_id=self.tokenizer.eos_token_id)
        # else:
        #   generated_ids = self.model.generate(model_inputs, max_new_tokens=512, do_sample=True, pad_token_id=self.tokenizer.eos_token_id, temperature=self.temperature)
        # decoded = self.tokenizer.batch_decode(generated_ids)

        #
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        # outputs = model.generate(**inputs, max_new_tokens=50,return_dict_in_generate=True, output_scores=True)
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_scores=True,
        )
        decoded = self.tokenizer.decode((outputs["sequences"][0]))
        print(decoded)
        #

        output = decoded.split("[/INST]")[1].replace("</s>", "").strip()

        if stop is not None:
            for word in stop:
                output = output.split(word)[0].strip()

        while not output.endswith("```"):
            output += "`"

        print(f"\n\n\n{input_text}\n{output}")
        if self.mode == "dataset_collection":
            self.train_df = get_step_score(self.train_df, input_text, output)

        return output
