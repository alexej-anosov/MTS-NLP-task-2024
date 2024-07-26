from config import config
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Mapping, Any
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

    def _call(self, prompt: str, stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        
        messages = [
         {"role": "user", "content": prompt},
        ]
        
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        encodeds = self.tokenizer.apply_chat_template(messages, return_tensors="pt")
        model_inputs = encodeds.to(self.model.device)

        generated_ids = self.model.generate(model_inputs, max_new_tokens=2048, do_sample=True, pad_token_id=self.tokenizer.eos_token_id, temperature=self.temperature)
        decoded = self.tokenizer.batch_decode(generated_ids)

        output = decoded[0].split("[/INST]")[1].replace("</s>", "").strip()

        if stop is not None:
          for word in stop:
            output = output.split(word)[0].strip()

        while not output.endswith("```"):
          output += "`"
        
        if self.mode == 'dataset_collection':
            self.train_df = get_step_score(self.train_df, input_text, output)
        
        return output