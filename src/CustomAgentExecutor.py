from langchain.agents import AgentExecutor
from typing import Dict, Any, Optional
from langchain_core.agents import AgentFinish
from langchain_core.callbacks import CallbackManagerForChainRun
import pandas as pd
import inspect
from langchain_core.load.dump import dumpd
from langchain_core.outputs import RunInfo
from langchain_core.runnables import RunnableConfig, ensure_config
from langchain.schema import RUN_KEY
from langchain_core.callbacks import CallbackManager
from langchain.chains.base import Chain


def invoke(
        self,
        input: Dict[str, Any],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        config = ensure_config(config)
        callbacks = config.get("callbacks")
        tags = config.get("tags")
        metadata = config.get("metadata")
        run_name = config.get("run_name") or self.get_name()
        run_id = config.get("run_id")
        include_run_info = kwargs.get("include_run_info", False)
        return_only_outputs = kwargs.get("return_only_outputs", False)

        inputs = self.prep_inputs(input)
        
        self.inputs = inputs['input'] # provide apportunity to get inputs from CustomAgentExecutor
        
        callback_manager = CallbackManager.configure(
            callbacks,
            self.callbacks,
            self.verbose,
            tags,
            self.tags,
            metadata,
            self.metadata,
        )
        new_arg_supported = inspect.signature(self._call).parameters.get("run_manager")

        run_manager = callback_manager.on_chain_start(
            dumpd(self),
            inputs,
            run_id,
            name=run_name,
        )
        try:
            self._validate_inputs(inputs)
            outputs = (
                self._call(inputs, run_manager=run_manager)
                if new_arg_supported
                else self._call(inputs)
            )

            final_outputs: Dict[str, Any] = self.prep_outputs(
                inputs, outputs, return_only_outputs
            )
        except BaseException as e:
            run_manager.on_chain_error(e)
            raise e
        run_manager.on_chain_end(outputs)

        if include_run_info:
            final_outputs[RUN_KEY] = RunInfo(run_id=run_manager.run_id)
        return final_outputs
    
    
Chain.invoke = invoke


class CustomAgentExecutor(AgentExecutor):
    mode: str
    evaluation_artifact: Optional[pd.DataFrame] = None
    inputs: Optional[str] = None
    
    
    def _return(
        self,
        output: AgentFinish,
        intermediate_steps: list,
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        if run_manager:
            run_manager.on_agent_finish(output, color="green", verbose=self.verbose)
        final_output = output.return_values
        if self.return_intermediate_steps:
            final_output["intermediate_steps"] = intermediate_steps
        
        if self.mode == 'evaluation': 
            conversation = ''
            for step in intermediate_steps:
                conversation += f'ACTION: {step[0]}\nRESPONSE: {step[1]}\n\n'
            print('\n\n\n'+conversation)
                
            score = int(input('Please provide a score for this conversation: '))
            self.evaluation_artifact.loc[len(self.evaluation_artifact)] = [self.inputs, conversation, score]
        
        return final_output
