import numpy as np
from nanoid import generate
import argparse
import os
import yaml
from config import config
from openai import OpenAI
import httpx
import warnings
import torch
from transformers.models.llama.tokenization_llama_fast import LlamaTokenizerFast
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from langchain.tools import BaseTool
import datetime
from typing import List
import pandas as pd
import os
from src.tools import GetCurrentTime, GetAvailibleCities, AskUserForInfo, AskUserToChoose, GetFlights, BuyTicket
from src.GptAgent import GptAgent
from src.prompts import system, human
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_json_chat_agent
from src.CustomAgentExecutor import CustomAgentExecutor
from utils.utils import get_last_commit_hash
from src.MistralAgent import MistralAgent
from utils.exceptions import UnknownModelException
from utils.s3_functions import download_file
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
    
    
def load_params(file_path):
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params


def main(config_path):
    
    if not os.path.exists('data/airplane_schedule.csv'):
        download_file('airplane_schedule.csv', 'data/airplane_schedule.csv')
    if not os.path.exists('data/eval_dataset.csv'):
        download_file('eval_dataset.csv', 'data/eval_dataset.csv')
    
    params = load_params(config_path)
    
    if params['mode'] == 'dataset_collection':
        if os.path.exists(params['dataset']):
            train_df = pd.read_csv(params['dataset'])
        else:
            train_df = pd.DataFrame(columns=['input', 'output'])
    else:
        train_df = None
    
    if params['mode'] == 'evaluation':
        evaluation_artifact = pd.DataFrame(columns=['request', 'conversation', 'score'])
    else:
        evaluation_artifact = None
        
    temperature = 0 if params['mode'] == 'evaluation' else 0.3
    
    if params['model_type'] == 'gpt':
        tokenizer = AutoTokenizer.from_pretrained(params['tokenizer'])
        OPENAI_CLIENT = OpenAI(api_key=config.OPENAI_API_KEY,
                            http_client=httpx.Client(proxy=config.PROXY_URL)) \
            if config.PROXY_URL \
                else OpenAI(api_key=config.OPENAI_API_KEY)
        llm = GptAgent(OPENAI_MODEL_NAME=params['model_name'], OPENAI_CLIENT=OPENAI_CLIENT, tokenizer=tokenizer, mode=params['mode'], train_df=train_df, temperature=temperature)
    elif params['model_type'] == 'mistral':
        quantization_config = BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForCausalLM.from_pretrained(params['model_name'], device_map="auto", quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(params['model_name'])
        tokenizer.pad_token = tokenizer.eos_token
        llm = MistralAgent(model=model, tokenizer=tokenizer, mode=params['mode'], train_df=train_df, temperature=temperature)
    else:
        raise UnknownModelException('Unknown model.')
    
    get_current_time_tool = GetCurrentTime()
    get_availible_cities_tool = GetAvailibleCities()
    ask_user_info_tool = AskUserForInfo()
    get_flights_tool = GetFlights()
    ask_user_to_choose_tool = AskUserToChoose()
    buy_ticket_tool = BuyTicket()
    tools = [get_current_time_tool, get_availible_cities_tool, ask_user_info_tool, get_flights_tool, ask_user_to_choose_tool, buy_ticket_tool]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", human),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_json_chat_agent(
        tools = tools,
        llm = llm,
        prompt = prompt,
        stop_sequence = ["STOP"],
        template_tool_response = "{observation}"
    )
    agent_executor = CustomAgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10, 
                                         mode=params['mode'], evaluation_artifact=evaluation_artifact)

    
    if params['mode'] == 'evaluation':
        eval_dataset = pd.read_csv(params['dataset'])
        experiment_id = generate('1234567890qwertyuiopasdfghjklzxcvbnm', 20)
        for request in eval_dataset['request'].values:
            # try:
            print(request)
            agent_executor.invoke({"input": request})
            # except:
            #     1/0
                # agent_executor.evaluation_artifact.loc[len(agent_executor.evaluation_artifact)] = [request, '', 0]
        evaluation_artifact = agent_executor.evaluation_artifact
        artifact_path = f'experiments/artifacts/{experiment_id}.csv'
        evaluation_artifact.to_csv(artifact_path)
        experiment_data = {
            'commit_hash': get_last_commit_hash(),
            'experiment_id': experiment_id,
            'config_path': config_path,
            'params': params,
            'score': round(float(sum(evaluation_artifact['score'].values)/(len(evaluation_artifact)*10) * 100), 2),
            'artifact_path': artifact_path}
        with open(f'experiments/{experiment_id}.yaml', 'w') as file:
            yaml.dump(experiment_data, file, default_flow_style=False)
            
    else:
        request = input('Please provide yoy request: ')
        print(agent_executor.invoke({"input": request}))
    
    if params['mode'] == 'dataset_collection':
        train_df.to_csv(params['dataset'], index=False)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Start agent using yaml-config.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to yaml-config.")
    args = parser.parse_args()
    main(config_path=args.config_path)