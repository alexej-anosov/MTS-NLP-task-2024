from config import config
import torch
from langchain.tools import BaseTool
import datetime
import pandas as pd
import os
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from typing import Optional, List, Mapping, Any, Union
import datetime


class GetCurrentTime(BaseTool):
    name = "get current time"
    description = """
Use this tool to get current time. Tool doesn't get any params.
Example of usage:
{"thought": "I need to know the current date to determine the date for next Tuesday",
 "action": "get current time",
 "action_input": {}}
"""
    def _run(self, *args):
        # datetime.datetime.now(), datetime.datetime.now().strftime('%A')
        return datetime.datetime.strptime('2024-07-10 17:00:00', '%Y-%m-%d %H:%M:%S'), 'Wednesday'


class GetAvailibleCities(BaseTool):
    name = "get availible cities"
    description = """
Use this tool to get list of availible cities. 
You can use it to check if cities from the user's request presented in the data.
Maybe there is a spelling mistake, so just correct it. If it is not spelling mistake tell the user that the city is not in the data.
Example of usage:
{"thought": "Check if the cities are available in the data",
 "action": "get availible cities",
 "action_input": {}}
"""

    def _run(self, *args):
        return ['Jakarta', 'Melbourne', 'Reykjavik', 'Munich', 'Madrid',
       'Buenos Aires', 'Auckland', 'Saratov', 'Lisbon', 'Mexico City',
       'Las Vegas', 'Kuala Lumpur', 'Warsaw', 'Doha', 'Brussels',
       'Toronto', 'Krasnoyarsk', 'Miami', 'New York', 'Barcelona',
       'Nizhny Novgorod', 'Havana', 'Los Angeles', 'Berlin', 'Geneva',
       'Washington', 'Seoul', 'Frankfurt', 'Bogota', 'Tolyatti', 'Zurich',
       'Rio de Janeiro', 'Johannesburg', 'Beijing', 'Tyumen', 'Ufa',
       'Voronezh', 'Luxembourg', 'Kazan', 'Dublin', 'Rome', 'Krasnodar',
       'Moscow', 'Dubai', 'Vienna', 'Lima', 'Delhi', 'Saint Petersburg',
       'Boston', 'London', 'Athens', 'Izhevsk', 'Bangkok',
       'Yekaterinburg', 'Montreal', 'Omsk', 'Perm', 'Prague',
       'Novosibirsk', 'Vancouver', 'Hong Kong', 'Tel Aviv', 'Amsterdam',
       'Paris', 'Volgograd', 'Cairo', 'Stockholm', 'Sydney',
       'Chelyabinsk', 'Helsinki', 'Edinburgh', 'Rostov-on-Don',
       'Istanbul', 'Copenhagen', 'Oslo', 'Tokyo', 'Singapore',
       'Milan', 'Shanghai', 'Budapest', 'Abu Dhabi', 'Mumbai', 'Santiago',
       'Chicago', 'Samara', 'Cape Town', 'San Francisco']


class AskUserToChoose(BaseTool):
    name = "ask user to choose or approve ticket"
    description = """
Use this tool to ask user to choose flight from several variants or to approve your choice.
Provide all the information about flights.
Be sure syntax is correct.
Provide your question as 'question' (string) and flights data as 'flights' (list of jsons).
BE SURE THE STRUCTURE IS CORRECT.
Some examples:
1. if there is a criteria of chooice (for exmaple user prefer cheapest) -> provide a cheapest flight and ask if user likes it or he/she wants to see more variants
2. if there is NO criteria of chooice> ->  provide all tickets and ask user to choothee or to provide the criteria
"""
    def _run(self, question, flights):
        info = f'\n{question}'
        for i, flight in enumerate(flights):
            info += f'\n{i+1}. {flight}'
        return input(f"\nAgent's question: {info}\nYour answer:\n")
    
    
class AskUserForInfo(BaseTool):
    name = "ask user for info"
    description = """
Use this tool to ask user a question about his preferences.
Use it only if you don't have enought information.
It requires a string input with your question and all the necessary information. 
If you have several questions use the tool several times. DO NOT ASK SEVERAL QUESTIONS IN ONE MESSAGE!!!
Be sure syntax is correct.
Example of usage:
{"thought": "I need to gather more information about the user's preferences to find the best flight options.",
 "action": "ask user for info",
 "action_input": "Do you have a preferred ticket class (economy or business)?"}
"""
    def _run(self, question):
        return input(f"\nAgent's question:\n{question}\nYour answer:\n")


def get_flights(
    filter__departure_datetime_ge: Union[pd.Timestamp, None] = None,
    filter__departure_datetime_le: Union[pd.Timestamp, None] = None,
    filter__departure_weekday: Union[int, None] = None,
    filter__departure_city: Union[str, None] = None,
    filter__arrival_datetime_ge: Union[pd.Timestamp, None] = None,
    filter__arrival_datetime_le: Union[pd.Timestamp, None] = None,
    filter__arrival_weekday: Union[int, None] = None,
    filter__arrival_city: Union[str, None] = None,
    filter__stops_le: Union[int, None] = None,
    filter__duration_le: Union[int, None] = None,
    filter__ticket_class: Union[str, None] = None,
    filter__ticket_price_le: Union[float, None] = None,
    sort_by: Union[str, None] = None,
    # ascending: Union[, None] = True
) -> List[dict]:
    
    df = pd.read_csv(os.path.join(os.getcwd(), 'data/airplane_schedule.csv'))

    if filter__departure_datetime_ge is not None:
        df = df[df['departure_datetime'] >= filter__departure_datetime_ge]
    if filter__departure_datetime_le is not None:
        df = df[df['departure_datetime'] <= filter__departure_datetime_le]
    if filter__departure_weekday is not None:
        df = df[df['departure_weekday'] == filter__departure_weekday]
    if filter__departure_city is not None:
        df = df[df['departure_city'] == filter__departure_city]
    if filter__arrival_datetime_ge is not None:
        df = df[df['arrival_datetime'] >= filter__arrival_datetime_ge]
    if filter__arrival_datetime_le is not None:
        df = df[df['arrival_datetime'] <= filter__arrival_datetime_le]
    if filter__arrival_weekday is not None:
        df = df[df['arrival_weekday'] == filter__arrival_weekday]
    if filter__arrival_city is not None:
        df = df[df['arrival_city'] == filter__arrival_city]
    if filter__stops_le is not None:
        df = df[df['stops'] <= filter__stops_le]
    if filter__duration_le is not None:
        df = df[df['duration'] <= filter__duration_le]
    if filter__ticket_class is not None:
        df = df[df['ticket_class'] == filter__ticket_class]
    if filter__ticket_price_le is not None:
        df = df[df['ticket_price'] <= filter__ticket_price_le]
    if sort_by is not None:
        df = df.sort_values(by=sort_by,  ascending=True)

    return df.to_dict('records')


class GetFlights(BaseTool):
    name = "get flights"
    description = """
Use this tool to get flights information.
It requires folowing arguments to filter data:
filter__departure_datetime_ge: Union[pd.Timestamp, None] (for example '2024-01-01 00:00:00'),
filter__departure_datetime_le: Union[pd.Timestamp, None] (for example '2024-01-01 23:59:99'),
filter__departure_weekday: Union[int, None] (for example 'Monday'),
filter__departure_city: Union[str, None] (for example 'Moscow'),
filter__arrival_datetime_ge: Union[pd.Timestamp, None] (for example '2024-01-01 00:00:00'),
filter__arrival_datetime_le: Union[pd.Timestamp, None] (for example '2024-01-01 23:59:99'),
filter__arrival_weekday: Union[int, None] (for example 'Monday'),
filter__arrival_city: Union[str, None] (for example 'Moscow'),
filter__stops_le: Union[int, None] (for example 1),
filter__duration_le: Union[int, None] (for example 10),
filter__ticket_class: Union[str, None] ('economy' or 'business'),
filter__ticket_price_le: Union[float, None] (for example 1000),
You can also order output (by one of fields: 'departure_datetime', 'departure_weekday',
   'departure_city', 'arrival_datetime', 'arrival_weekday', 'arrival_city',
   'stops', 'duration', 'ticket_class', 'ticket_price') using following arguments:
sort_by: Union[str, None] 
Be sure that you filter only by necessary fields according to user's request.
If the result is an empty list, check your input and try again. If the input is absolutely correct, inform the user that the desired flights were not found.
if you filter by departure_datetime or by arrival_datetime be sure you are using both _ge and _le filters.
Tool requires a json input.
Be sure syntax is correct.
"""
    def _run(self, **kwargs):
        try:
            return str(get_flights(**kwargs))
        except Exception as e:
            exception_name = type(e).__name__
            return f"Exception name: {exception_name}"   



class BuyTicket(BaseTool):
    name = "buy ticket"
    description = """
Use this tool to buy a ticket.
USE IT ONLY WHEN USER APPROVED THE FLIGHT.
Provide json with flight information as input.
Tool will return it back if the purchase is completed successfully.
Be sure syntax is correct."
"""
    def _run(self, **ticket):
        return ticket
