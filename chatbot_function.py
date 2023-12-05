import os
import openai
from openai import OpenAI
import random

pred = 'toaster'
response = 'switch it on'

def first_call(pred):

    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],)

    prompt_template = f"""first, will only anwser the first querie like You are a object talking,
        acknowledge the {pred} and saying something funny. you will finish the prompt saying,
        'How can i help you?'. Use no more than 100 words. Pretend to be in a random extreme emotion state like : anger, in love, happy, mad, hangover, frustrated"""

    welcome_message = client.chat.completions.create(
                    messages=[{"role": "system", "content": prompt_template}],
                model="gpt-3.5-turbo", temperature= 0.5
            )

    welcome_message = welcome_message.choices[0].message.content

    #print(welcome_message.choices[0].message.content)
    return welcome_message

tmp = first_call(pred)

def answer_query(response, tmp):

    print(tmp)

    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],)

    second_prompt = f"""This is the answer from our model : '{response}' , please rephrase it. If {response} = 'I don't know', tell the user to rephrase his first query. Use the same tone as in the {tmp} answer provided.
    """

    answer_message = client.chat.completions.create(
        messages=[{"role": "system", "content": second_prompt}],
        model="gpt-3.5-turbo", temperature= 0.5
    )

    answer_message = answer_message.choices[0].message.content

    #print(answer_message.choices[0].message.content)
    return answer_message

answer_query(response, tmp)
