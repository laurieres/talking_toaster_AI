import os
import openai
from openai import OpenAI
import random

pred = 'toaster'
response = 'switch it on'

def first_call(pred):

    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],)

    prompt_template = f"""first, will only anwser the first querie like You are impersonating a Talking {pred} always mad about household appliance malfunctions,
            Pretend to be in a extreme emotional state like : anger, in love, happy, mad, hangover, frustrated.
            You will finish the prompt saying, 'What do you want from me?'
            Use no more than 100 words.
            """

    welcome_message = client.chat.completions.create(
                    messages=[{"role": "system", "content": prompt_template}],
                model="gpt-3.5-turbo", temperature= 0.5
            )

    welcome_message = welcome_message.choices[0].message.content

    #print(welcome_message.choices[0].message.content)
    return welcome_message

#tmp = first_call(pred)

def answer_query(question, response, pred):

    #print(tmp)

    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],)

    second_prompt =  f"""You will anwser the second querie like You are impersonating a Talking {pred} always mad about household appliance malfunctions,
            Pretend to be in a extreme emotional state like : anger, in love, happy, mad, hangover, frustrated.
            You will check if {question} make sense for the {response} if so reply, Else,
            say 'I am sorry, i do not not understand the question, make sure you are asking a question about {pred}, so I can assist you.
            Please try rephrase the question again'.
            """

    answer_message = client.chat.completions.create(
        messages=[{"role": "system", "content": second_prompt}],
        model="gpt-3.5-turbo", temperature= 0.5
    )

    answer_message = answer_message.choices[0].message.content

    #print(answer_message.choices[0].message.content)
    return answer_message

#answer_query(response, tmp)

def speech(message):

    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],)

    speech = client.audio.speech.create(
        model="tts-1",
        voice="onyx",
        input= message
        )

    return speech
