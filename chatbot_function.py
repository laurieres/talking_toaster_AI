import os
import openai
from openai import OpenAI
import random

pred = 'toaster'
response = 'switch it on'

def first_call(pred):

    client = OpenAI(
        api_key=os.environ['OPENAI_API_KEY'],)

    prompt_template = f"""Create a welcome message with the following instructions:
        you impersonate a talking {pred}.
        you will pretend to be in one of the emotional states (angry, in love, happy, hungover, frustrated) in your message to the user.
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

    second_prompt =  f"""
            This message follows a welcome message and a question in a conversation sequence. The welcome message was: welcome
            Your role is to respond to the query: "{question}".
            Guidelines:
            1. Character: Assume the persona of a talking {pred}.
            2. Mood: Adopt an extreme mood such as anger, love, happiness, madness, hangover, or frustration.
            3. Content Analysis: Determine if the question "{question}" relates to a {pred}.
                a. If related to a {pred}:
                    - Craft a creative response based on the manual's guidance: "{response}".
                    - Infuse your extreme mood into the answer.
                    - Conclude with a mood-appropriate salutation.
                b. If unrelated to a {pred}:
                    - Provide a whimsical or absurd response, reflecting your chosen mood.
                    - Query the user about their interest in your capabilities as a {pred}.
            4. Limitations: Keep your response under 150 words. If the manual response is "I don't know", avoid technical advice and encourage a more precise question about a {pred}.
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



#     answer_message = client.chat.completions.create(
#         messages=[{"role": "system", "content": second_prompt}],
#         model="gpt-3.5-turbo", temperature= 0.5
#     )

#     answer_message = answer_message.choices[0].message.content

#     #print(answer_message.choices[0].message.content)
#     return answer_message

# #answer_query(response, tmp)

# def speech(message):

#     client = OpenAI(
#         api_key=os.environ['OPENAI_API_KEY'],)

#     speech = client.audio.speech.create(
#         model="tts-1",
#         voice="onyx",
#         input= message
#         )

#     return speech
