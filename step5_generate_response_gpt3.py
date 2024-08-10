from openai import OpenAI
import json
import pandas as pd
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def remove_assistant_entries(data):
    """
    Remove entries with the role 'assistant' from the data.
    """
    return [message for message in data['messages'] if message['role'] != 'assistant']


def prepare_api_payload(data):
    """
    Prepare the payload for the API, ensuring each message is an object.
    """
    cleaned_messages = []
    for message in data:
        if isinstance(message, str):
            logging.info('Original message is a string, attempting to parse')
            # Attempt to convert string back to dictionary if it's accidentally stringified
            try:
                message = json.loads(message)
            except json.JSONDecodeError as e:
                logging.error('JSON decoding failed', exc_info=True)
                continue
        cleaned_messages.append(message)
        
    return cleaned_messages

def call_model(client, model, formatted_user_data, max_tokens=150, temperature=0):
    """
    Call the appropriate model based on the model string.
    """
    if 'babbage-002' in model or 'davinci-002' in model: 
        try:
            response = client.completions.create(
            # model="ft:babbage-002:personal::9tbdQSFF",
                model=model,
                prompt=formatted_user_data,
                max_tokens=max_tokens,
                temperature=temperature
            )

            #only collecting the text part for now
            return response.choices[0].text   
        except Exception as e:
            print('Error getting response with',model,formatted_user_data, e)
    
    elif 'gpt-3.5' in model: 
        try:
            response = client.chat.completions.create(
                model=model,
                messages=formatted_user_data,
                max_tokens=max_tokens,
                temperature=temperature,
                )
            
            #only collecting the text part for now
            return response.choices[0].message.content
        except Exception as e:
            print('Error getting response with',model,formatted_user_data, e)
    
    else: 
        print('Unsupported Model')

def main():
    # Set the OPENAI_API_KEY environment variable
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    client = OpenAI(api_key=api_key)
    new_directory_name = 'fine_tuning_data_v2_detailedInstructions_gpt3format'
    # model = 'ft:gpt-3.5-turbo-0125:personal::9tg6PAnc' #3.5 turbo on detailed instructions
    model = 'gpt-3.5-turbo-0125'
    # outfile_name = 'results_3point5_ft.csv'
    outfile_name = 'results_3point5_og.csv'


    with open(f"{new_directory_name}/test.jsonl", 'r') as file:
        test_data = [json.loads(line.strip()) for line in file.readlines()]

    results = []    
    for idx, entry in enumerate(test_data): 
        data_for_testing = remove_assistant_entries(entry)
        user_input_data = prepare_api_payload(data_for_testing)        
        
        text = call_model(client=client, model=model, 
                            formatted_user_data=user_input_data)
            
        results.append({'index': idx, 'text': text, 'model': model})
    # Create a DataFrame and save to CSV
    df = pd.DataFrame(results)
    df.to_csv(f"{new_directory_name}/{outfile_name}", index=False)

    logging.info(f"Results have been saved to {outfile_name}")

if __name__ == "__main__":
    main()
