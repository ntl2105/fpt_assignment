from openai import OpenAI
import json
import pandas as pd
import os
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
            logging.info('Error getting response with',model,formatted_user_data, e)
    
    else: 
        logging.info('Unsupported Model')

def main():
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)
    
    # new_directory_name = 'fine_tuning_data_v2'
    # model = 'ft:babbage-002:personal::9tf9RTDu' #n_epoch = 4
    # model = 'ft:babbage-002:personal::9tg6PCpn' #n_epoch = 3
    # model = 'ft:babbage-002:personal::9tf9VhUQ'

    models_list = [
        'ft:babbage-002:personal::9tf9RTDu', 
        'ft:babbage-002:personal::9tg6PCpn', 
        'ft:babbage-002:personal::9tf9VhUQ', 
        'ft:babbage-002:personal::9tg8SIK0', 
        # 'ft:davinci-002:personal::9tgOuHsg', 
        #'ft:gpt-3.5-turbo-0125:personal::9tg6PAnc'
      ]   

    file_name_dict = {
        'ft:babbage-002:personal::9tf9RTDu': {
            'directory': 'fine_tuning_data_v2',
            'file_name': 'babbage-002_epoch4_result.csv'
        },
        'ft:babbage-002:personal::9tg6PCpn': {
            'directory': 'fine_tuning_data_v2',
            'file_name': 'babbage-002_epoch3_result.csv'
        },
        'ft:babbage-002:personal::9tf9VhUQ': {
            'directory': 'fine_tuning_data_v2_detailedInstructions',
            'file_name': 'babbage-002_epoch4_result.csv'
        },
        'ft:babbage-002:personal::9tg8SIK0': {
            'directory': 'fine_tuning_data_v2_detailedInstructions',
            'file_name': 'babbage-002_epoch3_result.csv'
        },
        'ft:davinci-002:personal::9tgOuHsg':{
            'directory': 'fine_tuning_data_v2_detailedInstructions',
            'file_name': 'davinci_epoch3_result.csv'
        },
        'ft:gpt-3.5-turbo-0125:personal::9tg6PAnc': {
            'directory': 'fine_tuning_data_v2_detailedInstructions',
            'file_name': 'gpt3point5_epoch3_result.csv'
        }
    }
    
    # new_directory_name = 'fine_tuning_data_v2_detailedInstructions'
    # model = 'ft:davinci-002:personal::9tgOuHsg'

    for model in models_list:
        directory_name = file_name_dict[model]['directory']
        file_name = file_name_dict[model]['file_name']

        with open(f"{directory_name}/test.jsonl", 'r') as file:
            test_data = [json.loads(line.strip()) for line in file.readlines()]

        results = []    
        for idx, entry in enumerate(test_data): 
            
            user_input_data = entry['prompt']
            text = call_model(client=client, model=model, 
                                formatted_user_data=user_input_data)    
            results.append({'index': idx, 'text': text, 'model': model})
        
        #Create a DataFrame and save to CSV
        df = pd.DataFrame(results)
        results_file_path = f"{directory_name}/{file_name}"
        df.to_csv(results_file_path, index=False)

        logging.info(f"Results have been saved to {results_file_path}")

if __name__ == "__main__":
    main()
