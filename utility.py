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
        'directory': 'fine_tuning_data_v2_detailedInstructions_gpt3format',
        'file_name': 'results_3point5_ft.csv'
    },
    'gpt-3.5-turbo-0125': {
        'directory': 'fine_tuning_data_v2_detailedInstructions_gpt3format',
        'file_name': 'results_3point5_og.csv'
    }
}

result_file_to_model_rename_map = {
    'babbage-002_epoch3_result_detailedInstr': "Babbage-002, Epoch 3, DetailedInstr",
    'babbage-002_epoch3_result': "Babbage-002, Epoch 3, Standard",
    'babbage-002_epoch4_result_detailedInstr': "Babbage-002, Epoch 4, DetailedInstr",
    'babbage-002_epoch4_result': "Babbage-002, Epoch 4, Standard",
    'davinci_epoch3_result_detailedInstr': "Davinci, Epoch 3, DetailedInstr",
    'results_3point5_ft_detailedInstr': "GPT-3.5, Fine-Tuned, DetailedInstr",
    'results_3point5_og_detailedInstr': "GPT-3.5, Original, DetailedInstr"
}

model_selection_to_model_name = {
    'ft:babbage-002:personal::9tf9RTDu':  "Babbage-002, Epoch 4, Standard",
    # {
    #     'directory': 'fine_tuning_data_v2',
    #     'file_name': 'babbage-002_epoch4_result.csv'
    # },
    'ft:babbage-002:personal::9tg6PCpn': "Babbage-002, Epoch 3, Standard",
    #     'directory': 'fine_tuning_data_v2',
    #     'file_name': 'babbage-002_epoch3_result.csv'
    # },
    'ft:babbage-002:personal::9tf9VhUQ': "Babbage-002, Epoch 4, DetailedInstr",
    # {
    #     'directory': 'fine_tuning_data_v2_detailedInstructions',
    #     'file_name': 'babbage-002_epoch4_result.csv'
    # },
    'ft:babbage-002:personal::9tg8SIK0': "Babbage-002, Epoch 3, DetailedInstr",
    # {
    #     'directory': 'fine_tuning_data_v2_detailedInstructions',
    #     'file_name': 'babbage-002_epoch3_result.csv'
    # },
    'ft:davinci-002:personal::9tgOuHsg': "Davinci, Epoch 3, DetailedInstr",
    # {
    #     'directory': 'fine_tuning_data_v2_detailedInstructions',
    #     'file_name': 'davinci_epoch3_result.csv'
    # },
    'ft:gpt-3.5-turbo-0125:personal::9tg6PAnc': "GPT-3.5, Fine-Tuned, DetailedInstr",
    # {
    #     'directory': 'fine_tuning_data_v2_detailedInstructions_gpt3format',
    #     'file_name': 'results_3point5_ft.csv'
    # },
    'gpt-3.5-turbo-0125': "GPT-3.5, Original, DetailedInstr"
    # {
    #     'directory': 'fine_tuning_data_v2_detailedInstructions_gpt3format',
    #     'file_name': 'results_3point5_og.csv'
    # }
}

prompt_instructions_standard = """
Given the following Airbnb description, Extract the number of bedrooms, determine the type of property, 
            determine whether Is any space shared?, and classify Overall vibes/atmosphere 
            return in JSON format:
"""

prompt_instructions_detailed = """
Consider the Airbnb listing description provided below. Your task involves a detailed extraction of specific attributes that are crucial for understanding the property's characteristics and appeal. Proceed as follows:

1. **Number of Bedrooms**: Identify and report the exact number of bedrooms mentioned. If the description implies a single bedroom area, such as in a studio, explicitly note it as '1'. If no specific number is mentioned, state 'Not specified'.

2. **Type of Property**: Determine the type of property based on descriptions such as 'studio', 'apartment', 'house', 'loft', etc. Provide the exact type as mentioned in the description. If the property type is not directly stated, use your judgment based on the description provided and categorize it as 'Not specified' if uncertain.

3. **Shared Space Indicator**: Assess whether any part of the property is shared with other guests or residents. This includes bathrooms, kitchens, living areas, or any mention of communal spaces. Return 'TRUE' if shared spaces are mentioned, and 'FALSE' if the listing indicates private use of all facilities or if there is no mention of shared facilities.

4. **Overall Vibe or Atmosphere**: Classify the atmosphere of the property based on the descriptors used in the listing. Use categories such as:
   - 'MODERN': Mention of contemporary design elements, modern furniture, or state-of-the-art facilities.
   - 'CHIC': Descriptions include terms like stylish, fashionable, or elegant.
   - 'ARTSY': The presence of artistic decor, vibrant colors, or a focus on creative environments.
   - 'HISTORIC': Properties that retain historical architecture or are situated in historically significant neighborhoods.
   - 'COMFORTABLE': Listings that emphasize comfort, coziness, or a relaxing environment.
   - 'PLAIN': Simple, minimalistic, or basic amenities without any specific decorative mentions.
If the vibe is not clearly defined or if the description lacks enough information for a definite classification, mark it as 'Not specified'.

Please structure your findings in a JSON format to maintain clarity and ease of further processing. Here is the description to analyze:
"""

def fix_and_separate_jsons(malformed_json):
    import json 
    """
    Takes in the raw text output from the LLM, 
    1) loads it in json format, if no error, return json and status "Perfect"
    2) if fail, tries to fix it by identifying and isolating only valid json format, ie. open and end bracket,
    and dropping the rest, loads again, if it works, return json, and status "Fixable" 
    3) if still getting error, returns None and status "Unfixable" 
    """
    try: 
        obj = json.loads(malformed_json)
        return (obj, "Perfect")
    except Exception as e: 
    
        brace_level = 0
        current_object = ''
        
        for char in malformed_json:
            if char == '{':
                brace_level += 1
            if brace_level > 0:
                current_object += char
            if char == '}':
                brace_level -= 1
            
            # When brace level returns to 0, try to parse the JSON object
            if brace_level == 0 and current_object:
                try:
                    obj = json.loads(current_object)
                    return (obj, "Fixable")
                except json.JSONDecodeError:
                    # Attempt to repair by properly closing the JSON
                    try:
                        repaired_object = current_object + '}'
                        obj = json.loads(repaired_object)
                        return (obj, "Fixable")
                    except json.JSONDecodeError:
                        current_object = ''  # Reset the current object if repair fails
        return (None, "Unfixable")   # If no valid JSON object was parsed

accepted_alternative_names_mappings = {
        'Number of Bedrooms': {'Bedrooms', 'Number of Beds', 'Bedroom Count'},
        'Type of Property': {'Property', 'Housing Type'},
        'Is any space shared?': {'space', 'Space', 'Shared'},
        'Overall vibes/atmosphere': {'Vibe', 'Atmosphere', 'vibe', 'atmosphere'}  
        }

def create_directory(directory_path):
    import os 
    try:
        # The exist_ok parameter, when set to True, allows the directory to exist without raising an error.
        os.makedirs(directory_path, exist_ok=True)
        print(f"Directory '{directory_path}' created successfully or already exists.")
    except Exception as e:
        # Handle any exceptions that might occur
        print(f"Failed to create directory '{directory_path}'. Error: {e}")
