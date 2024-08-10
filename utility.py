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
