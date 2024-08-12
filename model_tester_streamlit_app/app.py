import streamlit as st
from openai import OpenAI, api_key
from dotenv import load_dotenv
import os
import json 
from utility import fix_and_separate_jsons, model_selection_to_model_name, prompt_instructions_standard, prompt_instructions_detailed

# Define the function to format user data
def format_user_input(user_input, model_name):
    
    if "gpt-3.5" in model_name: 
        formatted_data = [
            {"role": "system", "content": """Consider the Airbnb listing description provided below. 
            Your task involves a detailed extraction of specific attributes that are crucial for 
            understanding the property's characteristics and appeal. 
            Proceed as follows: 1. **Number of Bedrooms**: Identify and report the exact number of bedrooms mentioned. 
            If the description implies a single bedroom area, such as in a studio, explicitly note it as '1'. 
            If no specific number is mentioned, state 'Not specified'. 
            2. **Type of Property**: Determine the type of property based on descriptions such as 'studio', 'apartment', 'house', 'loft', etc. 
            Provide the exact type as mentioned in the description. If the property type is not directly stated, use your judgment based on the description provided and categorize it as 'Not specified' if uncertain. 
            3. **Shared Space Indicator**: Assess whether any part of the property is shared with other guests or residents. This includes bathrooms, kitchens, living areas, or any mention of communal spaces. Return 'TRUE' if shared spaces are mentioned, and 'FALSE' if the listing indicates private use of all facilities or if there is no mention of shared facilities. 
            4. **Overall Vibe or Atmosphere**: Classify the atmosphere of the property based on the descriptors used in the listing. Use categories such as: 
            - 'MODERN': Mention of contemporary design elements, modern furniture, or state-of-the-art facilities. 
            - 'CHIC': Descriptions include terms like stylish, fashionable, or elegant. 
            - 'ARTSY': The presence of artistic decor, vibrant colors, or a focus on creative environments.
            - 'HISTORIC': Properties that retain historical architecture or are situated in historically significant neighborhoods. 
            - 'COMFORTABLE': Listings that emphasize comfort, coziness, or a relaxing environment. 
            - 'PLAIN': Simple, minimalistic, or basic amenities without any specific decorative mentions. 
            If the vibe is not clearly defined or if the description lacks enough information for a definite classification, mark it as 'Not specified'.
            Please structure your findings in a JSON format to maintain clarity and ease of further processing. Here is the description to analyze: """},
            {"role": "user", "content": user_input}
        ]
    
    elif 'detail' in model_name:
        formatted_data = [
            f"""
                {prompt_instructions_detailed} {user_input}
            """
        ] 
    else: 
        formatted_data = [
            f"""
                {prompt_instructions_standard} {user_input}
            """
        ] 

    return formatted_data

# Define the function to call the OpenAI API
def get_generated_text(formatted_user_data, model='ft:gpt-3.5-turbo-0125:personal::9tg6PAnc', max_tokens=150, temperature=0):
    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    if 'gpt-3.5' in model: 
        response = client.chat.completions.create(
                    model=model,
                    messages=formatted_user_data,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    )
                
                #only collecting the text part for now
        return response.choices[0].message.content
    else:
        response = client.completions.create(
            # model="ft:babbage-002:personal::9tbdQSFF",
                model=model,
                prompt=formatted_user_data,
                max_tokens=max_tokens,
                temperature=temperature
            )
        return response.choices[0].text   

# Streamlit app layout
st.title("Airbnb Listing Extractor")

# Define your text options
text_options = {
    "Option 1": """Hello! Welcome to my 1-bedroom apartment in downtown Manhattan. You'll be using the guest mattress in the living room.""",
    "Option 2": """This exquisitely furnished 2 bedroom apartment in a completely renovated brownstone is 
    located on a quiet tree lined street in a much sought after area of Bedford Stuyvesant.""",
    "Option 3": """Beautiful Greenwich Village fully furnished One bedroom / One bath apartment in landmark building available for short term (one month minimum) or long term lease. You have entire apartment for your use. Sunny southern exposure. Very safe neighborhood. Police Station on same block. Update….our building is having some facade work. Would you be ok with that? I can offer a discount because of this. Please ask.
    """
}

# Custom CSS to make the button red
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: rgb(255, 0, 0);
        color: white;
    }
    /* Adjusting the font size of sidebar labels */
    .sidebar .stRadio > label {
        font-size: 25px;
    }
    /* Adjusting the font size of main area text area and label */
    .stTextArea > label, .stTextInput > label {
        font-size: 25px;
    }
    /* Increasing the text area font size */
    textarea {
        font-size: 18px !important;
    }
    </style>""", unsafe_allow_html=True)

# Dropdown for model selection, using keys from the model_rename_map
model_options = list(model_selection_to_model_name.values())
model_keys = list(model_selection_to_model_name.keys())
default_model_index = model_keys.index('ft:gpt-3.5-turbo-0125:personal::9tg6PAnc')
user_model_selection = st.selectbox(
    "Choose a model:", 
    options=model_options, 
    index=default_model_index,)

# Mapping back the selected model to its key
selected_model_key = model_keys[model_options.index(user_model_selection)]

# Sidebar for text selection
option = st.sidebar.radio(
    "Choose a sample text (or write your own!):", 
    list(text_options.keys()),
 )

# Main content area - text area for user input
user_input = st.text_area(
    "Enter your listing text here:", 
    value=text_options[option], 
    height=180
    )

if st.button("Extract Key Information"):
    if user_input:
        formatted_data = format_user_input(user_input=user_input,
                                           model_name=selected_model_key)
        generated_text = get_generated_text(formatted_user_data=formatted_data, 
                                            model=selected_model_key
                                            )
        st.write("Here it is:", height=300)
        st.code(generated_text, language='text')  # Display the generated text in code style

        fixed_json, status = fix_and_separate_jsons(generated_text)
        if status == 'Perfect':
            st.write("The model output is in valid JSON formatting ✅")
            st.json(fixed_json)  # Display fixed JSON using Streamlit's JSON renderer
        elif status == 'Fixable': 
            st.write("The model output is not in valid JSON formatting, but it was fixed as follows:")
            st.json(fixed_json)  # Display fixed JSON using Streamlit's JSON renderer
        elif status == 'Unfixable': 
            st.write("The model output is not in valid JSON formatting, and we're currently unable to fix it.")
    else:
        st.write("Please enter some text to generate a response.")