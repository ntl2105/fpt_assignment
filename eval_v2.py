# For evaluation, I need to load the ground truth, load the generated text. 
# Process ground truth 
# Process generated text 

# Conduct evaluation 
# Questions to answer: 
# 1) Formatting, how often did the model return the correct json format 
# 2) Relevancy of returned answers: 
# how often was each of the expected field returned? a % for each attribute 
# did the output contained information not asked for?
# 3) Evaluation of required info: accuracy, precision, recall, f1 for each of the attributes 

import pandas as pd
import json
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
from collections import Counter

from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from utility import file_name_dict, create_directory, accepted_alternative_names_mappings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from collections import defaultdict

def load_data(path_to_test, path_to_result):
    # Load the ground truth and predicted data
    gt_df = pd.read_json(path_to_test, lines=True)
    pred_df = pd.read_csv(path_to_result)

    # Expanding the 'attributes' column into separate columns
    messages_df = gt_df['messages'].apply(pd.Series)
    messages_df.columns = ['system', 'user', 'assistant']
    messages_df['gt_text'] = messages_df['assistant'].apply(lambda x: x['content'])

    eval_df = pd.concat([messages_df, pred_df],axis=1)
    eval_df = eval_df.rename(columns={'text': 'gen_text'})

    return eval_df

def fix_and_separate_jsons(malformed_json):

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

def evaluate_json_format(eval_df, result_directory):    
    # Calculate percentages for each type of JSON status
    summary = eval_df['json_status'].value_counts(normalize=True)
    pd.DataFrame(summary).to_csv(f'{result_directory}/format_accuracy.csv')
    return summary

# def evaluate_fields_presence(eval_df, field_name_mappings, result_directory):
#     # Use fixed JSON for presence checking
#     def all_required_present(fixed_json):
#         presence_dict = {}
#         for field, alternatives in field_name_mappings.items():
#             presence_dict[field] = any(alt in key for key in fixed_json.keys() for alt in alternatives)
        
#         return presence_dict

#     # Apply the function using fixed_json
#     fields_present_df = pd.DataFrame(eval_df['fixed_json'].apply(all_required_present).tolist())
#     field_presence_percentages = fields_present_df.mean() * 100
#     return field_presence_percentages

def standardize_field_names(data, field_name_mappings):
    """
    Standardizes field names in the dictionary based on the presence of keywords in the field_name_mappings.
    """
    standardized_data = {}
    for key, value in data.items():
        standardized_key = next((standard_field for standard_field, alternatives in field_name_mappings.items() if any(alt in key for alt in alternatives)), key)
        standardized_data[standardized_key] = value
    return standardized_data

def evaluate_metrics(eval_df, field_name_mappings, results_directory):
    metrics = {}

    for field in field_name_mappings:
        gt_values = eval_df['gt_text'].apply(lambda x: str.lower(str(x.get(field, 'Field Not Found'))))
        pred_values = eval_df['standardized_json'].apply(lambda x: str.lower(str(x.get(field, 'Field Not Found'))))

        accuracy = accuracy_score(gt_values, pred_values)
        precision, recall, f1, _ = precision_recall_fscore_support(gt_values, pred_values, average='macro', zero_division=0)

        metrics[field] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    pd.DataFrame(metrics).to_csv(f"{results_directory}/metrics.csv")   
    return metrics

def populate_label_mappings(eval_df, fields, top_n=6):
    """
    Dynamically populate label mappings by analyzing the most common labels in a dataset.

    :param eval_df: DataFrame containing the data.
    :param fields: Dictionary where keys are fields to analyze and values are lists to store top labels.
    :param top_n: Number of top most common labels to retain.
    :return: Updated dictionary with the top common labels for each field.
    """
    label_mappings = {field: [] for field in fields.keys()}
    
    for field in fields:
        # Collect all labels for the field from the dataset
        all_labels = eval_df['gt_text'].apply(lambda x: str.lower(str(x.get(field, 'None'))))
        # Filter out 'None' or other undesirable values
        filtered_labels = [label for label in all_labels if label != 'None']
        # Get the most common labels
        most_common_labels = [label for label, _ in Counter(filtered_labels).most_common(top_n)]
        label_mappings[field] = most_common_labels
    
    return label_mappings

def evaluate_detailed_metrics(eval_df, field_labels_mappings, results_directory):
    """Evaluates precision, recall, f1 scores, and confusion matrices for specific labels, and saves plots."""
    metrics = {}

    for field, top_values in field_labels_mappings.items():
        
        gt_values = eval_df['gt_text'].apply(lambda x: str.lower(str(x.get(field, 'Field Not Found'))))
        pred_values = eval_df['standardized_json'].apply(lambda x: str.lower(str(x.get(field, 'Field Not Found'))))

        # we only want to filter to where gt is one of the top 6 most common values for each field 
        gt_values_filtered = []
        pred_values_filtered = []
        for i, val in enumerate(gt_values):
            if val in top_values: 
                gt_values_filtered.append(val)
                pred_values_filtered.append(pred_values[i])

        precision, recall, f1, _ = precision_recall_fscore_support(gt_values, pred_values, labels=list(top_values), average=None, zero_division=0)
        conf_matrix = confusion_matrix(gt_values, pred_values, normalize='true', labels=list(top_values))


        # Store results
        metrics[field] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            # 'confusion_matrix': conf_matrix
        }

        # Plot and save confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='.2%', cmap='Blues', xticklabels=top_values, yticklabels=top_values)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix for {field}')
        
        sanitized_field = field.replace(' ', '_').replace("/","_").lower()
        plot_path = f"{results_directory}/confusion_matrix_{sanitized_field}.png"
        
        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_df.columns = top_values
        conf_matrix_df.index = top_values

        conf_matrix_df.to_csv(f"{results_directory}/confusion_matrix_{sanitized_field}.csv")

        plt.savefig(plot_path)
        plt.close() 

    pd.DataFrame(metrics).to_csv(f"{results_directory}/metrics_{sanitized_field}.csv")
    return metrics


def main():

    directory = 'fine_tuning_data_v2_detailedInstructions_gpt3format'
    path_to_test = f"{directory}/test.jsonl"
    # path_to_result = f"{directory}/results_3point5_ft.csv"
    # path_to_result = f"{directory}/results_3point5_og.csv"

    for model_store_path in file_name_dict.values():
        generated_text_directory = model_store_path['directory']
        generated_text_file_name = model_store_path['file_name']

        # path_to_result = 'fine_tuning_data_v2_detailedInstructions/davinci_epoch3_result.csv'
        # path_to_result = 'fine_tuning_data_v2_detailedInstructions/davinci_epoch3_result.csv'
        # path_to_result = 'fine_tuning_data_v2_detailedInstructions/babbage-002_epoch4_result.csv'

        path_to_result = f"{generated_text_directory}/{generated_text_file_name}"
        
        if 'detail' in generated_text_directory:
            results_directory_per_model = f"eval_results/{generated_text_file_name.split('.')[0]}_detailedInstr"#removing the .csv ending
        else:
            results_directory_per_model = f"eval_results/{generated_text_file_name.split('.')[0]}"#removing the .csv ending
        create_directory(results_directory_per_model) 

        eval_df = load_data(
            path_to_test=path_to_test, 
            path_to_result=path_to_result
            )
        
        # Apply the fix_and_separate_jsons function to each 'gen_text'
        results = eval_df['gen_text'].apply(lambda x: fix_and_separate_jsons(x))
        
        # Unpack results into two separate lists
        fixed_jsons, statuses = zip(*results)
        
        eval_df['json_status'] = statuses
        eval_df['fixed_json'] = fixed_jsons

        # Ensure the JSON strings are correctly converted to dictionaries
        eval_df['gt_text'] = eval_df['gt_text'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

        # Standardizing field names 
        eval_df['standardized_json'] = eval_df['fixed_json'].apply(lambda x: standardize_field_names(x, accepted_alternative_names_mappings) if x else {})
        
        eval_df.to_csv(f'{results_directory_per_model}/eval_df.csv')

        # Evaluate field presence and unwanted fields
        json_summary = evaluate_json_format(eval_df, results_directory_per_model)
        print(f"Format Accuracy: {json_summary}")

        # fields_accuracy = evaluate_fields_presence(eval_df, accepted_alternative_names_mappings, results_directory_per_model)
        # print(f"Field Accuracy: {fields_accuracy}")
        
        metrics = evaluate_metrics(eval_df, accepted_alternative_names_mappings, results_directory_per_model)

        field_labels_mappings = populate_label_mappings(eval_df=eval_df, 
                                                        fields=accepted_alternative_names_mappings, 
                                                        top_n=5)
        

        detailed_metrics = evaluate_detailed_metrics(eval_df, field_labels_mappings, results_directory_per_model)
        print("DETAILED EVALUATION METRICS: \n", detailed_metrics)

        for field, scores in metrics.items():
            print("OVERALL METRICS")
            print(f"{field}:")
            print(f"  Accuracy: {scores['accuracy']:.2%}")
            print(f"  Precision: {scores['precision']:.2%}")
            print(f"  Recall: {scores['recall']:.2%}")
            print(f"  F1 Score: {scores['f1_score']:.2%}")
        

if __name__ == "__main__":
    main()



