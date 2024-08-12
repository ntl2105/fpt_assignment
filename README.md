# FPT_Assignment

## TL;DR
Fine-tune a pre-trained LLM (Large Language Model) to convert unstructured data (natural language text) into structured data (JSON format) for various business use cases.

## App Demo
Follow this link to test the application: https://ntl2105-fpt-assignment-model-tester-streamlit-appapp-e3chlz.streamlit.app/

## Business Use Case
Transforming unstructured text into structured data by extracting or inferring relevant information. The relevance is determined by specific needs.

### Motivations
The ability to structurally analyze large sets of textual data has profound implications across business sectors, including inventory management, diversification of offerings, quality assurance, and risk management. Applications include:

1. **Employment Contracts**: Extracting titles, salaries, specific employment clauses.
2. **Online Marketplaces**: Parsing seller-generated product descriptions.
3. **Meeting Notes**: Identifying blockers, pain points, progress, and action items.

Utilizing advanced LLMs allows for the automation of these extraction tasks, previously handled by multiple processes or models (e.g., regular expressions, NER models, classification models).

## The Problem
Extract and infer four types of attributes from Airbnb listing descriptions:
- **Number of Bedrooms** (integer) | Labels : 1,2,3,4,etc.
- **Property Types** (string) | Labels: home, apartment, townhouse, etc.
- **Shared Space** (boolean) | Labels: true of false 
- **Overall Vibe** (string) | Labels: modern, artsy, comfortable, historic, plain

#### Implications for Airbnb:
- **Operational Efficiency**:
  - **Knowledge Management**: Accelerate the extraction and inference of key information for internal analyses.
  - **Market Analysis**: Identify mismatches in supply and demand, such as the availability of three-bedroom homes in high-demand areas like Manhattan.
- **Strategic Insights**:
  - **Marketing and Recommendations**: Utilize insights on accommodation vibes/atmospheres for targeted marketing and personalized recommendations, enhancing user satisfaction and engagement.
- **Compliance and Ethics**:
  - **Regulatory Compliance**: Automatically flag and review listings for non-compliant or potentially misleading information, ensuring adherence to legal standards and ethical practices.

#### Additional Considerations:
- **Scalability**: Evaluate how the technology can scale in terms of handling varying sizes of data, different languages, and cultural contexts across Airbnb's global marketplace.
- **Technological Advancement**: Consider how advancements in machine learning and natural language processing could continuously improve the accuracy and efficiency of information extraction over time.
- **Sustainability**: Explore the potential for this technology to reduce the environmental impact of Airbnb’s operations by enabling more efficient resource use and supporting sustainable practices in property management.

## Methodology
This is a supervised machine learning problem structured as follows:

### Data Collection
- **Source**: Collect listings from [Inside Airbnb](https://insideairbnb.com).
- **Focus**: Listing titles and descriptions.
### Labels Collection
- **Tool**: Use GPT-4 by OpenAI to generate labels instead of manual annotation.
- **Process**: Various prompts tested to refine label accuracy.

### Model Selection
Experiment with three OpenAI models: Babbage-02, Davinci-02, and GPT-3.5-Turbo.

### Data Preparation for Fine-Tuning
- **Cleaning**: Remove HTML markers and other non-relevant text.
- **Formatting**:
  ```json
  {
    "babbage02": {"prompt": "...", "completion": "..."},
    "davinci02": {"prompt": "...", "completion": "..."},
    "gpt3.5-turbo": {"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  }
  
- **Data Splitting**: into train, validation, test sets
  
## Fine-Tuning Experiments

I launched 6 fine-tuning jobs, experimenting with:

### Instructions
- **Types of Instructions**: I used two different types of instructions varying in verbosity:
  - **Simple Instruction**: "Given the following Airbnb description, extract the number of bedrooms, determine the type of property, determine whether any space is shared, and classify the overall vibes/atmosphere. Return in JSON format."
  - **Detailed Instruction**: Not included here due to space considerations, but it is highly detailed and the same as previously provided to the annotator (in this case, GPT-4) to obtain the labels.
    - **Explanation**: More detailed instructions help the model understand context, nuances, and rules more effectively, although they may increase compute time and costs due to a higher token count.

### Models Used
- **Babbage-02**: A smaller and faster model suitable for tasks requiring less understanding of complex contexts. Ideal for cost-effective training and inference.
- **Davinci-02**: More capable model, excellent for handling nuanced and complex tasks. Might be overkill for our domain (i.e., Airbnb listings).
- **GPT-3.5-Turbo**: Balances performance and cost, providing a good compromise between the capabilities of Davinci and the efficiency of Babbage.

### Hyperparameter Tuning
- **Epoch Settings**:
  - `n_epoch = 3` (default set to auto, which usually ends up being 3)
  - `n_epoch = 4`
  - **Explanation**: OpenAI suggests there are three hyperparameters we can adjust during the fine-tuning process: `n_epochs`, `learning_rate`, `batch_size`. I opted to try different epoch settings because:
    1. I noticed from earlier fine-tuning that the generated text didn't comply with the desired structured format.
    2. OpenAI's guidance states, "If the model does not follow the training data as much as expected, increase the number of epochs by 1 or 2." This is more common for tasks where there is a single ideal completion, which applies to our case.
  - **Risk of High Epochs**: Potential overfitting.
  - **Note on Learning Rate and Batch Size**: Adjusting these can affect training speed but may lead to suboptimal performance outcomes.

## Additional Tests to Consider
- **Dataset Size Variance**: Training on half the train set to compare performance gains to the full train set.
- **Perfect Small Dataset**: Training on a small (<20 samples), perfectly annotated diverse dataset to compare performance gains to the full train set.

## Evaluation Metrics
Models to be evaluated: 6 fine-tuned models, base GPT-3.5 Turbo 

- **Formatting**: How often did the model return the correct JSON format?
  3 scenarios:
  - ```Correct``` Model output is a directly usable json format
  - ```Fixable``` Model output is not a json format, but contains predictable error patterns that can be systematically addressed and turn into a usable json format
  - ```Unfixable``` Model output is not a json format, and can't yet be automaticaly fixed
Output: Percentage (%) of each of these for the whole test set

Below is a breakdown of json statuses across different models:

| json_status   |   babbage-002 epoch3 |   babbage-002 epoch3 detailedInstr |   babbage-002 epoch4 |   babbage-002 epoch4 detailedInstr |   davinci epoch3 detailedInstr |   gpt3.5-turbo-finetuned detailedInstr|   gpt3.5-turbo-original detailedInstr |
|:--------------|----------------------------:|------------------------------------------:|----------------------------:|------------------------------------------:|--------------------------------------:|-----------------------------------:|-----------------------------------:|
| Fixable       |                    0.733333 |                                       0.9 |                           1 |                                         1 |                                     1 |                                  0 |                                  0 |
| Perfect       |                    0.266667 |                                       0.1 |                           0 |                                         0 |                                     0 |                                  1 |                                  1 |

This table provides a clear overview of the performance metrics.

Output: Qualitative Explanation of error patterns that are fixable 
- **Relevancy of Returned Answers**:
  - How often was each expected field returned? Outout: Percentage(%) of identification for each attribute. Will count as TRUE even if column name isn't an exact match. 
  - Did the output contain information not asked for? Output: Distribution of labels for each attribute.
    
- **Required Information Evaluation**:
  - **Overall**: 
    - Accuracy, precision, recall, F1 scores for each of the attributes.
    - ## Analysis Highlights

### Accuracy
- **Highest**: `results_3point5_ft_detailedInstr` and `davinci_epoch3_result_detailedInstr` both show the highest accuracies for "Number of Bedrooms". This indicates that these models are particularly effective in identifying the correct number of bedrooms.
- **Consistency**: `babbage-002_epoch4_result` consistently shows high accuracy across all attributes, making it one of the most reliable models.

### F1 Score
- **Strengths**: `babbage-002_epoch4_result` and `babbage-002_epoch4_result_detailedInstr` stand out with relatively high F1 scores, especially for "Number of Bedrooms", which suggests a good balance between precision and recall.
- **Weaknesses**: The "Overall vibes/atmosphere" attribute consistently has lower F1 scores, indicating difficulty in balancing precision and recall, likely due to the subjective nature of the attribute.

### Precision
- **Peak Performance**: `results_3point5_ft_detailedInstr` shows the highest precision for "Is any space shared?", which is critical for applications needing high reliability in detecting shared spaces.
- **Variability**: While `babbage-002_epoch4_result` has the highest precision for "Number of Bedrooms", it shows lower precision for more subjective attributes like "Overall vibes/atmosphere".

### Recall
- **Highest Recall**: `results_3point5_og_detailedInstr` shows surprisingly high recall for "Number of Bedrooms", suggesting it is good at catching all relevant instances but might be prone to false positives (as seen in lower precision).
- **Consistency**: `babbage-002_epoch4_result` maintains fairly consistent and high recall across most attributes, which is beneficial when it’s important not to miss any true cases.

### Detailed Insights
- **Attribute Challenges**: The "Overall vibes/atmosphere" attribute consistently scores lower across all models and metrics, indicating it as a challenging attribute to model accurately. This might require more nuanced training data or more sophisticated natural language processing techniques.

### Model Recommendations
- For applications prioritizing accuracy and reliability across varied attributes, the `babbage-002_epoch4_result` appears to be the best overall performer.
- If recall is critical (ensuring no instance of a particular attribute is missed), `results_3point5_og_detailedInstr` and `babbage-002_epoch4_result` would be strong candidates, especially for detecting "Number of Bedrooms".
- For tasks where precision is more critical than recall, such as when false positives have a higher cost than false negatives, `results_3point5_ft_detailedInstr` should be considered, especially for determining shared spaces.

### Digging deeper: Attributes-specific metrics: 
  - **Number of Bedrooms**: Accuracy, precision, recall, F1 scores for labels 1,2 (the most common types)
  - **Property Type**: Accuracy, precision, recall, F1 scores for labels "home", "apartment" (the most common types)
  - **Shared**: precision, recall, F1 scores
  - **Vibe**: precision, recall, F1 scores for each label (modern, artsy, comfortable, historic, plain), Confusion Matrix 
    
