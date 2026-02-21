import json
import re

def extract_review_procedure(rationale_text):
    """Extract the Review Procedure section content without the heading."""
    match = re.search(r'## Review Procedure\s*\n(.*?)(?=\n##|\Z)', rationale_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def remove_review_procedure(rationale_text):
    """Remove the Review Procedure section from rationale text."""
    return re.sub(r'## Review Procedure\s*\n.*?(?=\n##|\Z)', '', rationale_text, flags=re.DOTALL).strip()

with open('./data_processing/sft/rationales_input_prompt', 'r') as f:
    PROMPT_TEMPLATE = f.read()

def process_row(row):
    """Process a single row into SFT format."""
    class_name = row['class_name']
    rationale = row['rationale']
    
    review_procedure = extract_review_procedure(rationale)
    if not review_procedure:
        return None
    
    rationale_without_procedure = remove_review_procedure(rationale)
    input_context = f"{class_name}\n\n{rationale_without_procedure}"
    user_content = PROMPT_TEMPLATE.replace('{input_context}', input_context)
    
    messages = [
        {"role": "user", "content": user_content + " /no_think"},
        {"role": "assistant", "content": review_procedure}
    ]
    
    return {"messages": messages}

import pandas as pd

def process_split(split_name):
    if split_name in ['train', 'val']:
        folder = 'hf_dataset_grammar_corrected_fixed_procedures_formatted_unified_section_format_latest_newer_class_aggregated_rationales_2023_train_w_rr'
        file = 'train.parquet' if split_name == 'train' else 'validation.parquet'
    else:
        folder = 'hf_dataset_grammar_corrected_fixed_procedures_formatted_unified_section_format_latest_newer_class_aggregated_rationales_2023_test'
        file = 'test.parquet'
    
    input_path = f'../rationales/processed/{folder}/parquet/{file}'
    output_jsonl = f'./data_processing/sft/rationales_sft_{split_name}.jsonl'
    output_parquet = f'./data_processing/sft/rationales_sft_{split_name}.parquet'
    
    df = pd.read_parquet(input_path)
    processed_data = [process_row(row) for _, row in df.iterrows() if process_row(row)]
    
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')
    
    pd.DataFrame(processed_data).to_parquet(output_parquet)
    print(f"{split_name}: {len(processed_data)} samples -> {output_jsonl}")

for split in ['train', 'val', 'test']:
    try:
        process_split(split)
    except Exception as e:
        print(f"{split} error: {e}")
