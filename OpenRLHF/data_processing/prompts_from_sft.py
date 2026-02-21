import pandas as pd

for split in ['train', 'val', 'test']:
    df = pd.read_parquet(f'data_processing/sft/rationales_sft_{split}.parquet')
    prompts_df = pd.DataFrame({'context_messages': df['messages'].apply(lambda x: [x[0]])})
    prompts_df.to_parquet(f'data_processing/prompts/rationales_prompts_{split}.parquet')
