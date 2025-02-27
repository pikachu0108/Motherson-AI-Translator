import pandas as pd
import sentencepiece as spm
from sklearn.model_selection import train_test_split
import yaml

sp = spm.SentencePieceProcessor(model_file='spm_model.model')


df = pd.read_csv('Faizoutput.csv', encoding='utf-8')


tokenized_texts = []

for text in df['target']:  
    if isinstance(text, str):
        text = text.strip()
        
        if text:
            tokens = sp.encode(text, out_type=str)
            tokenized_text = " ".join(tokens)
            tokenized_texts.append(tokenized_text)

            
            decoded_text = sp.decode(tokens)
            print("Original:", text)
            print("Tokenized:", tokenized_text)
            print("Decoded:", decoded_text)
            print("-" * 40)
        else:
            tokenized_texts.append("")
    else:
        print("Skipping non-string value:", text)
        tokenized_texts.append("")


df['tokenized_text'] = tokenized_texts

df.to_excel('SPM_output.xlsx', index=False)

#print("File saved successfully as 'tokenizedoutput3.xlsx'")

df = pd.read_csv('SPM_output.csv', encoding='utf-8')


n_samples = len(df)
print(f"Total number of samples: {n_samples}")

if n_samples < 10:
    raise ValueError("Not enough samples to split the dataset. Please ensure the dataset is large enough.")

train_data, temp_data = train_test_split(df, test_size=0.40, random_state=42)  # 60% train
val_data, test_data = train_test_split(temp_data, test_size=0.50, random_state=42)  # Split remaining 40% into 20% test, 20% val

train_data.to_csv('Faiz_SPMTrain.csv', index=False, encoding='utf-8')
val_data.to_csv('Faiz_SPMVal.csv', index=False, encoding='utf-8')
test_data.to_csv('Faiz_SPMTest.csv', index=False, encoding='utf-8')

#print("New Data split and saved successfully.")


# Load the CSV files
train_df = pd.read_csv('Faiz_SPMTrain.csv')
val_df = pd.read_csv('Faiz_SPMVal.csv')
test_df = pd.read_csv('Faiz_SPMTest.csv')

# Save source and target columns to separate files for OpenNMT
train_df['source'].to_csv('Faiz_SPMTrain.src', index=False, header=False)
train_df['target'].to_csv('Faiz_SPMTrain.tgt', index=False, header=False)

val_df['source'].to_csv('Faiz_SPMVal.src', index=False, header=False)
val_df['target'].to_csv('Faiz_SPMVal.tgt', index=False, header=False)

test_df['source'].to_csv('Faiz_SPMTest.src', index=False, header=False)
test_df['target'].to_csv('Faiz_SPMTest.tgt', index=False, header=False)


train_df = pd.read_csv('Faiz_SPMTrain.csv')

source_sentences = train_df['source']
target_sentences = train_df['target']

# Function to tokenize and extract unique words
def build_vocab(sentences):
    vocab = set()  # Use a set to avoid duplicates
    for sentence in sentences:
        if isinstance(sentence, str):
            tokens = sentence.split()  # Tokenize by space
            vocab.update(tokens)
    return sorted(vocab)

# Build vocabularies for source and target
source_vocab = build_vocab(source_sentences)
target_vocab = build_vocab(target_sentences)

# Save vocabularies to files
with open('Faiz_SPMVocab.src', 'w', encoding='utf-8') as src_file:
    for token in source_vocab:
        src_file.write(token + '\n')

with open('Faiz_SPMVocab.tgt', 'w', encoding='utf-8') as tgt_file:
    for token in target_vocab:
        tgt_file.write(token + '\n')

#print("New Vocabulary files 'Faiz_SPMVocab.src' and 'Faiz_SPMVocab.tgt' created successfully!")

# Define the configuration as a Python dictionary
FaizSPMconfig = {
    'save_model': 'model_SPM',
    'save_checkpoint_steps': 500,
    'keep_checkpoint': 10,
    'data': {
        'corpus_1': {
            'path_src': 'Faiz_SPMTrain.src',
            'path_tgt': 'Faiz_SPMTrain.tgt'
        },
        'valid': {
            'path_src': 'Faiz_SPMVal.src',
            'path_tgt': 'Faiz_SPMVal.tgt'
        }
    },
    'rnn_size': 512,
    'word_vec_size': 512,
    'transformer_ff': 2048,
    'heads': 12,
    'layers': 6,
    'dropout': 0.1,
    'train_steps': 510,
    'valid_steps': 51,
    'batch_size': 32,
    'max_generator_batches': 2,
    'accum_count': [2],
    'optim': 'adam',
    'learning_rate': 1.0,
    'warmup_steps': 4000,
    'report_every': 100,
    'early_stopping': 8,
    'valid_batch_size': 16,
    'log_file': 'Faiz_SPMTrain.log',
    'src_vocab': 'Faiz_SPMVocab.src',
    'tgt_vocab': 'Faiz_SPMVocab.tgt',
    'test': {
        'path_src': 'Faiz_SPMTest.src',
        'path_tgt': 'Faiz_SPMTest.tgt'
    },
    'world_size': 1,
    'gpu_ranks': [1]
}


with open('Hassanconfig.yml', 'w') as file:
    yaml.dump(FaizSPMconfig, file, default_flow_style=False)

print("New YAML file 'Hassanconfig.yml' created successfully.")

