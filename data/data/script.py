import pandas as pd
import os
import re
from textblob import TextBlob
from langdetect import detect, LangDetectException
from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel
from jinja2 import Template
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import webbrowser


# Load the CSV file
input_csv_files = [
    'csvs/January.csv',
    'csvs/February.csv',
    'csvs/March.csv',
    'csvs/April.csv',
    'csvs/May.csv',
    'csvs/June.csv',
    'csvs/July.csv',
    'csvs/August.csv',
    'csvs/September.csv',
    'csvs/October.csv',
    'csvs/November.csv'
]

md_directory = 'mds' 

def find_md_content(row, md_directory):
    journaling_str = str(row['Journaling']) 
    for filename in os.listdir(md_directory):
        file_path = os.path.join(md_directory, filename)
        if os.path.isdir(file_path):
            continue
        if journaling_str in filename:
            with open(file_path, 'r') as file:
                return file.read()
    return 'Content Not Found'

combined_df = pd.DataFrame()

for csv_file in input_csv_files:
    df = pd.read_csv(csv_file)
    df['MD Content'] = df.apply(lambda row: find_md_content(row, md_directory), axis=1)

    combined_df = pd.concat([combined_df, df], ignore_index=True)



# output_csv_path = 'enriched_csvfile.csv' 
# combined_df.to_csv(output_csv_path, index=False)


def clean_md_content(text):
    header_pattern = r'#.*?(?:\n.+?:.+?)*\n'
    match = re.search(header_pattern, text, re.DOTALL)
    if match:
        cleaned_text = text[match.end():].strip()
        if not cleaned_text:
            return 'Content Not Found'
        return cleaned_text
    return text

combined_df['MD Content'] = combined_df['MD Content'].apply(clean_md_content)

# output_csv_path = 'cleaned_output.csv'
# combined_df.to_csv(output_csv_path, index=False)
#

tokenizer = RegexTokenizer()
model = FastTextSocialNetworkModel(tokenizer=tokenizer)

def detect_language(text):
    return detect(text)

def get_sentiment_en(text):
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity  # Polarity score: -1 to 1
    return normalize(sentiment)

def normalize(sentiment):
    return (sentiment + 1) / 2

def get_sentiment_ru(text):
    results = model.predict([text], k=1)
    sentiment = calculate_sentiment_score(results[0])
    # print(results)
    return normalize(sentiment)

def calculate_sentiment_score(sentiment_dict):
    score = 0.0
    total_weight = 0.0
    for sentiment, probability in sentiment_dict.items():
        if sentiment == 'positive':
            weight = 1.0
        elif sentiment == 'neutral':
            weight = 0.05
        elif sentiment == 'negative':
            weight = -1.0
        else:
            continue

        score += probability * weight
        total_weight += score

    return total_weight


def analyze_sentiment(text):
    lang = detect_language(text)
    if lang == 'en':
        return get_sentiment_en(text)
    elif lang == 'ru':
        return get_sentiment_ru(text)
    else:
        return None

# Apply sentiment analysis
combined_df['Sentiment'] = combined_df['MD Content'].apply(analyze_sentiment)

output_csv_path = 'final.csv'
combined_df.to_csv(output_csv_path, index=False)


# Run jupyter
notebook_filename = '../../src/jupyter.ipynb'

with open(notebook_filename) as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

ep.preprocess(nb, {'metadata': {'path': '../../src/'}})

with open(notebook_filename, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

    webbrowser.open('http://127.0.0.1:5500/src/index.html')