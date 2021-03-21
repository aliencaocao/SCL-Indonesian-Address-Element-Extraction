"""Reload and serve a saved model"""

__author__ = "Guillaume Genthial, modified by Billy Cao 2021"

import sys
from datetime import datetime
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow.contrib import predictor

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
print(f'Running on Tensor Flow {tf.__version__} and Python {sys.version}.')

print('Reading test data...')
test = pd.read_csv('test_expanded.csv')


def timenow():
    return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ':')


timeStart = str(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


def parse_fn(line):
    # Encode in Bytes for TF
    words = [w.encode() for w in line.strip().split()]

    # Chars
    chars = [[c.encode() for c in w] for w in line.strip().split()]
    lengths = [len(c) for c in chars]
    max_len = max(lengths)
    chars = [c + [b'<pad>'] * (max_len - l) for c, l in zip(chars, lengths)]
    return {'words': [words], 'nwords': [len(words)],
            'chars': [chars], 'nchars': [lengths]}


'''
List of tags:
B-POI
B-STR
E-POI
E-STR
I-POI
I-STR
O
'''


# Function to generate the csv file in the required format.
def generate_answer(raw_adr, predictions):
    list_of_tags = [x.decode('utf-8') for x in predictions[0]]  # Decoding needed to convert from binary into string.
    raw_adr_words = str(raw_adr).strip().split()
    poi_result = []
    street_result = []
    combined_result = []
    if len(raw_adr_words) == len(list_of_tags):
        for i in range(len(list_of_tags)):
            if list_of_tags[i] == 'B-POI' or list_of_tags[i] == 'I-POI' or list_of_tags[i] == 'E-POI':
                poi_result += [raw_adr_words[i]]
                continue
            elif list_of_tags[i] == 'B-STR' or list_of_tags[i] == 'I-STR' or list_of_tags[i] == 'E-STR':
                street_result += [raw_adr_words[i]]
                continue
            elif list_of_tags[i] == 'O':
                continue
            else:
                print(f'{raw_adr_words[i]}: {list_of_tags[i]} is not recognized as any tag!')
                break
    else:
        print('Raw address and prediction tags length don\'t match!')

    combined_result = f"{' '.join(poi_result).strip('[]')}/{' '.join(street_result).strip('[]')}"  # Merge the predicted POI and street name into one string, then add it into the combined_result list.
    return combined_result.strip('[]')

    # for i in range(len(combined_result)):
    #     template_df.iat[i, 1] = combined_result[i]

    # template_df.to_csv('result.csv', index=False)


if __name__ == '__main__':
    export_dir = 'saved_model'
    subdirs = [x for x in Path(export_dir).iterdir()
               if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    print(timenow(), 'Loading saved model predictor...')
    predict_fn = predictor.from_saved_model(latest)
    # Initialize the empty data frame
    print(timenow(), 'Initializing template data frame...')
    template_df = pd.DataFrame({'id': [i for i in range(50000)], 'POI/street': ['/' for i in range(50000)]},
                               columns=['id', 'POI/street'])

    # Append the decoded labels
    print(timenow(), 'Prediction started...(might take a while, last measured 17min)')
    # prediction input is a dictionary, thus need ['tags'] to read the predicted tags. The tags are in NumPy array format, thus tolist() convert it into a list.
    template_df["POI/street"] = test.apply(
        lambda x: generate_answer(x['raw_address_expanded'], predict_fn(parse_fn(x['raw_address']))['tags'].tolist()), axis=1)
    print(timenow(), 'Prediction complete. Saving results to results.csv...')
    template_df.to_csv('result.csv', index=False)
    print(timenow(), f'Results saved. Time started: {timeStart}')

    # for raw_adr in test['raw_address'][:2]:
    #     predictions = predict_fn(parse_fn(raw_adr))
    #     generate_answer(raw_adr, predictions['tags'].tolist())

    print()
