import logging
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

print(f'Running on Tensor Flow {tf.__version__} and Python {sys.version}.')

# Logging
Path('results').mkdir(exist_ok=True)  # Make a folder to contain result files
tf.compat.v1.logging.set_verbosity(logging.INFO)
handlers = [logging.FileHandler('results/main.log'), logging.StreamHandler(sys.stdout)]
logging.getLogger('tensorflow').handlers = handlers


def timenow():
    return str(datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ':')


# Read training data and testing data from csv files. Training data contains both X and Y (X is given data, Y is expected prediction). Testing data only contains X, and we need to submit predictions (Y) from testing data to Shopee.
print(timenow(), 'Reading training and testing data...')
train = pd.read_csv('train_clean_expanded.csv')
test = pd.read_csv('test.csv')
train_labeled = train.copy()


def label(raw, labels):
    rawAdr = str(raw).split()
    POILabels = labels.split("/", 1)[0].split()
    streetLabels = labels.split("/", 1)[1].split()
    len_a, len_p, len_s = len(rawAdr), len(POILabels), len(streetLabels)
    labelled = ['O' for i in range(len_a)]  # Initialize the labels to be all O
    for i in range(len_a - len_p + 1):
        if rawAdr[i:i+len_p] == POILabels:
            if len_p > 1:
                if all(x == 'O' for x in labelled[i:i+len_p]):  # Check if entire string is unlabelled
                    labelled[i:i + len_p] = ['I-POI'] * len_p
                    labelled[i] = 'B-POI'
                    labelled[i + len_p - 1] = 'E-POI'
                    break
                else:
                    continue
            elif len_p == 1:
                labelled[i] = 'B-POI'
                break
            else:
                continue
        else:
            continue

    for i in range(len_a - len_s + 1):
        if rawAdr[i:i+len_s] == streetLabels:
            if len_s > 1:
                if all(x == 'O' for x in labelled[i:i+len_s]):  # Check if entire string is unlabelled
                    labelled[i:i + len_s] = ['I-STR'] * len_s
                    labelled[i] = 'B-STR'
                    labelled[i + len_s - 1] = 'E-STR'
                    break
                else:
                    continue
            elif len_s == 1:
                labelled[i] = 'B-STR'
                break
            else:
                continue
        else:
            continue

    if len(labelled) == len(rawAdr):
        return labelled
    else:
        return 'ERROR: no. of words dont match no. of tags'


counter = 0


def findSame(labels):
    global counter
    POILabels = labels.split("/", 1)[0].split()
    streetLabels = labels.split("/", 1)[1].split()
    if POILabels == streetLabels and len(POILabels) != 0:
        print(POILabels, streetLabels)
        counter += 1


print(timenow(), 'Labelling training data...')
train_labeled['POI/street'] = train.apply(lambda x: label(x['raw_address'], x['POI/street']), axis=1)
train.apply(lambda x: findSame(x['POI/street']), axis=1)
print(counter)
print(timenow(), 'Training data labelled.')

# Spilt the training data to ratio of 4:3 (80% training 30% validation). Validation data is NOT testing data - val data contain the correct expected prediction (Y)
print(timenow(), 'Splitting training and validation data...')
train_x, val_x, train_y, val_y = train_test_split(train_labeled['raw_address'], train_labeled['POI/street'],
                                                  test_size=0.3, random_state=42)
print(timenow(), 'Saving training and validation data words...')
np.savetxt("train.words.txt", train_x.values, encoding='utf-8', fmt='%s')  # Save the training addresses as txt.
np.savetxt("val.words.txt", val_x.values, encoding='utf-8', fmt='%s')  # Save the validation addresses as txt.
print(timenow(), 'Processing training and validation labels...')
train_y = train_y.apply(lambda x: ' '.join(x).strip('[]'))  # Remove the bracket at start and end of the output
val_y = val_y.apply(lambda x: ' '.join(x).strip('[]'))  # Remove the bracket at start and end of the output
print(timenow(), 'Saving training and validation data labels...')
np.savetxt("train.tags.txt", train_y.values, encoding='utf-8', fmt='%s')  # Save the training labels as txt.
np.savetxt("val.tags.txt", val_y.values, encoding='utf-8', fmt='%s')  # Save the validation labels as txt.
print(timenow(), 'All data saved.')

# # NOT USING FOR NOW
# # Create empty lists to store POI and Street training and validation data separately.
# POI_train, POI_val = [], []
# street_train, street_val = [], []
#
# # Spilt at the character "/". Put whatever in front as POI and behind as Street Name. [0] returns strings before '/', [1] returns strings after '/'.
# for i in train_y:
#     POI_train += [i.split("/", 1)[0]]
#     street_train += [i.split("/", 1)[1]]
#
# for i in val_y:
#     POI_val += [i.split("/", 1)[0]]
#     street_val += [i.split("/", 1)[1]]
#
#
# max_len = int(train.raw_address.map(len).max())
# vectorize_layer = TextVectorization(output_sequence_length=max_len)
# vectorize_layer.adapt(train_x.tolist() + train_y.tolist() + val_x.tolist() + val_y.tolist())
#

print()
