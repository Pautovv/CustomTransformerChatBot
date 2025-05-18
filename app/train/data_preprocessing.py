from tensorflow import keras
import numpy as np, pandas as pd, os
from dotenv import load_dotenv
from tools import TOKENIZER

load_dotenv()

data = pd.read_csv(os.getenv("DATA_TRAIN_PATH"))

inputs = data['question']
targets = ['answer']
    
TOKENIZER.fit_on_texts(inputs+targets)

DICTION = TOKENIZER.word_index

inputs = TOKENIZER.texts_to_sequences(inputs)
targets = TOKENIZER.texts_to_sequences(targets)

MAX_LEN = int(os.getenv("MAX_LEN"))

inputs = keras.preprocessing.sequence.pad_sequences(inputs, padding='post', maxlen=MAX_LEN)
targets = keras.preprocessing.sequence.pad_sequences(targets, padding='post', maxlen=MAX_LEN)

inputs = np.array(inputs)
targets = np.array(targets)
