import os
from dotenv import load_dotenv
from tensorflow import keras

load_dotenv()

TOKENIZER = keras.preprocessing.text.Tokenizer(num_words=int(os.getenv("NUM_WORDS")))
OPTIMIZER = keras.optimizers.Adam()
LOSS_FUNC = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')