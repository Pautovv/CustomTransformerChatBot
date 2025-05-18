import numpy as np, os
from tools import TOKENIZER
from tensorflow import keras
from dotenv import load_dotenv
from train.training import model
from train.data_preprocessing import DICTION, targets

load_dotenv()

def GET_ANSWER(REQUEST):
  request = TOKENIZER.texts_to_sequences([REQUEST])
  request = keras.preprocessing.sequence.pad_sequences(request, padding='post', maxlen=int(os.getenv("MAX_LEN")))
  request = np.array(request)

  predict = model([request, targets[:1]], T_flag=False)
  pred_id = np.argmax(predict[0, -1, :]).item()

  for word, index in DICTION.items():
      if index == pred_id:
          answer = word
          break
  else: answer = f"на данный момент я такое не умею"

  return answer