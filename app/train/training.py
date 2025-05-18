import tensorflow as tf, os
from tensorflow import keras
from dotenv import load_dotenv
from models.model import Transformer
from tools import OPTIMIZER, LOSS_FUNC
from data_preprocessing import DICTION, inputs, targets

load_dotenv()

L_count = int(os.getenv("L_count"))
EMB_dim = int(os.getenv("EMB_dim"))
HEAD_count = int(os.getenv("HEAD_count"))
FFL_dim = int(os.getenv("FFL_dim"))
DO_rate = float(os.getenv("DO_rate"))
EPOCHS = int(os.getenv("EPOCHS"))
MAXSEQ_len = int(os.getenv("MAX_LEN"))
INVOC_dim = len(DICTION) + 1
OUTVOC_dim = len(DICTION) + 1

model = Transformer(L_count, EMB_dim, HEAD_count, FFL_dim, DO_rate, INVOC_dim, OUTVOC_dim, MAXSEQ_len)

def LOSS(target, pred):
  mask = tf.math.logical_not(tf.math.equal(target, 0))
  loss_ = LOSS_FUNC(target, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

train_loss = keras.metrics.Mean(name='train_loss')
train_accuracy = keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')


@tf.function
def TRAIN(inputs, targets):
  with tf.GradientTape() as tape:
    preds = model([inputs, targets], T_flag=True)
    loss = LOSS(targets, preds)

  grads = tape.gradient(loss, model.trainable_variables)
  OPTIMIZER.apply_gradients(zip(grads, model.trainable_variables))

  train_loss(loss)
  train_accuracy(targets, preds)
  

for epoch in range(EPOCHS):
  train_loss.reset_state()
  train_accuracy.reset_state()

  TRAIN(inputs, targets)
  print(f'EPOCH{epoch + 1}, LOSS: {train_loss.result()}, ACCURACY: {train_accuracy.result()}')


