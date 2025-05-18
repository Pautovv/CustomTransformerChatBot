import tensorflow as tf, numpy as np
from tensorflow import keras

class EncoderBlock(keras.layers.Layer):
  def __init__(self, EMB_dim, HEAD_count, FFL_dim, DO_rate):
    super(EncoderBlock, self).__init__()
    self.MHA = keras.layers.MultiHeadAttention(num_heads=HEAD_count, key_dim = EMB_dim // HEAD_count)
    self.FFL = keras.Sequential([
        keras.layers.Dense(FFL_dim, activation='relu'),
        keras.layers.Dense(EMB_dim)
    ])
    self.AN_1 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.AN_2 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.DO_1 = keras.layers.Dropout(rate=DO_rate)
    self.DO_2 = keras.layers.Dropout(rate=DO_rate)
  
  def call(self, X, T_flag):
    MHA_out = self.MHA(X, X, X)
    DO_1_out = self.DO_1(MHA_out, training=T_flag)
    AN_1_out = self.AN_1(DO_1_out + X)

    FFL_out = self.FFL(AN_1_out)
    DO_2_out = self.DO_2(FFL_out, training=T_flag)
    AN_2_out = self.AN_2(DO_2_out + AN_1_out)

    return AN_2_out

class DecoderBlock(keras.layers.Layer):
  def __init__(self, EMB_dim, HEAD_count, FFL_dim, DO_rate):
    super(DecoderBlock, self).__init__()
    self.MHA_1 = keras.layers.MultiHeadAttention(num_heads=HEAD_count, key_dim=EMB_dim // HEAD_count)
    self.MHA_2 = keras.layers.MultiHeadAttention(num_heads=HEAD_count, key_dim=EMB_dim // HEAD_count)
    self.FFL = keras.Sequential([
        keras.layers.Dense(FFL_dim, activation='relu'),
        keras.layers.Dense(EMB_dim)
    ])
    self.AN_1 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.AN_2 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.AN_3 = keras.layers.LayerNormalization(epsilon=1e-6)
    self.DO_1 = keras.layers.Dropout(rate=DO_rate)
    self.DO_2 = keras.layers.Dropout(rate=DO_rate)
    self.DO_3 = keras.layers.Dropout(rate=DO_rate)

  def call(self, X, ENC_out, T_flag, LA_mask, PAD_mask):
    MHA_1_out = self.MHA_1(X, X, X)
    DO_1_out = self.DO_1(MHA_1_out, training=T_flag)
    AN_1_out = self.AN_1(DO_1_out + X)

    MHA_2_out = self.MHA_2(AN_1_out, ENC_out, ENC_out)
    DO_2_out = self.DO_2(MHA_2_out, training=T_flag)
    AN_2_out = self.AN_2(DO_2_out + AN_1_out)

    FFL_out = self.FFL(AN_2_out)
    DO_3_out = self.DO_3(FFL_out, training=T_flag)
    AN_3_out = self.AN_3(DO_3_out + AN_2_out)

    return AN_3_out

class Transformer(keras.models.Model):
  def __init__(self, L_count, EMB_dim, HEAD_count, FFL_dim, DO_rate, INVOC_dim, OUTVOC_dim, MAXSEQ_len):
    super(Transformer, self).__init__()

    self.ENC_emb = keras.layers.Embedding(INVOC_dim, EMB_dim)
    self.DEC_emb = keras.layers.Embedding(OUTVOC_dim, EMB_dim)
    self.POS_enc = self.PositionEncoding(MAXSEQ_len, EMB_dim)
    self.FINAL = keras.layers.Dense(OUTVOC_dim)
    self.DO = keras.layers.Dropout(rate=DO_rate)

    self.ENC = [EncoderBlock(EMB_dim, HEAD_count, FFL_dim, DO_rate) for _ in range(L_count)]
    self.DEC = [DecoderBlock(EMB_dim, HEAD_count, FFL_dim, DO_rate) for _ in range(L_count)]

  def PositionEncoding(self, MAXSEQ_len, EMB_dim):
    ARG = np.arange(MAXSEQ_len)[:, np.newaxis] / np.power(10_000, (2 * (np.arange(EMB_dim) // 2)) / np.float32(EMB_dim))
    ARG[:, 0::2] = np.sin(ARG[:, 0::2])
    ARG[:, 1::2] = np.cos(ARG[:, 1::2])
    POS_ENC = ARG[np.newaxis, ...]
    return tf.cast(POS_ENC, dtype=tf.float32)
  
  def PaddingMask(self, SEQ):
    SEQ = tf.cast(tf.math.equal(SEQ, 0), tf.float32)
    return SEQ[:, tf.newaxis, tf.newaxis, :]
  
  def LookAheadMask(self, SIZE):
    LA_mask = 1 - tf.linalg.band_part(tf.ones((SIZE, SIZE)), -1, 0)
    return LA_mask
  
  def call(self, X, T_flag):
    INPUT, TARGET = X
    ENC_PAD = self.PaddingMask(INPUT)
    DEC_PAD_IN = self.PaddingMask(INPUT)
    DEC_PAD_OUT = self.PaddingMask(TARGET)
    LA_mask = self.LookAheadMask(tf.shape(TARGET)[1])
    MASK = tf.maximum(DEC_PAD_OUT, LA_mask)

    ENC_IN = self.ENC_emb(INPUT)
    ENC_IN += self.POS_enc[:, :tf.shape(INPUT)[1], :]
    ENC_OUT = self.DO(ENC_IN, training=T_flag)
    for enc in self.ENC:
      ENC_OUT = enc(ENC_OUT, T_flag=T_flag)
    
    DEC_IN = self.DEC_emb(TARGET)
    DEC_IN += self.POS_enc[:, :tf.shape(TARGET)[1], :]
    DEC_OUT = self.DO(DEC_IN, training=T_flag)
    for dec in self.DEC:
      DEC_OUT = dec(DEC_OUT, ENC_OUT, T_flag=T_flag, LA_mask=MASK, PAD_mask=DEC_PAD_IN)
    
    FINAL_OUT = self.FINAL(DEC_OUT)

    return FINAL_OUT