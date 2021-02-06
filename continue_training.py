import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import re
import marshal
import os
from concurrent.futures import ThreadPoolExecutor, wait
from architecture.models import Transformer
from architecture.callbacks.model_callbacks import PredictCallback

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

os.environ['TF_GPU_THREAD_MODE'] = "gpu_private"

print(f"TensorFlow Version: {tf.__version__}")
print(f"Numpy Version: {np.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

checkpoint_name = str(input("Please enter the directory for the checkpoint NO SPACES: "))
EPOCHS = int(input("How many epochs would you like to train for?: "))
INITAL_EPOCH = int(input("At what epoch do you want to start at: "))
checkpoint_path = f"bunchOfLogs/{checkpoint_name}/cp.ckpt"
log_dir = "bunchOfLogs/" + checkpoint_name

with open(f"{log_dir}/values/hparams.txt", "r", encoding="utf8") as f:
    lines = f.readlines()
    formatted = []
    for line in lines:
        formatted.append(line.replace("\n", ""))
    MAX_SAMPLES = int(formatted[0])
    Modelname = formatted[1]
    MAX_LENGTH = int(formatted[2])
    BATCH_SIZE = int(formatted[3])
    BUFFER_SIZE = int(formatted[4])
    NUM_LAYERS = int(formatted[5])
    D_MODEL = int(formatted[6])
    NUM_HEADS = int(formatted[7])
    UNITS = int(formatted[8])
    DROPOUT = float(formatted[9])
    VOCAB_SIZE = int(formatted[10])
    TARGET_VOCAB_SIZE = int(formatted[11])
    print(f"""
Imported Hyper Parameters from {log_dir}/values/hparams.txt are these correct?
MAX_SAMPLES: {MAX_SAMPLES}
MAX_LENGTH: {MAX_LENGTH}
BATCH_SIZE: {BATCH_SIZE}
BUFFER_SIZE: {BUFFER_SIZE}
NUM_LAYERS: {NUM_LAYERS}
D_MODEL: {D_MODEL}
NUM_HEADS: {NUM_HEADS}
UNITS: {UNITS}
DROPOUT: {DROPOUT}
VOCAB_SIZE: {VOCAB_SIZE}
VOCAB_TARGET_SIZE: {TARGET_VOCAB_SIZE}
""")
    f.close()


def load_marshal(filepath):
    with open(filepath, 'rb') as fp:
        return marshal.load(fp)


def load_files(file1, file2):
    with ThreadPoolExecutor(2) as executor:
        fut1 = executor.submit(load_marshal, file1)
        fut2 = executor.submit(load_marshal, file2)
        wait((fut1, fut2))

        exc1 = fut1.exception()
        if exc1 is not None:
            raise exc1

        exc2 = fut2.exception()
        if exc2 is not None:
            raise exc2

        return fut1.result(), fut2.result()


print("Loading Tokenizer.")
tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(f"{log_dir}/tokenizer/vocabTokenizer")
print("Done Tokenizer.")

# Define start and end token to indicate the start and end of a sentence
START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

# Vocabulary size plus start and end token
print(f"Vocab size: {VOCAB_SIZE}")

swearwords = ['cock', 'tf', 'reggin', 'bellend', 'twat',
              'bollocks', 'wtf', 'slag', 'fucker', 'rapist',
              'shit', 'bitch', 'minger', 'nigger', 'fking',
              'wanker', 'hentai', 'ffs', 'porn', 'tits',
              'fucking', 'knob', 'minge', 'clunge', 'whore',
              'bloodclat', 'fuck', 'cunt', 'crap', 'pissed',
              'prick', 'nickger', 'cocks', 'pussy', "fucking",
              "bullshit", "slut", "fuckin'", "slut"]




# decoder inputs use the previous target as input
# remove s_token from targets
print("Beginning Dataset shuffling, batching and prefetch")
train_dataset_path = f"{log_dir}/pickles/train/"
validation_dataset_path = f"{log_dir}/pickles/validation/"
dataset_train = tf.data.experimental.load(path=train_dataset_path, element_spec={{'inputs': tf.TensorSpec(shape=(), dtype=tf.float32)}, {'outputs': tf.TensorSpec(shape=(), dtype=tf.float32)}}, compression="GZIP")
dataset_val = tf.data.experimental.load(path=validation_dataset_path, element_spec={{'inputs': tf.TensorSpec(shape=(), dtype=tf.float32)}, {'outputs': tf.TensorSpec(shape=(), dtype=tf.float32)}}, compression="GZIP")


dataset_train = dataset_train.cache()
dataset_train = dataset_train.shuffle(BUFFER_SIZE)
dataset_train = dataset_train.batch(BATCH_SIZE)
dataset_train = dataset_train.prefetch(tf.data.experimental.AUTOTUNE)
dataset_val = dataset_val.cache()
dataset_val = dataset_val.shuffle(BUFFER_SIZE)
dataset_val = dataset_val.batch(BATCH_SIZE)
dataset_val = dataset_val.prefetch(tf.data.experimental.AUTOTUNE)
print("Done Dataset shuffling, batching and prefetch")


def preprocess_sentence(sentence):
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,'])", r"\1", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,']+", " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-z?.!,']+", " ", sentence)
    sentence = sentence.strip()
    # adding start and an end token to the sentence
    return sentence


mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    base = Transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)
    model = base.return_model()


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                         reduction='none')(y_true, y_pred)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


def checkSwear(sentence):
    listSentence = sentence.split()
    listSentenceClean = []
    for word in range(len(listSentence)):
        stars = []
        if listSentence[word].lower() in swearwords:
            firstLetter = str(listSentence[word][0])
            for x in range(len(listSentence[word]) - 1):
                stars.append("#")
            stars = "".join(stars)
            censored = firstLetter + stars
            listSentenceClean.append(censored)
        else:
            listSentenceClean.append(listSentence[word])
        # print(listSentenceClean)
    listSentenceClean = " ".join(listSentenceClean)
    return listSentenceClean


def evaluate(sentence):
    sentence = preprocess_sentence(sentence)

    sentence = tf.expand_dims(START_TOKEN + tokenizer.encode(sentence) + END_TOKEN, axis=0)

    output = tf.expand_dims(START_TOKEN, 0)

    for i in range(MAX_LENGTH):
        predictions = model(inputs=[sentence, output], training=False)

        # select the last word from the seq length dimension
        predictions = predictions[:, -1:, :]
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        if tf.equal(predicted_id, END_TOKEN[0]):
            break

        # concatenated the predicted_id to the output which is given the decoder
        # as its input
        output = tf.concat([output, predicted_id], axis=-1)
    return tf.squeeze(output, axis=0)


def predict(sentence):
    prediction = evaluate(sentence)

    predicated_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
    predicated_sentence = checkSwear(predicated_sentence)
    # print("Input: {}".format(sentence))
    # print("Output: {}".format(predicated_sentence))

    return predicated_sentence


# noinspection PyAbstractClass,PyShadowingNames
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    def get_config(self):
        config = {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps
        }
        return config


learning_rate = CustomSchedule(D_MODEL)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch="510, 520",
                                                      update_freq='epoch')
predict_callback = PredictCallback(tokenizer=tokenizer, start_token=START_TOKEN, end_token=END_TOKEN, max_length=MAX_LENGTH,
                                       log_dir=log_dir)

model.summary()
model.load_weights(checkpoint_path)
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
with tf.profiler.experimental.Trace("Train"):
    model.fit(dataset_train, validation_data=dataset_val, epochs=EPOCHS, callbacks=[cp_callback, tensorboard_callback,
                                                                                    predict_callback],
              initial_epoch=INITAL_EPOCH)
