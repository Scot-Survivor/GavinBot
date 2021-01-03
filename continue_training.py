import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import re
import marshal
import os
from concurrent.futures import ThreadPoolExecutor, wait

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

os.environ['TF_GPU_THREAD_MODE'] = "gpu_private"

print(f"TensorFlow Version: {tf.__version__}")
print(f"Numpy Version: {np.__version__}")
print(f"Eager execution: {tf.executing_eagerly()}")

print(f"Using Tensorflow version {tf.__version__}")
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


print("Importing Question and Answer data. This may take a second...")
questionsPickle = f"{log_dir}/pickles/{Modelname}_questions.marshal"
answersPickle = f"{log_dir}/pickles/{Modelname}_answers.marshal"
questions, answers = load_files(questionsPickle, answersPickle)
print("Done importing")

answer = input("Would you like to load the tokenizer? y/n\n>")
if answer == "y":
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(f"{log_dir}/tokenizer/vocabTokenizer")
else:
    print("Starting Tokenizer this may take a while....")
    # Build tokenizer using tfds for both questions and answers
    tokenizer = tfds.deprecatedTest.text.SubwordTextEncoder.build_from_corpus(
        questions + answers, target_vocab_size=TARGET_VOCAB_SIZE)
    VOCAB_SIZE = tokenizer.vocab_size + 2
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


# Tokenize, filter and pad sentences
def tokenize_and_filter(inputs, outputs):
    # Get rid of any inputs/outputs that don't meet the max_length requirement (save the model training on large sentences
    new_inputs, new_outputs = [], []
    for i, (sentence1, sentence2) in enumerate(zip(inputs, outputs)):
        if len(sentence1) <= MAX_LENGTH - 2 and len(sentence2) <= MAX_LENGTH - 2:
            new_inputs.append(sentence1)
            new_outputs.append(sentence2)
    inputs, outputs = new_inputs, new_outputs

    # Init the shapes for the array.
    shape_inputs = (len(inputs), MAX_LENGTH)
    shape_outputs = (len(outputs), MAX_LENGTH)
    # create the empty numpy arrays
    # Add the start token at the start of all rows
    inputs_array = np.zeros(shape=shape_inputs)
    inputs_array[:, 0] = START_TOKEN[0]
    outputs_array = np.zeros(shape=shape_outputs)
    outputs_array[:, 0] = START_TOKEN[0]
    # Iterate over each sentence in both inputs and outputs.
    for i, (sentence1, sentence2) in enumerate(zip(inputs, outputs)):
        # encode both sentences
        tokenized_sentence1 = tokenizer.encode(sentence1)
        tokenized_sentence2 = tokenizer.encode(sentence2)
        # This check length doesn't technically matter but its here as a fail safe.
        if len(tokenized_sentence1) <= MAX_LENGTH -2 and len(tokenized_sentence2) <= MAX_LENGTH -2:
            # Add the tokenized sentence into array.
            # This acts as padding for hte
            inputs_array[i, 1:len(tokenized_sentence1)+1] = tokenized_sentence1
            inputs_array[i, len(tokenized_sentence1)+1] = END_TOKEN[0]

            outputs_array[i, 1:len(tokenized_sentence2)+1] = tokenized_sentence2
            inputs_array[i, len(tokenized_sentence2)+1] = END_TOKEN[0]

    return inputs_array, outputs_array


print("Filtering data")
questions, answers = tokenize_and_filter(questions, answers)
print("Done filtering")
questions_train = questions[int(round(len(questions)*.8, 0)):]
answers_train = answers[int(round(len(answers)*0.8, 0)):]
questions_val = questions[:int(round(len(questions)*.2, 0))]
answers_val = answers[:int(round(len(answers)*.2, 0))]


# decoder inputs use the previous target as input
# remove s_token from targets
print("Beginning Dataset shuffling, batching and prefetch")
dataset_train = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions_train,
        'dec_inputs': answers_train[:, :-1]
    },
    {
        'outputs': answers_train[:, 1:]
    }
))
dataset_val = tf.data.Dataset.from_tensor_slices((
    {
        'inputs': questions_val,
        'dec_inputs': answers_val[:, :-1]
    },
    {
        'outputs': answers_val[:, 1:]
    }
))
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


def scaled_dot_product_attention(query, key, value, mask):
    # Calculate the attention weights
    matmul_qk = tf.matmul(query, key, transpose_b=True)

    # scale matmul_qk
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # add the mask to zero out padding tokens
    if mask is not None:
        logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k)
    attention_weights = tf.nn.softmax(logits, axis=-1)

    output = tf.matmul(attention_weights, value)

    return output


# noinspection PyMethodOverriding
class MultiHeadAttention(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, name="multi_head_attention"):
        super(MultiHeadAttention, self).__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.query_dense = tf.keras.layers.Dense(units=d_model)
        self.key_dense = tf.keras.layers.Dense(units=d_model)
        self.value_dense = tf.keras.layers.Dense(units=d_model)

        self.dense = tf.keras.layers.Dense(units=d_model)

    def split_heads(self, inputs, batch_size):
        inputs = tf.reshape(
            inputs, shape=(batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(inputs, perm=[0, 2, 1, 3])

    def call(self, inputs):
        query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
        batch_size = tf.shape(query)[0]

        # linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # split heads
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        # scaled dot-production attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # concatenation of heads
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        # final linear layer
        outputs = self.dense(concat_attention)

        return outputs

    def get_config(self):
        cfg = super().get_config()
        return cfg


def create_padding_mask(x):
    mask = tf.cast(tf.math.equal(x, 0), tf.float32)
    # batch_size, 1, 1, sequence_length
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


# noinspection PyMethodOverriding,PyMethodMayBeStatic
class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model=d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(tf.cast(10000, tf.float32), (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :], d_model=d_model)

        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        cfg = super().get_config()
        return cfg


# noinspection PyTypeChecker
def encoding_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttention(d_model, num_heads, name="attention")({
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': padding_mask
    })
    attention = tf.keras.layers.Dropout(rate=dropout)(attention)
    attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


# noinspection PyTypeChecker
def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name='encoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model=d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = encoding_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name="encoding_layer{}".format(i),
        )([outputs, padding_mask])

    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)


def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
    inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
    enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention1 = MultiHeadAttention(d_model, num_heads, name="attention1")(inputs={
        'query': inputs,
        'key': inputs,
        'value': inputs,
        'mask': look_ahead_mask
    })
    attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)

    attention2 = MultiHeadAttention(d_model, num_heads, name="attention2")(inputs={
        'query': attention1,
        'key': enc_outputs,
        'value': enc_outputs,
        'mask': padding_mask
    })
    attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
    attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

    outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
    outputs = tf.keras.layers.Dense(units=d_model)(outputs)
    outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)


def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
    inputs = tf.keras.Input(shape=(None,), name='inputs')
    enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
    look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
    embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    embeddings = PositionalEncoding(vocab_size, d_model=d_model)(embeddings)

    outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

    for i in range(num_layers):
        outputs = decoder_layer(
            units=units,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            name='decoder_layer_{}'.format(i)
        )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])
    return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)


def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
    inputs = tf.keras.Input(shape=(None,), name="inputs")
    dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

    enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None),
                                              name="enc_padding_mask")(inputs)
    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None),
                                             name="look_ahead_mask")(dec_inputs)

    # mask the encoder outputs for the 2nd attention block
    dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(
        inputs)

    enc_outputs = encoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout
    )(inputs=[inputs, enc_padding_mask])

    dec_outputs = decoder(
        vocab_size=vocab_size,
        num_layers=num_layers,
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout
    )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

    outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

    return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)


mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    model = transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)


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

model.summary()
model.load_weights(checkpoint_path)
model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
with tf.profiler.experimental.Trace("Train"):
    model.fit(dataset_train, validation_data=dataset_val, epochs=EPOCHS, callbacks=[cp_callback, tensorboard_callback], initial_epoch=INITAL_EPOCH)
