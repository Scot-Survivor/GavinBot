import tensorflow as tf
import tensorflow_datasets as tfds
import re

from GavinBackend.models import Transformer
from GavinBackend.functions import evaluate

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.set_random_seed(1234)

print(f"Using Tensorflow version {tf.__version__}")


def load_model(checkpoint_name):
    checkpoint_path = f"{checkpoint_name}/cp.ckpt"
    save_path = checkpoint_name
    with open(f"{save_path}/values/hparams.txt", "r", encoding="utf8") as f:
        lines = f.readlines()
        formatted = []
        for line in lines:
            formatted.append(line.replace("\n", ""))
        MAX_SAMPLES = int(formatted[0])
        ModelName = formatted[1]
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
        hparams = [MAX_SAMPLES, MAX_LENGTH, BATCH_SIZE, BUFFER_SIZE, NUM_LAYERS, D_MODEL, NUM_HEADS, UNITS, DROPOUT,
                   VOCAB_SIZE, TARGET_VOCAB_SIZE]
        print(f"""
    Imported Hyper Parameters from {save_path}/values/hparams.txt
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

    tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(f"{save_path}/tokenizer/vocabTokenizer")

    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]

    # Vocabulary size plus start and end token
    print(f"Vocab size: {VOCAB_SIZE}")

    base = Transformer(
        vocab_size=VOCAB_SIZE,
        num_layers=NUM_LAYERS,
        units=UNITS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dropout=DROPOUT)

    model = base.return_model()

    model.load_weights(checkpoint_path).expect_partial()
    return START_TOKEN, END_TOKEN, tokenizer, MAX_LENGTH, model, ModelName, hparams


def checkSwear(sentence, swearwords):
    listSentence = sentence.split()
    listSentenceClean = []
    for word in range(len(listSentence)):
        stars = []
        word_regex = re.sub(r"[^a-zA-Z]+", "", listSentence[word].lower())
        if word_regex in swearwords:
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


def predict(sentence, tokenizer, swear_words, start_token, end_token, max_len, model):
    prediction = evaluate(sentence, start_token, end_token, tokenizer, max_len, model)

    predicated_sentence = tokenizer.decode([i for i in prediction if i < tokenizer.vocab_size])
    predicated_sentence = checkSwear(predicated_sentence, swear_words)
    # print("Input: {}".format(sentence))
    # print("Output: {}".format(predicated_sentence))

    return predicated_sentence
