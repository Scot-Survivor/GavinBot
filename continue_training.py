import re
import marshal
import os

import tensorflow_datasets as tfds
import numpy as np
import GavinBackend.functions as gbf
import GavinBackend.preprocessing.tokenise as gbpt

from concurrent.futures import ThreadPoolExecutor, wait
from GavinBackend.models import Transformer, tf
from GavinBackend.callbacks.model_callbacks import PredictCallback

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

os.environ['TF_GPU_THREAD_MODE'] = "gpu_private"
if __name__ == "__main__":
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Numpy Version: {np.__version__}")
    print(f"Eager execution: {tf.executing_eagerly()}")

    checkpoint_name = str(input("Please enter the directory for the checkpoint NO SPACES: "))
    EPOCHS = int(input("How many epochs would you like to train for?: "))
    INITAL_EPOCH = int(input("At what epoch do you want to start at: "))
    CORES = int(input("How many cores would you like to use for preprocessing?: "))
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
Imported Hyper Parameters from {log_dir}/values/hparams.txt
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

    # Load data
    print("Loading Data")
    questions_path = f"{log_dir}/pickles/{Modelname}_questions.marshal"
    answers_path = f"{log_dir}/pickles/{Modelname}_answers.marshal"
    questions, answers = load_files(questions_path, answers_path)
    print("Finished Loading Data.")
    print("Filtering Data.")
    questions, answers = gbpt.tokenize_and_filter(questions, answers, CORES, MAX_LENGTH, START_TOKEN, END_TOKEN,
                                                  tokenizer)
    print("Done filtering Data.")

    sizes = (len(questions), len(answers))

    questions_train = questions[0: int(sizes[0] * .80)]
    questions_val = questions[int(sizes[0] * 0.80):]
    answers_train = answers[0: int(sizes[1] * .80)]
    answers_val = answers[int(sizes[1] * .80):]


    # decoder inputs use the previous target as input
    # remove s_token from targets
    print("Beginning Dataset shuffling, batching and prefetch")
    dataset_train = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions_train,  # Source
            'dec_inputs': answers_train[:, :-1],  # Targets
        },
        {
            'outputs': answers_train[:, 1:]  # Outputs
        }
    ))
    dataset_val = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': questions_val,  # Source
            'dec_inputs': answers_val[:, :-1],  # Targets
        },
        {
            'outputs': answers_val[:, 1:]  # Outputs
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
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch="500, 600",
                                                          update_freq='epoch')
    predict_callback = PredictCallback(tokenizer=tokenizer, start_token=START_TOKEN, end_token=END_TOKEN, max_length=MAX_LENGTH,
                                           log_dir=log_dir)

    model.summary()
    model.load_weights(checkpoint_path)
    model.compile(optimizer=optimizer, loss=gbf.loss_function, metrics=['accuracy'])
    with tf.profiler.experimental.Trace("Train"):
        model.fit(dataset_train, validation_data=dataset_val, epochs=EPOCHS, callbacks=[cp_callback, tensorboard_callback,
                                                                                        predict_callback],
                  initial_epoch=INITAL_EPOCH)
