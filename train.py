import os
import re
import marshal
import numpy as np
from datetime import datetime
from random import shuffle, randint
from multiprocessing import Pool
from functools import partial
from collections import Iterable
os.environ['TF_GPU_THREAD_MODE'] = "gpu_private"
if __name__ == "__main__":
    # import tensorflow as tf not needed since its imported through architecture.models
    import tensorflow_datasets as tfds
    from tensorflow.keras.utils import plot_model
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    from concurrent.futures import ThreadPoolExecutor, wait
    from tensorboard.plugins import projector
    from architecture.models import Transformer, tf
    from architecture.callbacks.model_callbacks import PredictCallback
    # from keras.preprocessing.text import Tokenizer  Different Tokenizer.


#  INIT all the variables to avoid "can be undefined" errors.
path_to_dataset = path_to_movie_conversations = path_to_movie_lines = questions = answers = dataset_train = dataset_val = regex = None
MAX_SAMPLES = MAX_LENGTH = name = log_dir = load = tokenizerPath = checkpoint_path = tokenizer = optimizer = other_policy = MIXED = None
NUM_LAYERS = D_MODEL = NUM_HEADS = UNITS = DROPOUT = EPOCHS = BATCH_SIZE = BUFFER_SIZE = cores = TARGET_VOCAB_SIZE = VOCAB_SIZE = 0
reddit_set_max = movie_dialog_max = START_TOKEN = END_TOKEN = regex_cores = 0

if __name__ == "__main__":
    other_policy = 'n'  # input("Do you want to enabled mixed precision? y/n (NOT SUPPORTED YET): ")
    if other_policy == 'y':
        MIXED = True
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(f"{len(gpus)} Physical GPUS, {len(logical_gpus)} Logical GPUS.")
            except RuntimeError as e:
                print(e)
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_policy(policy)
    else:
        MIXED = False
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)

    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Numpy Version: {np.__version__}")
    print(f"Eager execution: {tf.executing_eagerly()}")

    path_to_dataset = "cornell movie-dialogs corpus"

    path_to_movie_lines = os.path.join(path_to_dataset, "movie_lines.txt")
    path_to_movie_conversations = os.path.join(path_to_dataset, "movie_conversations.txt")
    # tf.compat.v1.set_random_seed(1234)
    MAX_SAMPLES = int(input("MAX_SAMPLES: "))
    name = input("Please enter a ModelName for this train: ")
    log_dir = "bunchOfLogs/" + name
    BATCH_SIZE = int(input("BATCH_SIZE(32): "))
    BUFFER_SIZE = 20_000
    MAX_LENGTH = 40 + 2  # Max Length changed back to 40 to save model from useless large information

    # Hyper-parameters
    NUM_LAYERS = int(input("Please enter the number of NUM_LAYERS(4): "))
    D_MODEL = int(input("Please enter the d_model(256): "))
    NUM_HEADS = int(input("Please enter the NUM_HEADS(8): "))
    UNITS = int(input("Please enter the number of units(512): "))
    DROPOUT = float(input("Please enter the DROPOUT(0.175): "))
    EPOCHS = int(input("Please enter the number of epochs(15): "))
    load = input("Would you like to load the tokenizer? y/n: ")
    tokenizerPath = None
    if load == "y":
        tokenizerPath = input("Please enter the path the tokenizer: ")
    regex = input("Do you need to run regex? y/n: ")
    cores = 4
    regex_cores = os.cpu_count()
    TARGET_VOCAB_SIZE = 2 ** 14

    checkpoint_path = f"{log_dir}/cp.ckpt"
    try:
        os.mkdir(f"{log_dir}")
        os.mkdir(f"{log_dir}/model/")
        os.mkdir(f"{log_dir}/pickles/")
        os.mkdir(f"{log_dir}/tokenizer")
        os.mkdir(f"{log_dir}/values/")
        os.mkdir(f"{log_dir}/images/")
        os.mkdir(f"{log_dir}/logs/")
    except FileExistsError:
        print("Already exists not creating folders")
        pass

    reddit_set_max = MAX_SAMPLES
    movie_dialog_max = 0
    while reddit_set_max > MAX_SAMPLES or None:
        reddit_set_max = int(input("Please enter a valid number\n>"))
    if movie_dialog_max > 600000:
        reddit_set_max = int(input("Please enter a valid number. The movie dialog only has 600k samples: "))


# tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)


def preprocess_sentence(sentence):
    sentence = sentence.strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,'*])", r" \1 ", sentence)
    sentence = re.sub(r'["]+', " ", sentence)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",", "'")
    sentence = re.sub(r"[^a-zA-z?.!,'*]+", " ", sentence)
    sentence = sentence.strip()
    # adding start and an end token to the sentence
    return sentence


# noinspection PyShadowingNames,PyPep8Naming
def load_conversations(reddit_set_max, movie_dialog_max):
    id2line = {}
    inputs, outputs = [], []
    if movie_dialog_max > 0:
        with open(path_to_movie_lines, errors="ignore") as file:
            lines = file.readlines()
        for line in lines:
            parts = line.replace('\n', '').split(' +++$+++ ')
            id2line[parts[0]] = parts[4]

        with open(path_to_movie_conversations, 'r') as file:
            lines2 = file.readlines()
        for line2 in lines2:
            parts = line2.replace('\n', '').split(" +++$+++ ")
            # get the conversation in a list of line ID
            conversation = [line2[1:-1] for line2 in parts[3][1:-1].split(', ')]
            for i in range(len(conversation) - 1):
                inputs.append(preprocess_sentence(id2line[conversation[i]]))
                outputs.append(preprocess_sentence(id2line[conversation[i + 1]]))
                if len(inputs) >= movie_dialog_max:
                    break

    with open("D:\\Datasets\\reddit_data\\files\\train.from", "r", encoding="utf8", buffering=1000) as file:
        newline = " newlinechar "
        for line in file:
            if newline in line:
                line = line.replace(newline, "\n")
            inputs.append(line)
            if len(inputs) >= reddit_set_max / 2:
                break
        file.close()

    with open("D:\\Datasets\\reddit_data\\files\\train.to", "r", encoding="utf8", buffering=1000) as file:
        newline = " newlinechar "
        for line in file:
            if newline in line:
                line = line.replace(newline, "\n")
            outputs.append(line)
            if len(outputs) >= reddit_set_max / 2:
                file.close()
                return inputs, outputs
        file.close()
    return inputs, outputs


def save_marshal(item, f_path):
    file = open(f_path, 'ab')
    marshal.dump(item, file)
    file.close()


def save_files(item1, item2, file1, file2):
    with ThreadPoolExecutor(2) as executor:
        executor.submit(save_marshal, item1, file1)
        executor.submit(save_marshal, item2, file2)
        wait((executor.submit(save_marshal, item1, file1), executor.submit(save_marshal, item2, file2)))


def generate_data(max_data, f_path):
    num_data = 0
    return_phrases = []
    new_phrases = file_generator(f_path)
    while not num_data >= max_data:
        new_phrase = next(new_phrases)
        phrase = preprocess_sentence(new_phrase)
        if phrase not in return_phrases and not len(return_phrases) >= max_data:
            return_phrases.append(phrase)
            num_data += 1
    return return_phrases


def sort_data(max_data, filepath_one="train.from", filepath_two="train.to"):
    inputs = generate_data(f_path=filepath_one, max_data=max_data)
    outputs = generate_data(f_path=filepath_two, max_data=max_data)
    return inputs, outputs


def file_generator(f_path):
    with open(f_path, "r", encoding="utf8", buffering=10000) as file:
        newline = " newlinechar "
        for line in file:
            if newline in line:
                line = line.replace(newline, "\n")
            yield line


def chunk(lst, count):  # Make a list into N number of lists
    size = len(lst) // count  # figure out the size of them all
    for i in range(0, count):
        s = slice(i * size, None if i == count - 1 else (
                                                                i + 1) * size)  # Using slice here because you can't store 2:3 as a variable
        yield lst[s]  # Yield the list


def preprocess_process(sentences):
    outputs = []
    for sentence in sentences:
        outputs.append(preprocess_sentence(sentence))
    return outputs


def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for i in flatten(item):
                yield i
        else:
            yield item


if __name__ == "__main__":
    print("Loading files...")
    questions, answers = load_conversations(reddit_set_max, movie_dialog_max)
    print(f"Answers: {len(answers)}\nQuestions: {len(questions)}")
    # questions, answers = sort_data(reddit_set_max)
    shuffleThis = list(zip(questions, answers))
    for x in range(randint(0, 4)):
        shuffle(shuffleThis)
    questions, answers = zip(*shuffleThis)
    print("Done loading...")
    if regex == "y":  # If we're running the regex do this.
        print("Starting Regex....")
        # Create the Generator Objects
        generator_q = chunk(questions, regex_cores)
        generator_a = chunk(answers, regex_cores)

        # Use a 2D List to split the questions/answers into chunks for each core (How many used defined by Regex_cores)
        lists_q = [next(generator_q) for _ in range(regex_cores)]
        lists_a = [next(generator_a) for _ in range(regex_cores)]

        # Init the Pool with workers = the number of regex cores
        p = Pool(regex_cores)
        process_outputs = p.map(preprocess_process, lists_q)  # Input the questions into a pool map and gather outputs
        p.close()  # close the pool (releases the works and saves memory)

        questions = list(flatten(process_outputs))  # Flatten the outputs back into a 1D list of strings

        # Do the same for answers
        p = Pool(regex_cores)
        process_outputs = p.map(preprocess_process, lists_a)
        p.close()
        del lists_q, lists_a, generator_a, generator_q  # Delete the objects to save memory later down the line

        answers = list(flatten(process_outputs))  # Flatten the outputs back into a 1D list of strings
        print("Finished Regex")
    if load == "n":  # If we're not loading the tokenizer then generate this
        print("Starting Tokenizer this may take a while....")
        # Build tokenizer using tfds for both questions and answers
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            questions + answers, target_vocab_size=TARGET_VOCAB_SIZE)
        tokenizer.save_to_file(f"{log_dir}/tokenizer/vocabTokenizer")
    else:  # load the tokenizer
        tokenizer = tfds.deprecated.text.SubwordTextEncoder.load_from_file(tokenizerPath)
        tokenizer.save_to_file(f"{log_dir}/tokenizer/vocabTokenizer")
    print("Done Tokenizer.")
    # Define start and end token to indicate the start and end of a sentence
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]  # Set the START and END tokens

    # Vocabulary size plus start and end token
    VOCAB_SIZE = tokenizer.vocab_size + 2  # In create the vocab size to account for the start end token


# Function the process will be mapped to
def check_length(m_length, train_data):
    # Get rid of any inputs/outputs that don't meet the max_length requirement (save the model training on large sentences)
    output_data = {'inputs': [], 'outputs': []}
    for _, sentences in enumerate(train_data):
        if len(sentences[0]) <= m_length - 2 and len(sentences[1]) <= m_length - 2:
            output_data['inputs'].append(sentences[0])
            output_data['outputs'].append(sentences[1])
    return output_data


def tokenize(m_length, s_token, e_token, u_tokenizer, train_data):
    # Init the shapes for the arrays
    shape_inputs = (len(train_data), m_length)
    shape_outputs = (len(train_data), m_length)
    # Create empty arrays
    # Add the Start token at the start of all rows
    inputs_array = np.zeros(shape=shape_inputs, dtype=np.float32)
    inputs_array[:, 0] = s_token[0]
    outputs_array = np.zeros(shape=shape_outputs, dtype=np.float32)
    outputs_array[:, 0] = s_token[0]

    # Iterate over each sentence in both inputs and outputs
    for _, sentences in enumerate(train_data):

        # Encode the sentences
        tokenized_sentence1 = u_tokenizer.encode(sentences[0])
        tokenized_sentence2 = u_tokenizer.encode(sentences[1])

        # This check length doesn't technically matter but its here as a fail safe.
        if len(tokenized_sentence1) <= m_length - 2 and len(tokenized_sentence2) <= m_length - 2:
            # Add the tokenized sentence into array.
            # This acts as padding for the
            inputs_array[_, 1:len(tokenized_sentence1) + 1] = tokenized_sentence1
            inputs_array[_, len(tokenized_sentence1) + 1] = e_token[0]

            outputs_array[_, 1:len(tokenized_sentence2) + 1] = tokenized_sentence2
            outputs_array[_, len(tokenized_sentence2) + 1] = e_token[0]
    return {'inputs': inputs_array, 'outputs': outputs_array}


def tokenize_and_filter(inputs, outputs):
    training_data = list(zip(inputs, outputs))  # Zip the inputs and outputs
    data_gen = chunk(training_data, cores)  # Create the list chunk generator (chunk into how many cores we're using
    partial_iter = partial(check_length, MAX_LENGTH)  # Use partial iter to give more than 1 argument to Pool.map

    process_pool = Pool(processes=cores)  # Init the pool with workers = to number of cores we set
    lists = [next(data_gen) for _ in range(cores)]  # Create a 2D list with each list inside being 1 to give to a worker.
    _process_outputs = process_pool.map(partial_iter, lists)  # Map the function and get outputs
    process_pool.close()  # close pool to kill workers and save memory

    inputs = []  # Set inputs to nothing
    for i in range(cores):
        inputs.extend(_process_outputs[i]['inputs'])  # Extend with the outputs from the pool

    outputs = []  # Set outputs ot nothing
    for i in range(cores):
        outputs.extend(_process_outputs[i]['outputs'])  # Extend with the outputs from the pool

    training_data = list(zip(inputs, outputs))  # re-zip the inputs and outputs
    del inputs, outputs  # Delete the old inputs and outputs to save memory
    data_gen = chunk(training_data, cores)  # Create a new generator using the new data
    # Create a 2D list with each list inside being 1 to give to a worker.
    partial_iter = partial(tokenize, MAX_LENGTH, START_TOKEN, END_TOKEN, tokenizer)  # Use this to pass multiple arguments to pool.map
    process_pool = Pool(processes=cores)  # Init the pool with workers = to number of cores we set
    lists = [next(data_gen) for _ in range(cores)]  # Create a 2D list with each list inside being 1 to give to a worker.
    _process_outputs = process_pool.map(partial_iter, lists)   # Map the function to the lists and return outputs
    process_pool.close()  # Close pool to kill workers and save memory

    inputs_array = np.concatenate((_process_outputs[i]['inputs'] for i in range(cores)))  # Concat the inputs together (similar to .extend used earlier)

    outputs_array = np.concatenate((_process_outputs[i]['outputs'] for i in range(cores)))  # Concat the outputs together (similar to .extend used earlier)
    del lists, data_gen  # delete old objects to help with memory usage

    return inputs_array, outputs_array  # Return the final arrays


if __name__ == "__main__":
    print("Filtering data")
    questions, answers = tokenize_and_filter(questions, answers)  # Filter all the data
    print("Done filtering")

    # Splits the data into Train and Validation
    questions_train = questions[int(round(len(questions) * .8, 0)):]
    answers_train = answers[int(round(len(answers) * 0.8, 0)):]
    questions_val = questions[:int(round(len(questions) * .2, 0))]
    answers_val = answers[:int(round(len(answers) * .2, 0))]
    del questions, answers  # Delete the old objects to help with memory

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
    train_dataset_path = f"{log_dir}/pickles/train/"
    validation_dataset_path = f"{log_dir}/pickles/validation/"
    tf.data.experimental.save(dataset=dataset_val, path=validation_dataset_path, compression="GZIP")
    tf.data.experimental.save(dataset=dataset_train, path=train_dataset_path, compression="GZIP")  # Save the datasets to be loaded later

    mirrored_strategy = tf.distribute.MirroredStrategy()   # Use mirrored strategy to use multi gpu

    with mirrored_strategy.scope():  # Use the mirrored strategy to create the model
        transformer = Transformer(
            vocab_size=VOCAB_SIZE,
            num_layers=NUM_LAYERS,
            units=UNITS,
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dropout=DROPOUT,
            mixed=MIXED)
        model = transformer.return_model()


def loss_function(y_true, y_pred):
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')(y_true, y_pred)

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = tf.multiply(loss, mask)

    return tf.reduce_mean(loss)


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

    print("Input: {}".format(sentence))
    print("Output: {}".format(predicated_sentence))

    return predicated_sentence


if __name__ == "__main__":
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


def accuracy(y_true, y_pred):
    # ensure labels have shape (batch_size, MAX_LENGTH - 1)
    y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
    return tf.metrics.SparseCategoricalAccuracy()(y_true, y_pred)


if __name__ == "__main__":
    print("Writing metadata")

    with open(os.path.join(log_dir, 'metadata.tsv'), "w", encoding="utf-8") as f:
        for subwords in tokenizer.subwords:
            f.write(f"{subwords}\n")
        for unknown in range(1, tokenizer.vocab_size - len(tokenizer.subwords)):
            f.write(f"unknown #{unknown}\n")

    projector_config = projector.ProjectorConfig()
    embedding = projector_config.embeddings.add()

    embedding.metadata_path = 'metadata.tsv'
    projector.visualize_embeddings(log_dir, projector_config)

    linebreak = "--------------------------------"
    log = f"""\nDate: {datetime.now().strftime("%d/%m/%Y %H-%M-%S")},
     Name: {name},
     PATH: {checkpoint_path},
     LogDir: {log_dir},
     Image_Path: {log_dir}/images/combined_{name}.png,
     EPOCHS: {EPOCHS}
     MAX_SAMPLES: {MAX_SAMPLES},
     MAX_LENGTH: {MAX_LENGTH},
     NUM_LAYERS: {NUM_LAYERS},
     D_MODEL: {D_MODEL},
     NUM_HEADS: {NUM_HEADS},
     UNITS: {UNITS},
     DROPOUT: {DROPOUT},
     BATCH_SIZE: {BATCH_SIZE},
     BUFFER_SIZE: {BUFFER_SIZE},
     VOCAB_SIZE: {VOCAB_SIZE},
{linebreak}"""
    with open("Parameters.txt", "a") as f:
        f.write(log)
    with open(f"{log_dir}/values/hparams.txt", "w", encoding="utf8") as f:
        data = f"""{str(MAX_SAMPLES)}
{name}
{str(MAX_LENGTH)}
{str(BATCH_SIZE)}
{str(BUFFER_SIZE)}
{str(NUM_LAYERS)}
{str(D_MODEL)}
{str(NUM_HEADS)}
{str(UNITS)}
{str(DROPOUT)}
{str(VOCAB_SIZE)}
{str(TARGET_VOCAB_SIZE)}
    """
        f.write(data)
        f.close()
    print("Done writing metadata")
    print("Writing Image Structure of the model")
    try:
        plot_model(model, f"{log_dir}/images/{name}_Image.png", expand_nested=True, show_shapes=True)
    except Exception as e:
        with open(f"{log_dir}/images/{name}_Image_Error.txt", "w") as f:
            f.write(f"Image error: {e}")
            print(f"Image error: {e}")
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch="510, 520",
                                                          update_freq='epoch')
    predict_callback = PredictCallback(tokenizer=tokenizer, start_token=START_TOKEN, end_token=END_TOKEN, max_length=MAX_LENGTH,
                                       log_dir=log_dir)
    print("Done.")
    print("Starting train....")

    model.compile(optimizer=optimizer, loss=loss_function, metrics=['accuracy'])
    with tf.profiler.experimental.Trace("Train"):
        model.fit(dataset_train, validation_data=dataset_val, epochs=EPOCHS,
                  callbacks=[cp_callback, tensorboard_callback, predict_callback], use_multiprocessing=True)
    print(log)
    print(linebreak)
    model.summary()
