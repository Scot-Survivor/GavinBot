import os
import re
import tensorflow_datasets as tfds
from random import randint, shuffle


path_to_dataset = "cornell movie-dialogs corpus"

path_to_movie_lines = os.path.join(path_to_dataset, "movie_lines.txt")
path_to_movie_conversations = os.path.join(path_to_dataset, "movie_conversations.txt")


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

    with open("train.from", "r", encoding="utf8", buffering=1000) as file:
        newline = " newlinechar "
        for line in file:
            if newline in line:
                line = line.replace(newline, "\n")
            inputs.append(line)
            if len(inputs) >= reddit_set_max / 2:
                break
        file.close()

    with open("train.to", "r", encoding="utf8", buffering=1000) as file:
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


MAX_SAMPLES = 1000000
MAX_LENGTH = 80 + 2
TARGET_VOCAB_SIZE = int(input("Please enter the vocab size: "))
print(f"Size: {TARGET_VOCAB_SIZE}")
save_path = input("Please enter your save path: ")

reddit_set_max = MAX_SAMPLES
movie_dialog_max = 0

questions, answers = load_conversations(reddit_set_max, movie_dialog_max)

shuffleThis = list(zip(questions, answers))
for x in range(randint(0, 10)):
    shuffle(shuffleThis)
questions, answers = zip(*shuffleThis)
print("Starting Tokenizer this may take a while....")
# Build tokenizer using tfds for both questions and answers
tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions + answers, target_vocab_size=TARGET_VOCAB_SIZE)
tokenizer.save_to_file(f"{save_path}")

