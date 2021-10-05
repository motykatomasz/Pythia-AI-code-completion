import os
from collections import Counter
from typing import List
import json
import subprocess

import numpy as np

from preprocessing.ast_utils import dfs
from preprocessing.ast_utils import transform_ast
from preprocessing.ast_utils import NonTerminal
import astroid
from tqdm import tqdm
import sentencepiece as spm


class Datawriter:

    def __init__(self, batch_size, folder):
        self.batch_size = batch_size
        self.folder = folder
        self.total_count = 0
        self.current_file = 0
        self.cache = []

    def write(self, sample):
        self.cache.append(sample)

        if len(self.cache) >= self.batch_size:
            self.write_to_file()

    def write_to_file(self):
        with open(f"{self.folder}{self.current_file}.csv", "w") as file:
            for sample in self.cache:
                file.write(sample + "\n")
            file.flush()
            file.close()

            self.current_file += 1
            self.total_count += len(self.cache)
            self.cache = []

    def close(self):
        self.write_to_file()
        print(f"Wrote {self.total_count} samples to {self.current_file} files in {self.folder}.")


def deduplicate_filepaths(
        repo_file_path: str,
        duplicates_file_path: str,
        spill_to_disk: bool = False
        ):
    """
    Get all data from the 150k Python dataset and deduplicate it.
    :param repo_file_path: str
    :param duplicates_file_path: str
    :param spill_to_disk: bool
    :return: list of deduplicated file_paths.
    """

    duplicates = []

    # Get all duplicates and add to list.
    with open(duplicates_file_path) as duplicates_file:
        for line in duplicates_file:
            duplicates.append(line)

    deduplicated = []

    # Deduplicate using duplicates list.
    with open(repo_file_path) as repo_file:
        for i, line in enumerate(repo_file):
            if line not in duplicates:
                deduplicated.append(line)

    print(
        f"Removed {(i + 1) - len(deduplicated)} duplications"
        f"out of {(i + 1)} file paths."
        )

    # Add deduplicated file_list to dataset folder.
    if spill_to_disk:
        dir_path = os.path.dirname(os.path.realpath(repo_file_path))
        print(f"Now storing the deduplicated urls in: {dir_path}")

        with open(
                repo_file_path.replace(".txt", "_deduplicated.txt"),
                "w"
                ) as file:

            for line in deduplicated:
                file.write(line)

            file.flush()
            file.close()


def parse_ast(base_dir: str, file_name: str):
    """
    Parse ast of file in directory
    :param base_dir str: directory containing the file
    :param file_name str: file to translate to AST
    """
    with open(os.path.join(base_dir, file_name)) as file:
        file_raw = file.read()

    try:
        return transform_ast(astroid.parse(file_raw), file=file_name)
    except astroid.AstroidSyntaxError:
        return None
    finally:
        del file_raw

def parse_all_asts_and_save(repo_file_path: str, count = -1):
    """
    Parse AST from all the file specified in repo_file_path and then
    saved in a file
    :param repo_file_path str: file containing per file line
    the file to convert to AST
    :param limit int: Total number of files to parse
    """
    dir_path = os.path.dirname(os.path.realpath(repo_file_path)) + "/"
    print(f"Parsing ASTs from data directory: {dir_path}")
    with open(repo_file_path) as repos:
        repo_lines = [i.strip() for i in repos]

    if count >= 0:
        repo_lines = repo_lines[:count]

    batch_size = 3000

    for i in range(0, len(repo_lines), batch_size):
        start = i

        if (i + batch_size) > len(repo_lines):
            end = len(repo_lines)
        else:
            end = i + batch_size

        print(f"Current batch: {start}:{end} out of {len(repo_lines)}.")

        subprocess.call(['python3', 'preprocessing_batched.py', str(repo_file_path), str(start), str(end)])


def flatten_ast(ast) -> List[str]:
    """
    Pre-order DFS of the AST tree into a list of string representation.
    :param ast: The AST to which apply the DFS search
    :return: list of String representation of the AST nodes
    """
    visited = list()
    dfs(visited, ast)
    return [node.to_string() for node in visited]


def build_vocabulary_and_save(raw_asts: str, counter_threshold: int=2, vocab_for_bpe: bool=True, bpe_voc_size: int=10000):
    """
    Builds and stores vocabulary of AST as a file
    :param file raw_asts str: Pickled AST file
    """
    cnt = Counter()
    ast_count = 0
    with open(raw_asts, "r") as ast_file:
        for ast_unparsed in tqdm(ast_file.readlines()):
            ast_count += 1

            ast = NonTerminal.to_tree(json.loads(ast_unparsed))
            for token in flatten_ast(ast):
                token = token.split(':')[0]
                cnt[token.replace(":@CALL", "")] += 1

    if vocab_for_bpe:
        # removed the threshold for considering item as token
        # Repeat item 'count' number of times to reflect a corpus.
        voc_bpe_list = []
        for item, count in cnt.items():
            voc_bpe_list.extend([item]*count)
        # TODO: Read directly from list (must be some method in sentence-piece)
        with open('bpe_vocab.txt', 'w') as f:
            for item in list(voc_bpe_list):
                f.write(f"{item}\n")
        # We can look to add our own method to perform BPE compression if this doesn't work out.
        spm.SentencePieceTrainer.Train(input="bpe_vocab.txt", model_prefix='voc_bpe', vocab_size=bpe_voc_size, model_type='bpe',
                                       user_defined_symbols='</s>')
        # This will save a model and a .vocab file in the root directory
    else:

        voc = [item for item, count in cnt.items() if count >= counter_threshold]
        voc.append(".")

        np.save("../data/voc.npy", np.array(voc))

        print("-- Vocabulary stats --")
        print(f"Amount of ASTs: {ast_count}")
        print(f"Threshold for vocabulary: {counter_threshold}")
        print(f"Vocabulary size: {len(voc)}")

        with open("vocab_unfiltered.txt", 'w') as f:
            for item, count in cnt.items():
                f.write(str(item) + " ")
                f.write(str(count))
                f.write("\n")
            f.flush()
            f.close()


def load_vocabulary(voc_file: str):
    """
    Loads vocabulary previously saved from
    the function `build_vocabulary_and_save`
    :param file voc_file str: File representing Vocabulary
    """
    voc = np.load(voc_file)
    voc_indexed = {}
    voc_inverse_indexed = {}

    for i, v in enumerate(voc):
        voc_indexed[v] = i
        voc_inverse_indexed[i] = v

    return voc_indexed, voc_inverse_indexed


def get_bpe_model(model_name: str):
    '''
    This function returns the model which is specified using the model_name.
    :param model_name: name of the sentence piece model you want to load.
    :return: sentence piece model with name = model_name.
    '''
    sp_bpe = spm.SentencePieceProcessor()
    sp_bpe.load(model_name)

    return sp_bpe


def load_sentence_piece_vocabulary(sp):
    """
    Loads vocabulary based on the sentence piece model provided by first running the get_bpe_model function.
    :param sp: sentencepiece model
    :return: returns a dictionary of vocab words indexed witht the encodings
    """
    voc = [sp.id_to_piece(id) for id in range(sp.get_piece_size())]
    voc_indexed = {}
    voc_inverse_indexed = {}

    for i, v in enumerate(voc):
        voc_indexed[v] = i
        voc_inverse_indexed[i] = v

    return voc_indexed, voc_inverse_indexed

def get_type(token):
    split_token = token.split(":")

    if len(split_token) > 1:
        return split_token[1]

    return None

def raw_asts_to_samples_save(ast_file, vocab_file, lookback_tokens_T=1000):
    vocab, vocab_inverse = load_vocabulary(vocab_file)

    out_of_voc = len(vocab)
    unknown_token = out_of_voc
    padding_token = out_of_voc + 1

    ast_count = 0
    sample_count = 0
    ast_no_calls = 0

    correct_types = 0

    incorrect_types_unfixed = 0 # If var:correct_type is not in vocabulary it cannot be fixed.
    incorrect_types_fixed = 0 # If var:corect_type is in vocabulary it can be fixed.

    data_writer = Datawriter(10000, "../data/training/")

    with open(ast_file, "r") as ast_file:
        for ast_unparsed in tqdm(ast_file.readlines()):
            ast_count += 1

            ast = NonTerminal.to_tree(json.loads(ast_unparsed))
            ast_flat_encoding = []
            call_locations = []

            for i, token in enumerate(flatten_ast(ast)):
                if token in vocab:
                    ast_flat_encoding.append(vocab[token])
                elif token.replace(":@CALL", "") in vocab:
                    call_locations.append(i)
                    ast_flat_encoding.append(vocab[token.replace(":@CALL", "")])
                else:
                    ast_flat_encoding.append(unknown_token)

            if len(call_locations) == 0:
                ast_no_calls += 1

            for call in call_locations:
                if call >= lookback_tokens_T:
                    sample_encoding = ast_flat_encoding[
                            (call-lookback_tokens_T+1):call]
                else:
                    sample_encoding = ast_flat_encoding[:call]

                call_token = vocab_inverse[ast_flat_encoding[call]]
                call_token_type = get_type(call_token)

                if call_token_type is not None:
                    if sample_encoding[-1] != unknown_token:
                        caller_token = vocab_inverse[sample_encoding[-1]]

                        if call_token_type != get_type(caller_token):
                            correct_caller_token = f"var:{call_token_type}"

                            if correct_caller_token in vocab:
                                incorrect_types_fixed += 1
                                sample_encoding = sample_encoding[:-1] + [vocab[correct_caller_token]]
                            else:
                                incorrect_types_unfixed += 1
                                sample_encoding = sample_encoding[:-1] + [unknown_token]
                        else:
                            correct_types += 1
                    else:
                        correct_types += 1 # If the caller token is unknown, we assume the type is correct.
                else:
                    correct_types += 1

                # This encodes the 'method' call and end of sequence character.
                sample_encoding.append(vocab["."])

                while len(sample_encoding) < lookback_tokens_T:
                    sample_encoding.append(padding_token)

                sample_encoding.insert(0, len([encoding for encoding in sample_encoding if encoding != padding_token]))
                sample_encoding.append(ast_flat_encoding[call])

                data_writer.write(','.join(['%.5f' % num for num in sample_encoding]))
                sample_count += 1
    data_writer.close()


    print("-- Training encoding stats --")
    print(f"Amount of ASTs: {ast_count}")
    print(f"Amount of ASTs without method calls: {ast_no_calls}")
    print(f"Ratio incorrect types: {(incorrect_types_unfixed + incorrect_types_fixed)} / {(incorrect_types_unfixed + incorrect_types_fixed + correct_types)}")
    print(f"Ratio fixed types: {(incorrect_types_fixed)} / {incorrect_types_fixed + incorrect_types_unfixed}")
    print(
        f"Total training samples: {sample_count} from"
        f" {ast_count- ast_no_calls} ASTs"
    )


def raw_asts_to_samples_save_bpe(ast_file, vocab_file, lookback_tokens_T=100):
    vocab, vocab_inverse  = load_vocabulary(vocab_file)
    # TODO: take the model name as an argument
    bpe_model = get_bpe_model('voc_bpe.model')
    bpe_vocab, bpe_vocab_inverse = load_sentence_piece_vocabulary(bpe_model)

    # hopefully the no token is longer than this
    max_call_len = 20

    # padding and unknown tokens are at the end of the vocab
    out_of_voc = len(bpe_vocab)
    # the unk token is already defined for the bpe vocab
    unknown_token = 0
    padding_token = out_of_voc

    # Amount of ASTs we (try to) transform to samples.
    ast_count = 0

    # Amount of samples generated.
    sample_count = 0

    # Amount of ASTs without method calls so no samples (method call might also be OoV).
    ast_no_calls = 0

    # Amount of samples for which the type was correct (i.e. call token had same type as token before the call).
    correct_types = 0

    # If var:correct_type is not in vocabulary it cannot be fixed.
    incorrect_types_unfixed = 0

    # If var:correct_type is in vocabulary it can be fixed.
    incorrect_types_fixed = 0

    known_tokens = 0
    unknown_tokens = 0

    revert_eval = 0
    not_revert_eval = 0

    data_writer = Datawriter(10000, "../data/BPE/training/")

    with open(ast_file, "r") as ast_file:
        for ast_unparsed in tqdm(ast_file.readlines()):
            ast_count += 1

            # Load ASTs.
            ast = NonTerminal.to_tree(json.loads(ast_unparsed))

            # Keep track of the AST encoding.
            ast_flat_encoding = []

            # Keep track of the call locations so we can generate samples.
            call_locations = []

            for i, token in enumerate(flatten_ast(ast)):
                # if token is found in the original vocab, we need to now parse that token using sentencepiece
                if token in vocab:
                    # Token will not be found as its a sub token
                    # drop the type after ':'
                    # token = token.split(':')[0]
                    ast_flat_encoding.extend(bpe_model.encode_as_ids(token.split(':')[0]+'</s>'))
                elif token.replace(":@CALL", "") in vocab:
                    # normally, since every word is a single encoding, we append 'i'
                    # Now we should append length(ast_flat_encoding) or +1???
                    call_token = token.replace(":@CALL", "")
                    call_as_bpe_enc = bpe_model.encode_as_ids(call_token.split(':')[0]+'</s>')
                    call_locations.append((len(ast_flat_encoding), len(call_as_bpe_enc)))
                    ast_flat_encoding.extend(call_as_bpe_enc)
                else:
                    # TODO:Need to pick the unknown token from BPE vocab?
                    unknown_tokens += 1
                    # In BPE 0 is the id for unk token
                    ast_flat_encoding.append(unknown_token)
                    # ast_flat_encoding.extend(bpe_model.encode_as_ids(token+'</s>'))

            if len(call_locations) == 0:
                ast_no_calls += 1

            for call, call_len in call_locations:
                # Make sure the token before the call is not unknown.
                # add the extra tokens that were encoded using the BPE 999
                # Handle endge case of >998 tokens
                if call >= (lookback_tokens_T - 2):
                    sample_encoding = ast_flat_encoding[
                                      (call - lookback_tokens_T + 2):call
                                      ]
                else:
                    sample_encoding = ast_flat_encoding[:call]

                # This encodes the 'method' call and end of sequence character.
                # sample_encoding.append(bpe_vocab["."])

                # +2 as we are inserting 2 values at the beginning
                while len(sample_encoding) + 2 < lookback_tokens_T:
                    sample_encoding.append(padding_token)

                sample_encoding.insert(0, len([encoding for encoding in sample_encoding if encoding != padding_token]))
                sample_encoding.insert(1, call_len)
                # add all subtokens till call_len as they are the subtokens of call

                call_encoding = ast_flat_encoding[call: call+call_len]
                # prevent case with longer sequences.
                if len(call_encoding) > max_call_len:
                    call_encoding = call_encoding[:max_call_len]
                else:
                    while len(call_encoding) < max_call_len:
                        call_encoding.append(padding_token)
                # add all subtokens till call_len as they are the subtokens of call
                sample_encoding.extend(call_encoding)

                data_writer.write(','.join(['%.5f' % num for num in sample_encoding]))
                sample_count += 1
    data_writer.close()


    print("-- Training encoding stats --")
    print(f"Amount of ASTs: {ast_count}")
    print(f"Amount of ASTs without method calls: {ast_no_calls}")
    print(
        f"Total training samples: {sample_count} from"
        f" {ast_count- ast_no_calls} ASTs"
    )



if __name__ == '__main__':

    dataset_path = "../../Downloads/py150k/"

    #deduplicate_filepaths(f"{dataset_path}python100k_train.txt", "../duplicates_default.txt", True)
    #deduplicate_filepaths(f"{dataset_path}python50k_eval.txt", "../duplicates_default.txt", True)

    #parse_all_asts_and_save(f"{dataset_path}python100k_train_deduplicated.txt", count=5000)

    # Run this to generate the voc_bpe model file that can then be loaded by sentencepiece.
    '''
    bpe_voc_size is the max size of vocab for BPE
    '''
    # build_vocabulary_and_save("../data/raw_asts_big.jsonl", counter_threshold=5, vocab_for_bpe=True, bpe_voc_size=10000)
    raw_asts_to_samples_save_bpe("../data/raw_asts.jsonl", "../data/voc.npy")
    # raw_asts_to_samples_save("../data/raw_asts.jsonl", "../data/voc.npy")