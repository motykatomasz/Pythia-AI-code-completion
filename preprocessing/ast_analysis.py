from tqdm import tqdm
import json
from ast_utils import NonTerminal
from preprocessing import flatten_ast
"""
This Python file does some simple analysis on the AST to find if the call token type is equal to the callee token type. 
"""


def analyze_asts(ast_file="../data/raw_asts_small.jsonl"):
    with open(ast_file, "r") as ast_file:
        for ast_unparsed in ast_file.readlines():
            ast = NonTerminal.to_tree(json.loads(ast_unparsed))
            file = ast.file

            flat_ast = flatten_ast(ast)

            print(f"Now analyzing {file}.")
            for i, token in enumerate(flat_ast):
                if token.endswith(":@CALL"):
                    token_no_call = token.replace(":@CALL", "")
                    ty = get_type(token_no_call)

                    if ty is not None and ty != get_type(flat_ast[i - 1]):
                        print(f"{ty} == {get_type(flat_ast[i-1])}")
            print()


def get_type(token):
    split_token = token.split(":")

    if len(split_token) > 1:
        return split_token[1]

    return None


analyze_asts("../data/raw_asts.jsonl")
