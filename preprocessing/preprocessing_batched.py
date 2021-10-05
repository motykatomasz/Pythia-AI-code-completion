import sys
from tqdm import tqdm
import astroid
import os
import json
from ast_utils import transform_ast
"""
Parses Python files into ASTs and stores them into a "jsonl" file in a batched fashion. 
"""


def parse_asts(repos, dir_path, store_file):
    count = 0
    with open(f"../data/{store_file}", "a") as ast_write:
        errors = 0
        for i, line in enumerate(tqdm(repos)):
            ast = parse_ast(dir_path, line.strip())

            if ast is not None:
                count += 1
                ast_write.write(json.dumps(ast.to_dict()) + "\n")

                if count % 1000 == 0:  # Flush every now and then.
                    ast_write.flush()
            else:
                errors += 1

            del ast
        ast_write.flush()
        ast_write.close()

    print(f"Out of {i} files, {errors} couldn't be parsed into an AST.")
    print(f"{i - errors} are saved to disk.")


def parse_ast(base_dir: str, file_name: str):
    """
    Parse ast of file in directory
    :param base_dir str: directory containing the file
    :param file_name str: file to translate to AST
    """

    try:
        with open(os.path.join(base_dir, file_name)) as file:
            file_raw = file.read()
        return transform_ast(astroid.parse(file_raw), file=file_name)
    except:
        return None


if len(sys.argv) < 5:
    print("Not enough arguments: [repos_file] [start] [end] [store_file]")
    sys.exit(0)

repos_file_path = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
file = str(sys.argv[4])

with open(repos_file_path) as repos:
    repo_lines = [i.strip() for i in repos]

repos = repo_lines[start:end]

print(
    f"Covering {start} up and until {end - 1}, which are {len(repos)} elements."
)
dir_path = os.path.dirname(os.path.realpath(repos_file_path)) + "/"

parse_asts(repos, dir_path, file)
