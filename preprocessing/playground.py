import glob
import pickle
import json

from ast_utils import dfs
from ast_utils import NonTerminal
from ast_utils import transform_ast
from astroid import parse
"""
This Python file is used to play around with parsing and flat mapping of AST files. 
"""


def parse_asts_from_directory(directory, dump_to_disk=False):
    files = glob.glob(directory + "/*.py", recursive=True)
    print(files)
    for file in files:
        parse_ast(file).tree().visualize()


def parse_ast(filename):
    str = open(filename).read()
    return transform_ast(parse(str), file=filename)


#parse_ast("../data/example.py").tree().visualize()
pickle.dump(parse_ast("../data/example.py"), open("../data/example.pl", "wb"))
ast = pickle.load(open("../data/example.pl", "rb"))
ast.tree().visualize()

encoded = json.dumps(ast.to_dict())
print(encoded)
NonTerminal.to_tree(json.loads(encoded)).tree().visualize()
#print(NonTerminal.to_tree(json.loads(encoded)).tree() == ast.tree())
visited = list()
dfs(visited, ast)
print([node.to_string() for node in visited])
