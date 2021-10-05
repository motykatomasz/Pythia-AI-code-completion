from astroid import *
from astroid.node_classes import NodeNG
from treeviz.treeviz import Node
"""
This files stores classes and some utility methods to deal with ASTs in Astroid to our own format. 
"""


class NonTerminal:
    """
    None terminal node of the ast tree.
    attribute name str: name of the type of the ast node
    attribute childrenlist: children of the ast node
    """
    def __init__(self, name: str, children: list, file=None):
        self.name = name
        self.children = children
        self.file = file

    def to_string(self):
        """
        Stringify the NonTerminal token.
        :return: string representation of the NT token.
        """
        return self.name

    def to_dict(self):
        """
        Transforms the NT token to a dict so that it can be serialized to JSON.
        :return: dict version of NT token.
        """
        non_terminal = {
            "name": self.name,
            "children": [child.to_dict() for child in self.children]
        }

        if self.file:
            non_terminal["file"] = self.file

        return non_terminal

    def tree(self):
        """
        Create tree object for this NT token. Can be used to visualize a (sub)-AST.
        :return: Node object.
        """
        this_node = Node(self.name)
        [this_node.add_child(child.tree()) for child in self.children]

        return this_node

    @staticmethod
    def to_tree(json):
        """
        Parses a NT token from JSON.
        :param json: the JSON parse from.
        :return: NT object.
        """
        name = json["name"]
        children = [
            NonTerminal.to_tree(child)
            if "children" in child else Terminal.to_tree(child)
            for child in json["children"]
        ]

        if "file" in json:
            return NonTerminal(name, children, json["file"])
        return NonTerminal(name, children)


class Terminal:
    """
    Terminal node of the ast tree.
    attribute value: value of the ast node
    attribute is_function_call: whether the ast node is a function call or not
    """
    def __init__(self, value, type, is_function_call: bool = False):
        self.value = value
        self.type = type
        self.is_function_call = is_function_call

    def to_string(self) -> str:
        """
        Stringify the NonTerminal token. If it is a "call" then the suffix ":@CALL" is added.
        :return: string representation of the T token.
        """
        is_call = ":@CALL" if self.is_function_call else ""

        if self.type == Uninferable or self.type is None or self.type == "None" or self.type == "Uninferable":
            return str(self.value) + is_call
        else:
            return str(self.value) + ":" + str(self.type) + is_call

    def tree(self) -> Node:
        """
        Create tree object for this T token. Can be used to visualize a (sub)-AST.
        :return: Node object.
        """
        return Node(self.to_string())

    def to_dict(self):
        """
        Transforms the T token to a dict so that it can be serialized to JSON.
        :return: dict version of T token.
        """
        terminal = {
            "value": str(self.value),
            "type": str(self.type),
            "is_function_call": self.is_function_call
        }
        return terminal

    @staticmethod
    def to_tree(json):
        """
        Parses a T token from JSON.
        :param json: the JSON parse from.
        :return: T object.
        """
        value = json["value"]
        ty = json["type"]
        is_function_call = json["is_function_call"]
        return Terminal(value, ty, is_function_call)


def dfs(visited, node):
    if node not in visited:
        visited.append(node)

        if hasattr(node, "children"):
            for child in node.children:
                dfs(visited, child)


def transform_ast(node: NodeNG, ty=None, file=None):
    """
    Transforms a astroid ast tree to an easy to access type.
    :param: astroid ast root node to be parsed.
    :return: The transformed ast tree.
    """
    if isinstance(node, FunctionDef):
        if node.returns:  # Return type annotation is there.
            body = [(n, node.returns) if isinstance(n, Return) else n
                    for n in node.body]
        else:
            # We just stick to the body without any return (types).
            body = node.body
        return NonTerminal("FunctionDef", [transform_ast(node.args)] +
                           [transform_ast(b) for b in body])
    if isinstance(node, Arguments):
        return NonTerminal(
            "Arguments",
            [
                parse_terminal_annotated(arg, ann)
                for (arg, ann) in zip(node.args, node.annotations)
            ],
        )
    if isinstance(node, BinOp):
        return NonTerminal("BinOp:" + node.op, [transform_ast(node.left)] +
                           [transform_ast(node.right)])
    if isinstance(node, UnaryOp):
        return NonTerminal(
            "UnaryOp:" + node.op,
            [transform_ast(child) for child in node.get_children()])
    if isinstance(node, BoolOp):
        return NonTerminal(
            "BoolOp:" + node.op,
            [transform_ast(child) for child in node.get_children()])
    if (isinstance(node, tuple) and len(node) == 2
            and isinstance(node[0], Return) and isinstance(node[1], Name)
        ):  # The case where we already know the type of the return variable.
        if isinstance(node[0].value, Name):
            # We only annotate a return type if
            # its direct child is already a terminal.
            return NonTerminal("Return",
                               [Terminal(node[0].value.name, node[1].name)])
        return NonTerminal("Return", [transform_ast(node[0].value)])
    if isinstance(node, Assign):
        # if isinstance(
        #         node.value,
        #         Call):  # If the right side is a call, we don't infer the type!
        #     return NonTerminal("Assign", [
        #         transform_ast(target, "METHOD_CALL") for target in node.targets
        #     ] + [transform_ast(node.value)])

        value_ty = get_type(node.value)
        return NonTerminal(
            "Assign",
            [transform_ast(target, value_ty)
             for target in node.targets] + [transform_ast(node.value)])
    if isinstance(node, AnnAssign):
        return NonTerminal(
            "Assign",
            [parse_terminal_annotated(node.target, node.annotation)] +
            [transform_ast(node.value)],
        )
    if isinstance(node, Const):
        return Terminal("const",
                        node.inferred()[0].pytype().replace("builtins.", ""))
    if isinstance(node, Attribute):
        return NonTerminal(
            "Attribute",
            [
                transform_ast(node.expr),
                Terminal(node.attrname, get_type(node.expr), True)
            ],
        )
    if isinstance(node, Import):
        return NonTerminal(
            "Import",
            [
                Terminal("var", module) if alias is None else Terminal(
                    "var", module) for module, alias in node.names
            ],
        )
    if isinstance(node, ImportFrom):
        import_module = node.modname
        return NonTerminal(
            "ImportFrom",
            [
                Terminal("var", import_module + "." + module) if alias is None
                else Terminal("var", import_module + "." + module)
                for module, alias in node.names
            ],
        )
    if isinstance(node, AssignName):
        return parse_terminal(node, ty)
    if isinstance(node, Name):
        return parse_terminal(node, ty)
    else:
        if file:
            return NonTerminal(
                type(node).__name__,
                [transform_ast(c) for c in node.get_children()], file)
        return NonTerminal(
            type(node).__name__,
            [transform_ast(c) for c in node.get_children()])


def parse_terminal_annotated(node, annotation=None):
    """
    If there is a type annotation for a terminal, we add it.
    :param node: the annotated AST node.
    :param annotation: the annotation.
    :return: a terminal.
    """
    if not annotation:
        return parse_terminal(node)
    else:
        return Terminal("var", annotation.name)


def get_type(node):
    """
    Tries to infer the type of a node (using astroid type inference).
    :param node: node to infer from.
    :return: inferred type or None if it couldn't be inferred.
    """
    # Assume there is only one type inferred
    # If there are multiple types inferred we have to
    # choose which one to pick
    try:
        if len(node.inferred()) > 0:
            ty_infer = node.inferred()[0]
            if isinstance(ty_infer, Module):
                ty = ty_infer.name
            elif isinstance(ty_infer, ClassDef):
                ty = ty_infer.name
            elif isinstance(ty_infer, type(Uninferable)):
                ty = None
            else:
                ty = ty_infer.pytype().replace("builtins.", "").lstrip(".")
        else:
            ty = None
    except Exception as err:
        ty = None

    return ty


def parse_terminal(node, ty=None):
    """
    Parses a terminal with the correct type.
    :param node: node to create terminal from.
    :param ty: type to annotate terminal with.
    :return: Terminal object.
    """
    if ty == "METHOD_CALL":
        return Terminal("var", None)
    elif ty:
        return Terminal("var", ty)
    else:
        return Terminal("var", get_type(node))
