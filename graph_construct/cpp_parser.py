from typing import List, Dict

import tree_sitter
import dgl
import torch

ATOM_TYPES = ("init_declarator", "assignment_expression", "condition_clause", "binary_expression", "return_statement",
              "call_expression", "update_expression")
COMPARE = (">", ">=", "<", "<=", "==")

DIRECT_LINK = 0
LAST_READ = 1
LAST_WRITE = 2
COMPUTED_FROM = 3
EDGE_KIND = 4


class Node:
    def __init__(self, node: tree_sitter.Node, father=None):
        self.node = node
        self.father = father
        self.idx = 0

        self.children = list([])
        self.named_children = list([])
        self.is_named_leaf = True
        self.type = node.type

        for child in node.children:
            if child.type != "comment" and child.type != "preproc_arg":
                if child.type == "if_statement":
                    self.children.append(IfStatement(node, father))
                else:
                    self.children.append(Node(child, self))
                if child.is_named:
                    self.is_named_leaf = False
                    self.named_children.append(self.children[-1])

        self.last_write = set()
        self.last_read = set()
        self.computed_from = list([])
        self.return_stmt = list([])
        self.method_implementation = None

        if self.father is None:
            self.assign_idx(0)

    def assign_idx(self, idx):
        self.idx = idx
        last_idx = idx
        for child in self.children:
            last_idx = child.assign_idx(last_idx + 1)
        return last_idx

    @staticmethod
    def _get_text(start_point, end_point, text):
        if start_point[0] == end_point[0]:
            expression = text[start_point[0]][start_point[1]:end_point[1]]
        else:
            expression = " " + text[start_point[0]][start_point[1]:]
            for i in range(start_point[0] + 1, end_point[0]):
                expression += " " + text[i]
            expression += " " + text[end_point[0]][:end_point[1]]
        return expression

    def get_text(self, text: list):
        return Node._get_text(self.node.start_point, self.node.end_point, text)

    def str(self, text: list):
        return f"id: {self.idx} " + self.node.type + " " + self.get_text(text)

    def traverse(self, depth, text: list):
        print("…" * depth + self.node.type + " " + self.get_text(text) + f" id: {self.idx}")
        for node in self.children:
            node.traverse(depth + 1, text)

    def get_named_leaf(self):
        if self.is_named_leaf:
            return [self]
        else:
            leaves = list([])
            for child in self.named_children:
                leaves += child.get_named_leaf()
            return leaves


class NormalNode(Node):
    def __init__(self, node: tree_sitter.Node, father=None):
        super(NormalNode, self).__init__(node=node, father=father)


class CompoundStatement(Node):
    def __init__(self, node: tree_sitter.Node, father=None):
        super(CompoundStatement, self).__init__(node=node, father=father)


class IfStatement(Node):
    def __init__(self, node: tree_sitter.Node, father=None):
        super(IfStatement, self).__init__(node=node, father=father)
        ptr = 0
        n_children = len(self.children)

        # find the conditional clause
        self.condition_clause = None
        while ptr < n_children:
            if self.children[ptr].type == "condition_clause":
                self.condition_clause = self.children[ptr]
                break
            else:
                ptr += 1
        if self.condition_clause is None:
            raise ValueError(f"Parse Error: No condition clause in node {self.node.start_point, self.node.end_point}")
        else:
            self.condition_clause: Node

        self.branches = list([])
        consequence = self.node.child_by_field_name("consequence")
        if consequence is None:
            raise ValueError(f"Parse Error: No consequence in node {self.node.start_point, self.node.end_point}")
        alternative = self.node.child_by_field_name("alternative")

        while ptr < n_children:
            if self.children[ptr] == consequence or self.children[ptr] == alternative:  # can't use 'is'
                self.branches.append(self.children[ptr])
            ptr += 1


class ForStatement(Node):
    def __init__(self, node: tree_sitter.Node, father=None):
        super(ForStatement, self).__init__(node=node, father=father)

        initializer = self.node.child_by_field_name("initializer")
        self.initializer = None
        if initializer is not None:
            for child in self.named_children:
                if initializer == child.node:  # you can't use 'is' here because they have different ids
                    self.initializer = child
                    break
            self.initializer: Node

        condition = self.node.child_by_field_name("condition")
        self.condition = None
        if condition is not None:
            for child in self.named_children:
                if condition == child.node:
                    self.condition = child
                    break
            self.condition: Node

        update = self.node.child_by_field_name("update")
        self.update = None
        if update is not None:
            for child in self.named_children:
                if child == update:
                    self.update = child
                    break
            self.update: None

        self.loop_body = self.children[-1]


class WhileStatement(Node):
    def __init__(self, node: tree_sitter.Node, father=None):
        super(WhileStatement, self).__init__(node=node, father=father)
        assert self.node.child_by_field_name("condition_clause") == self.children[1].node
        self.condition_clause = self.children[1]
        assert self.node.child_by_field_name("body") == self.children[2].node
        self.loop_body = self.children[2]


class DoStatement(Node):
    def __init__(self, node: tree_sitter.Node, father=None):
        super(DoStatement, self).__init__(node=node, father=father)
        self.body = self.children[1]
        self.condition = self.children[3]


class FuncDefinition(Node):
    def __init__(self, node: tree_sitter.Node, father=None):
        super(FuncDefinition, self).__init__(node=node, father=father)
        declarator = node.child_by_field_name("declaration")
        if declarator is None:
            raise ValueError(f"Error: declarator of {node.start_point, node.end_point} is None")
        else:
            declarator: tree_sitter.Node
            self.declarator = None
            for child in self.children:
                if child.node == declarator:
                    self.declarator = child
                    break
            self.declarator: Node

        if self.declarator.node.type != "function_declarator":
            raise ValueError(f"Declarator of {node.start_point, node.end_point} is {self.declarator.node.type}")
        else:
            func_declarator = self.declarator.node.child_by_field_name("declarator")
            parameter_list = self.declarator.node.child_by_field_name("parameters")
            if func_declarator is None:
                raise ValueError(f"No declarator of {node.start_point, node.end_point}")
            else:
                for child in self.declarator.children:
                    if child.node == func_declarator:
                        self.func_declarator = child

            if parameter_list is None:
                raise ValueError(f"No parameter list of {node.start_point, node.end_point}")
            else:
                for child in self.declarator.children:
                    if child.node == parameter_list:
                        self.parameter_list = parameter_list


