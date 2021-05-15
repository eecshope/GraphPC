from typing import List
from variable_table import VariableTable
import tree_sitter
import abc

ATOM_TYPES = ("init_declarator", "assignment_expression", "condition_clause", "binary_expression", "return_statement",
              "call_expression", "update_expression", "declaration")
COMPARE = (">", ">=", "<", "<=", "==")

DIRECT_LINK = 0
LAST_READ = 1
LAST_WRITE = 2
COMPUTED_FROM = 3
EDGE_KIND = 4


class Node:
    def __init__(self, node: tree_sitter.Node, code: List, father=None):
        self.node = node
        self.father = father
        self.idx = 0
        self.token = self.get_text(code)

        self.children = list([])
        self.named_children = list([])
        self.is_named_leaf = True
        self.type = node.type

        for child in node.children:
            if child.type != "comment" and child.type != "preproc_arg":
                if child.type == "if_statement":
                    self.children.append(IfStatement(child, code, self))
                elif child.type == "for_statement":
                    self.children.append(ForStatement(child, code, self))
                elif child.type == "while_statement":
                    self.children.append(WhileStatement(child, code, self))
                elif child.type == "do_statement":
                    self.children.append(DoStatement(child, code, self))
                elif child.type == "function_definition":
                    self.children.append(FuncDefinition(child, code, self))
                else:
                    self.children.append(NormalNode(child, code, self))
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

    def traverse(self, depth):
        print("…" * depth + self.node.type + " " + self.token + f" id: {self.idx}")
        for node in self.children:
            node.traverse(depth + 1)

    def get_named_leaf(self):
        if self.is_named_leaf:
            return [self]
        else:
            leaves = list([])
            for child in self.named_children:
                leaves += child.get_named_leaf()
            return leaves

    @abc.abstractmethod
    def simulate_data_flow(self, variable_table: VariableTable):
        pass


class NormalNode(Node):
    def __init__(self, node: tree_sitter.Node, code: List, father=None):
        super(NormalNode, self).__init__(node=node, code=code, father=father)

    def simulate_data_flow(self, variable_table: VariableTable):
        if self.type not in ATOM_TYPES:
            for child in self.named_children:
                child.simulate_data_flow(variable_table)

        elif self.type == "declaration":
            for child in self.named_children:
                if child.type == "init_declaration":
                    child.simulate_data_flow(variable_table)
                elif child.type == "identifier":
                    unit = variable_table.add_reference(child.token)
                    unit["lr"].add(child)
                    unit["lw"].add(child)

        elif self.type == "init_declarator":
            from_variables = [e for e in self.children[0].get_named_leaf() if e.type == "identifier"]
            to_variables = [e for e in self.children[2].get_named_leaf() if e.type == "identifier"]

            for variable in from_variables:
                unit = variable_table.add_reference(variable.token)
                unit["lw"].add(variable)

            for variable in to_variables:
                unit = variable_table.find_reference(variable.token)
                variable.last_read |= unit["lr"]
                variable.last_write |= unit["lw"]
                unit["lr"] = {variable}

        elif self.type == "assignment_expression":
            # proc right
            left_variables = [e for e in self.children[0].get_named_leaf() if e.type == "identifier"]
            right_variables = [e for e in self.children[2].get_named_leaf() if e.type == "identifier"]

            self.children[2].simulate_data_flow(variable_table)  # i = ++i; is undefined and it is not allowed

            for variable in left_variables:
                unit = variable_table.find_reference(variable.token)
                variable.last_read |= unit["lr"]
                variable.last_write |= unit["lw"]
                unit["lw"] = {variable}

            for variable in right_variables:
                unit = variable_table.find_reference(variable.token)
                variable.last_read |= unit["lr"]
                variable.last_write |= unit["lw"]
                unit["lr"] = {variable}

        elif self.type == "update_expression":
            if self.children[0].node == self.node.child_by_field_name("operator"):
                variable = self.children[1].get_named_leaf()
            else:
                variable = self.children[0].get_named_leaf()

            unit = variable_table.find_reference(variable[0].token)
            variable[0].last_read |= unit["lr"]
            variable[0].last_read.add(variable[0])
            variable[0].last_write |= unit["lw"]
            variable[0].last_write.add(variable[0])
            unit["lr"] = {variable[0]}
            unit["lw"] = {variable[0]}

        elif self.type == "call_expression":
            arguments = self.node.child_by_field_name("arguments")
            find_argument = False
            for named_child in self.named_children:
                if named_child.node == arguments:
                    arguments = named_child
                    find_argument = True

            if not find_argument:
                raise ValueError(f"{self.token} has no arguments")

            variables = arguments.get_named_leaf()
            for v in variables:
                unit = variable_table.find_reference(v.token)
                v.last_write |= unit["lw"]
                v.last_read |= unit["lr"]
                unit["lr"] = {v}

        else:
            leaves = [e for e in self.get_named_leaf() if e.type == "identifier"]
            for leaf in leaves:
                unit = variable_table.find_reference(leaf.token)
                leaf.last_read |= unit["lr"]
                leaf.last_write |= unit["lw"]
                unit["lr"] = {leaf}
                unit["lw"] = {leaf}


class CompoundStatement(Node):
    def __init__(self, node: tree_sitter.Node, code: List, father=None):
        super(CompoundStatement, self).__init__(node=node, code=code, father=father)

    def simulate_data_flow(self, variable_table: VariableTable):
        variable_table = variable_table.add_variable_table()
        for named_child in self.named_children:
            named_child.simulate_data_flow(variable_table=variable_table)
        variable_table.pop_self()


class IfStatement(Node):
    def __init__(self, node: tree_sitter.Node, code: List, father=None):
        super(IfStatement, self).__init__(node=node, code=code, father=father)
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
            if self.children[ptr].node == consequence or self.children[ptr].node == alternative:  # can't use 'is'
                self.branches.append(self.children[ptr])
            ptr += 1

    def simulate_data_flow(self, variable_table: VariableTable):
        if_table = variable_table.add_variable_table()
        self.condition_clause.simulate_data_flow(if_table)

        branch_tables = list([])
        for branch in self.branches:
            branch_table = VariableTable(if_table)
            branch.simulate_data_flow(branch_table)
            branch_tables.append(branch_table)

        for branch_table in branch_tables:
            if_table.child = branch_table
            branch_table.merge_and_pop_self()

        if_table.pop_self()


class ForStatement(Node):
    def __init__(self, node: tree_sitter.Node, code: List, father=None):
        super(ForStatement, self).__init__(node=node, code=code, father=father)

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
                if child.node == update:
                    self.update = child
                    break
            self.update: None

        self.loop_body = self.children[-1]

    def simulate_data_flow(self, variable_table: VariableTable):
        for_table = variable_table.add_variable_table()

        if self.initializer is not None:
            self.initializer.simulate_data_flow(for_table)

        if self.condition is not None:
            self.condition.simulate_data_flow(for_table)

        self.loop_body.simulate_data_flow(for_table)

        if self.condition is not None:
            self.condition.simulate_data_flow(for_table)

        self.loop_body.simulate_data_flow(for_table)

        for_table.pop_self()


class WhileStatement(Node):
    def __init__(self, node: tree_sitter.Node, code: List, father=None):
        super(WhileStatement, self).__init__(node=node, code=code, father=father)
        assert self.node.child_by_field_name("condition_clause") == self.children[1].node
        self.condition_clause = self.children[1]
        assert self.node.child_by_field_name("body") == self.children[2].node
        self.loop_body = self.children[2]

    def simulate_data_flow(self, variable_table: VariableTable):
        while_table = variable_table.add_variable_table()

        self.condition_clause.simulate_data_flow(while_table)
        self.loop_body.simulate_data_flow(while_table)
        self.condition_clause.simulate_data_flow(while_table)
        self.loop_body.simulate_data_flow(while_table)

        while_table.pop_self()


class DoStatement(Node):
    def __init__(self, node: tree_sitter.Node, code: List, father=None):
        super(DoStatement, self).__init__(node=node, code=code, father=father)
        self.body = self.children[1]
        self.condition = self.children[3]

    def simulate_data_flow(self, variable_table: VariableTable):
        do_table = variable_table.add_variable_table()

        self.body.simulate_data_flow(do_table)
        self.condition.simulate_data_flow(do_table)
        self.body.simulate_data_flow(do_table)
        self.condition.simulate_data_flow(do_table)

        do_table.pop_self()


class FuncDefinition(Node):
    def __init__(self, node: tree_sitter.Node, code: List, father=None):
        super(FuncDefinition, self).__init__(node=node, code=code, father=father)
        declarator = node.child_by_field_name("declarator")
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

        body = node.child_by_field_name("body")
        self.body = None
        if body is not None:
            for child in self.children:
                if child.node == body:
                    self.body = child
                    break

    def simulate_data_flow(self, variable_table: VariableTable):
        func_table = variable_table.add_variable_table()

        parameters = self.parameter_list.get_named_leaf()
        for parameter in parameters:
            if parameter.type == "identifier":
                token = parameter.token
                unit = func_table.add_reference(token)
                unit["lr"].add(parameter)
                unit["lw"].add(parameter)

        if self.body is not None:
            self.body.simulate_data_flow(func_table)

        func_table.pop_self()
