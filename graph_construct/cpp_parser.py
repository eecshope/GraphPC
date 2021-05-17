from typing import List
from .variable_table import VariableTable
import tree_sitter
import abc

ATOM_TYPES = ("init_declarator", "array_declarator", "parameter_declaration", "function_declarator",
              "expression_statement", "pointer_declarator",  "declaration", "condition_declaration",
              "return_statement", "condition_clause", "argument_list", "parameter_list", "initializer_list",
              "initializer_pair", "translation_unit",
              "comma_expression", "assignment_expression", "binary_expression", "call_expression", "update_expression",
              "subscript_expression", "conditional_expression", "identifier")


COMPARE = (">", ">=", "<", "<=", "==")

MODE = ("r", "w", "d", "o")

DIRECT_LINK = 0
LAST_READ = 1
LAST_WRITE = 2
COMPUTED_FROM = 3
EDGE_KIND = 4


def special_array(node, vt):
    _name = node.named_children[0]
    _index = node.named_children[1]

    if _index.type == "identifier":
        _index_unit = vt.find_reference(_index.token)
        _index.last_write |= _index_unit["lw"]
        _index.last_read |= _index_unit["lr"]
        _index_unit["lr"] = {_index}
    else:
        _index.simulate_data_flow(vt)

    if _name.type == "identifier":
        _name_unit = vt.find_reference(_name.token)
        _name.last_write |= _name_unit["lw"]
        _name.last_read |= _name_unit["lr"]
        _name_unit["lw"] = {_name}
    else:
        special_array(_name, vt)


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
    def simulate_data_flow(self, variable_table: VariableTable, mode):
        pass


class NormalNode(Node):
    def __init__(self, node: tree_sitter.Node, code: List, father=None):
        super(NormalNode, self).__init__(node=node, code=code, father=father)

    def simulate_data_flow(self, variable_table: VariableTable, mode, table_verbose=False):
        if mode not in MODE:
            raise ValueError(f"mode '{mode}' is not available for data flow simulation")
        try:
            if self.type not in ATOM_TYPES:
                for child in self.named_children:
                    child.simulate_data_flow(variable_table, mode)

            elif self.type == "translation_unit":
                for child in self.named_children:
                    child.simulate_data_flow(variable_table, "o")  # "d" is the default status

            elif self.type == "declaration":  # Done
                for child in self.named_children:
                    child.simulate_data_flow(variable_table, "d")

            elif self.type == "pointer_declarator":  # Done
                declarator = self.children[-1]
                declarator.simulate_data_flow(variable_table=variable_table, mode="d")

            elif self.type == "init_declarator":  # Done
                self.named_children[0].simulate_data_flow(variable_table=variable_table, mode="d")
                self.named_children[1].simulate_data_flow(variable_table=variable_table, mode="r")

            elif self.type == "array_declarator":  # Done
                self.named_children[0].simulate_data_flow(variable_table=variable_table, mode="d")

            elif self.type == "parameter_declaration":  # done
                for named_child in self.named_children:
                    if "_declarator" in named_child.type:
                        named_child.simulate_data_flow(variable_table, "d")

            elif self.type == "function_declarator":  # Done
                self.children[1].simulate_data_flow(variable_table=variable_table, mode="d")

            elif self.type == "condition_declaration":
                self.named_children[1].simulate_data_flow(variable_table=variable_table, mode="d")
                for child in self.named_children[2:]:
                    child.simulate_data_flow(variable_table=variable_table, mode="r")

            elif self.type == "argument_list":  # Done
                for named_child in self.named_children:
                    named_child.simulate_data_flow(variable_table, mode)

            elif self.type == "parameter_list":  # Done
                for named_child in self.named_children:
                    named_child.simulate_data_flow(variable_table, "d")

            elif self.type == "initializer_list":  # Done
                for named_child in self.named_children:
                    named_child.simulate_data_flow(variable_table, "r")

            elif self.type == "initializer_pair":  # Done
                self.named_children[0].simulate_data_flow(variable_table, mode)
                self.named_children[1].simulate_data_flow(variable_table, "r")

            elif self.type == "expression_statement":  # Done
                if mode == "o":
                    declare_expression = ("identifier", "subscript_expression", "comma_expression")  # comma may error
                    for expression in self.named_children:
                        if expression.type in declare_expression:
                            expression.simulate_data_flow(variable_table, "d")
                        else:
                            expression.simulate_data_flow(variable_table, "r")
                elif mode == "r":
                    for expression in self.named_children:
                        expression.simulate_data_flow(variable_table, "r")
                else:
                    raise ValueError(f"Expression statement encounter mode {mode}")

            elif self.type == "comma_expression":  # Done
                if mode == "r":
                    for expression in self.named_children:
                        expression.simulate_data_flow(variable_table, "r")

                elif mode == "o":
                    declare_expression = ("identifier", "subscript_expression", "comma_expression")
                    for expression in self.named_children:
                        if expression.type in declare_expression:
                            expression.simulate_data_flow(variable_table, "d")
                        else:
                            expression.simulate_data_flow(variable_table, "r")
                else:
                    raise ValueError(f"Comma expression encounter mode {mode}")

            elif self.type == "assignment_expression":  # Done
                # proc right
                self.children[2].simulate_data_flow(variable_table, "r")  # i = ++i; is undefined and it is not allowed

                variable = self.children[0]
                support_expression = ("identifier", "subscript_expression", "pointer_expression")
                if variable.type in support_expression:
                    variable.simulate_data_flow(variable_table, "w")
                else:
                    raise ValueError(f"{variable.type} not implemented for assignment_expression {self.token}")

            elif self.type == "update_expression":  # done
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

            elif self.type == "binary_expression":  # Done
                self.children[0].simulate_data_flow(variable_table, "r")
                self.children[2].simulate_data_flow(variable_table, "r")

            elif self.type == "call_expression":  # Done
                self.children[1].simulate_data_flow(variable_table, "r")

            elif self.type == "condition_clause":  # Done
                if len(self.named_children) == 2: # init, value
                    init = self.named_children[0]
                    if init.type == "declarator":
                        init.simulate_data_flow(variable_table=variable_table, mode="d")
                    else:
                        init.simulate_data_flow(variable_table=variable_table, mode="r")
                    value = self.named_children[1]
                    value.simulate_data_flow(variable_table=variable_table, mode="r")
                else:
                    value = self.named_children[0]  # conditional declaration
                    value.simulate_data_flow(variable_table=variable_table, mode="d")

            elif self.type == "subscript_expression":  # Done
                name = self.named_children[0]
                index = self.named_children[1]

                index.simulate_data_flow(variable_table, "r")

                if mode == "r":
                    name.simulate_data_flow(variable_table, "r")
                elif mode == "w":
                    name.simulate_data_flow(variable_table, "w")
                else:
                    raise ValueError(f"Subscript expression {self.token} does not support declaration")

            elif self.type == "conditional_expression":  # a ? b : c
                condition = self.children[0]
                consequence = self.children[2]
                alternative = self.children[4]

                if_table = variable_table.add_variable_table()
                condition.simulate_data_flow(if_table, "r")

                consequence_table = VariableTable(if_table)
                consequence.simulate_data_flow(consequence_table, "r")

                alternative_table = VariableTable(if_table)
                alternative.simulate_data_flow(alternative_table, "r")

                if_table.child = consequence_table
                consequence_table.merge_and_pop_self()
                if_table.child = alternative_table
                alternative_table.add_variable_table()
                if_table.pop_self()

            elif self.type == "identifier":
                if mode == "r":
                    variable_table.find_and_update(self.token, self, "ro")
                elif mode == "w":
                    variable_table.find_and_update(self.token, self, "wo")
                elif mode == "d":
                    variable_table.add_reference(self.token, self)
                else:
                    raise ValueError(f"Something Wrong {mode}")

        except Exception as e:
            print(e)
            print(f"Error: Node {self.idx, self.token}")
            print(variable_table)
            raise ValueError("Parse Failed")

        # print(f"idx: {self.idx}, table: {variable_table}")


class CompoundStatement(Node):
    def __init__(self, node: tree_sitter.Node, code: List, father=None):
        super(CompoundStatement, self).__init__(node=node, code=code, father=father)

    def simulate_data_flow(self, variable_table: VariableTable, mode):
        variable_table = variable_table.add_variable_table()
        for named_child in self.named_children:
            named_child.simulate_data_flow(variable_table=variable_table, mode=mode)
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

    def simulate_data_flow(self, variable_table: VariableTable, mode):
        if_table = variable_table.add_variable_table()
        self.condition_clause.simulate_data_flow(if_table, mode=mode)

        branch_tables = list([])
        for branch in self.branches:
            branch_table = VariableTable(if_table)
            branch.simulate_data_flow(branch_table, mode=mode)
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
            self.update: Node

        self.loop_body = self.children[-1]

    def simulate_data_flow(self, variable_table: VariableTable, mode):
        for_table = variable_table.add_variable_table()

        if self.initializer is not None:
            self.initializer.simulate_data_flow(for_table, mode=mode)

        if self.condition is not None:
            self.condition.simulate_data_flow(for_table, mode=mode)

        self.loop_body.simulate_data_flow(for_table, mode=mode)

        if self.update is not None:
            self.update = self.update.simulate_data_flow(for_table, mode=mode)

        if self.condition is not None:
            self.condition.simulate_data_flow(for_table, mode=mode)

        self.loop_body.simulate_data_flow(for_table, mode=mode)

        for_table.pop_self()


class WhileStatement(Node):
    def __init__(self, node: tree_sitter.Node, code: List, father=None):
        super(WhileStatement, self).__init__(node=node, code=code, father=father)
        assert self.node.child_by_field_name("condition_clause") == self.children[1].node
        self.condition_clause = self.children[1]
        assert self.node.child_by_field_name("body") == self.children[2].node
        self.loop_body = self.children[2]

    def simulate_data_flow(self, variable_table: VariableTable, mode):
        while_table = variable_table.add_variable_table()

        self.condition_clause.simulate_data_flow(while_table, mode=mode)
        self.loop_body.simulate_data_flow(while_table, mode=mode)
        self.condition_clause.simulate_data_flow(while_table, mode=mode)
        self.loop_body.simulate_data_flow(while_table, mode=mode)

        while_table.pop_self()


class DoStatement(Node):
    def __init__(self, node: tree_sitter.Node, code: List, father=None):
        super(DoStatement, self).__init__(node=node, code=code, father=father)
        self.body = self.children[1]
        self.condition = self.children[3]

    def simulate_data_flow(self, variable_table: VariableTable, mode):
        do_table = variable_table.add_variable_table()

        self.body.simulate_data_flow(do_table, mode=mode)
        self.condition.simulate_data_flow(do_table, mode=mode)
        self.body.simulate_data_flow(do_table, mode=mode)
        self.condition.simulate_data_flow(do_table, mode=mode)

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
                        self.parameter_list = child

        body = node.child_by_field_name("body")
        self.body = None
        if body is not None:
            for child in self.children:
                if child.node == body:
                    self.body = child
                    break

    def simulate_data_flow(self, variable_table: VariableTable, mode):
        func_table = variable_table.add_variable_table()

        parameters = self.parameter_list.get_named_leaf()
        for parameter in parameters:
            if parameter.type == "identifier":
                token = parameter.token
                unit = func_table.add_reference(parameter.token, parameter)
                unit["lr"].add(parameter)
                unit["lw"].add(parameter)

        if self.body is not None:
            self.body.simulate_data_flow(func_table, mode=mode)

        func_table.pop_self()
