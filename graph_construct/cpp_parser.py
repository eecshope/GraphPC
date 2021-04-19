from typing import List

import tree_sitter

ATOM_TYPES = ("init_declarator", "assignment_expression", "condition_clause", "binary_expression")
COMPARE = (">", ">=", "<", "<=", "==")


class Node:
    def __init__(self, node: tree_sitter.Node, father):
        self.node = node
        self.father = father

        self.direct_next = list([])
        self.is_leaf = True
        for child in node.children:
            if child.is_named:
                self.is_leaf = False
                self.direct_next.append(Node(child, self))

        self.last_write = set()
        self.last_read = set()
        self.computed_from = list([])

    def get_text(self, text):
        sp = self.node.start_point[1]
        ep = self.node.end_point[1]
        expression = text[sp:ep]
        return expression

    def str(self, text):
        return self.node.type + " " + self.get_text(text)

    def traverse(self, depth, text):
        print("…" * depth + self.node.type + " " + self.get_text(text))
        for node in self.direct_next:
            node.traverse(depth+1, text)

    def get_leaf(self):
        if self.is_leaf:
            return [self]
        else:
            leaves = list([])
            for child in self.direct_next:
                leaves += child.get_leaf()
            return leaves

    def get_computed_from(self):
        if self.node.type == "init_declarator" or self.node.type == "assignment_expression":
            assert len(self.direct_next) == 2
            left_nodes = self.direct_next[0].get_leaf()
            right_nodes = self.direct_next[1].get_leaf()

            for node in left_nodes:
                node.computed_from = right_nodes


def simulate_data_flow(node: Node, text: str, table: List):

    def merge(father_scope, child_scope, _merge=False):
        for var in child_scope:  # var: a tuple (name, type)
            if var[1] == "l":  # local declared
                continue
            else:  # inherent from outer scope
                if (var[0], "l") in father_scope:  # local declared has more priority
                    if _merge:
                        father_scope[(var[0], "l")]["lr"] |= child_scope[var]["lr"]
                        father_scope[(var[0], "l")]["lw"] |= child_scope[var]["lw"]
                    else:
                        father_scope[(var[0], "l")]["lr"] = child_scope[var]["lr"]
                        father_scope[(var[0], "l")]["lw"] = child_scope[var]["lw"]
                elif var not in father_scope:
                    father_scope[var] = child_scope[var]
                else:
                    if _merge:
                        father_scope[var]["lr"] |= child_scope[var]["lr"]
                        father_scope[var]["lw"] |= child_scope[var]["lw"]
                    else:
                        father_scope[var]["lr"] = child_scope[var]["lr"]
                        father_scope[var]["lw"] = child_scope[var]["lw"]

    if node.node.type == "compound_statement":
        table.append(dict({}))

        for child_node in node.direct_next:
            simulate_data_flow(child_node, text, table)

        if len(table) >= 2:
            merge(table[-2], table[-1])
            table.pop(-1)

    elif node.node.type == "while_statement" or node.node.type == "for_statement":
        table.append(dict({}))

        for child_node in node.direct_next:
            simulate_data_flow(child_node, text, table)
        for child_node in node.direct_next:
            simulate_data_flow(child_node, text, table)

        if len(table) >= 2:
            merge(table[-2], table[-1])
            table.pop(-1)

    elif node.node.type == "if_statement":
        table.append(dict({}))
        scope_cache = list([])
        for child_node in node.direct_next:
            if child_node.node.type == "condition_clause":
                simulate_data_flow(child_node, text, table)
            else:
                table.append(dict({}))
                simulate_data_flow(child_node, text, table)
                scope_cache.append(table[-1])
                table.pop(-1)
        for scope in scope_cache:
            merge(table[-1], scope, True)
        if len(table) >= 2:
            merge(table[-2], table[-1])
            table.pop(-1)

    elif node.node.type in ATOM_TYPES:

        def find_declare_scope(node_token):
            _declared_var = None
            if (node_token, "l") in table[-1]:
                return table[-1][(node_token, "l")]
            elif (node_token, "g") in table[-1]:
                return table[-1][(node_token, "g")]

            for _scope in reversed(table):
                if (node_token, "l") in _scope:
                    _declared_var = _scope[(node_token, "l")]
                    break
                elif (node_token, "g") in _scope:
                    _declared_var = _scope[(node_token, "g")]
                    break
            # make sure that it is declared before use
            if _declared_var is None:
                raise ValueError(f"Identifier {node_token} used before declared with atom {node.str(text)}")
            else:
                table[-1][(node_token, "g")] = _declared_var
                return _declared_var

        # self update, like ++x, x++, will be dealt with separately
        if node.node.type == "init_declarator" or node.node.type == "assignment_expression":
            assert len(node.direct_next) == 2
            # get the left value and the right values
            right_nodes = [node for node in node.direct_next[1].get_leaf() if node.node.type == "identifier"]
            left_nodes = [node for node in node.direct_next[0].get_leaf() if node.node.type == "identifier"]

            # deal with the right node
            for right_node in right_nodes:
                token = right_node.get_text(text)
                # find declaration
                declared_var = find_declare_scope(token)
                if declared_var["lr"] is not None:
                    right_node.last_read.add(declared_var["lr"])
                if declared_var["lw"] is not None:
                    right_node.last_write.add(declared_var["lw"])
                declared_var["lr"] = {right_node}

            # deal with the left node
            for left_node in left_nodes:
                token = left_node.get_text(text)
                if node.node.type == "init_declarator":
                    table[-1][(token, "l")] = {"lr": None, "lw": left_node}
                else:
                    declared_var = find_declare_scope(token)
                    if declared_var["lr"] is not None:
                        left_node.last_read.add(declared_var["lr"])
                    if declared_var["lw"] is not None:
                        left_node.last_write.add(declared_var["lw"])
                    declared_var["lw"] = {left_node}
        else:
            expression = node.get_text(text)
            valid = False
            for compare in COMPARE:
                if compare in expression:
                    valid = True
                    break
            if valid:
                terminals = [t_node for t_node in node.get_leaf() if t_node.node.type == "identifier"]
                for terminal in terminals:
                    token = terminal.get_text(text)
                    declared_var = find_declare_scope(token)
                    if declared_var["lr"] is not None:
                        terminal.last_read.add(declared_var["lr"])
                    if declared_var["lw"] is not None:
                        terminal.last_write.add(declared_var["lw"])
                    declared_var["lr"] = {terminal}

    else:
        for child_node in node.direct_next:
            simulate_data_flow(child_node, text, table)
