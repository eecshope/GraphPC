from typing import List

import tree_sitter

ATOM_TYPES = ("init_declarator", "assignment_expression", "condition_clause", "binary_expression", "return_statement",
              "call_expression", "update_expression")
COMPARE = (">", ">=", "<", "<=", "==")


class Node:
    def __init__(self, node: tree_sitter.Node, father=None):
        self.node = node
        self.father = father
        self.idx = 0

        self.direct_next = list([])
        self.is_leaf = True

        for child in node.children:
            if child.is_named:
                self.is_leaf = False
                self.direct_next.append(Node(child, self))

        self.last_write = set()
        self.last_read = set()
        self.computed_from = list([])

        if self.father is None:
            self.assign_idx(0)

    def assign_idx(self, idx):
        self.idx = idx
        last_idx = idx
        for child in self.direct_next:
            last_idx = child.assign_idx(last_idx + 1)
        return last_idx

    def get_text(self, text):
        sp = self.node.start_point[1]
        ep = self.node.end_point[1]
        expression = text[sp:ep]
        return expression

    def str(self, text):
        return f"id: {self.idx} " + self.node.type + " " + self.get_text(text)

    def traverse(self, depth, text):
        print("…" * depth + self.node.type + " " + self.get_text(text) + f" id: {self.idx}")
        for node in self.direct_next:
            node.traverse(depth + 1, text)

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
                    father_scope[var] = {"lr": child_scope[var]["lr"], "lw": child_scope[var]["lw"]}
                else:
                    if _merge:
                        father_scope[var]["lr"] |= child_scope[var]["lr"]
                        father_scope[var]["lw"] |= child_scope[var]["lw"]
                    else:
                        father_scope[var]["lr"] = child_scope[var]["lr"]
                        father_scope[var]["lw"] = child_scope[var]["lw"]

    if node.node.type == "function_definition":
        assert node.direct_next[1].node.type == "function_declarator"
        function_declarator = node.direct_next[1]
        assert function_declarator.direct_next[1].node.type == "parameter_list"
        parameter_list = function_declarator.direct_next[1]
        table.append({})
        for p in [n for n in parameter_list.get_leaf() if n.node.type == "identifier"]:
            table[-1][(p.get_text(text), "l")] = {"lr": {p}, "lw": {p}}
        assert node.direct_next[2].node.type == "compound_statement"
        simulate_data_flow(node.direct_next[2], text, table)
        table.pop(-1)

    elif node.node.type == "compound_statement":
        table.append(dict({}))

        for child_node in node.direct_next:
            simulate_data_flow(child_node, text, table)

        if len(table) >= 2:
            merge(table[-2], table[-1])
            table.pop(-1)

    elif node.node.type == "for_statement":
        table.append(dict({}))

        meta = list([])

        for i, c in enumerate(node.node.children):
            meta.append(c)
            sp = c.start_point[1]
            ep = c.end_point[1]
            if text[sp:ep] == ")":
                break
        meta = meta[2:-1]

        loop_start = 0
        if meta[0].is_named:
            for_init = None
            for i in range(loop_start, len(node.direct_next)):
                if node.direct_next[i].node is meta[0]:
                    for_init = node.direct_next[i]
                    loop_start = i + 1
                    break
        else:
            for_init = None

        if meta[1].is_named:
            for_cond = None
            incre_idx = 3
            for i in range(loop_start, len(node.direct_next)):
                if node.direct_next[i].node is meta[1]:
                    for_cond = node.direct_next[i]
                    loop_start = i + 1
                    break
        else:
            for_cond = None
            incre_idx = 2

        if meta[incre_idx].is_named:
            for_incre = None
            for i in range(loop_start, len(node.direct_next)):
                if node.direct_next[i].node is meta[incre_idx]:
                    for_incre = node.direct_next[i]
                    loop_start = i + 1
                    break
        else:
            for_incre = None

        if len(node.direct_next[loop_start:]) == 0:
            raise RuntimeError(f"For Loop parsed error at {node.str(text)}")

        if for_init is not None:
            simulate_data_flow(for_init, text, table)

        for i in range(2):
            if for_cond is not None:
                simulate_data_flow(for_cond, text, table)
            for child_node in node.direct_next[loop_start:]:
                simulate_data_flow(child_node, text, table)

            if for_incre is not None:
                simulate_data_flow(for_incre, text, table)

        if len(table) >= 2:
            merge(table[-2], table[-1])
            table.pop(-1)

    elif node.node.type == "if_statement":
        table.append(dict({}))
        scope_cache = list([])
        for child_node in node.direct_next:
            if child_node.node.type == "condition_clause":
                table.append(dict({}))
                simulate_data_flow(child_node, text, table)
                scope_cache.append(table[-1])
                merge(table[-2], table[-1])
                table.pop(-1)
            else:
                table.append(dict({}))
                simulate_data_flow(child_node, text, table)
                scope_cache.append(table[-1])
                table.pop(-1)
        for scope in scope_cache:
            merge(table[-1], scope, True)
        if len(table) >= 2:
            merge(table[-2], table[-1], True)
            table.pop(-1)

    elif node.node.type == "declaration":
        for child in node.direct_next:
            if child.node.type == "identifier":
                table[-1][(child.get_text(text), "l")] = {"lr": {child}, "lw": {child}}
            else:
                simulate_data_flow(child, text, table)

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
                # raise ValueError(f"Identifier {node_token} used before declared with atom {node.str(text)}")
                # It's from the outside
                table[-1][(node_token, "g")] = {"lw": set(), "lr": set()}
            else:
                table[-1][(node_token, "g")] = {"lw": _declared_var["lw"], "lr": _declared_var["lr"]}

            return table[-1][(node_token, "g")]

        # self update, like ++x, x++, will be dealt with separately
        if node.node.type == "init_declarator" or node.node.type == "assignment_expression":
            assert len(node.direct_next) == 2
            # get the left value and the right values
            right_nodes = [node for node in node.direct_next[1].get_leaf() if node.node.type == "identifier"]
            left_nodes = [node for node in node.direct_next[0].get_leaf() if node.node.type == "identifier"]

            # deal with the right node
            for right_node in right_nodes:
                if right_node.node.type == "call_expression":
                    simulate_data_flow(right_node, text, table)
                token = right_node.get_text(text)
                # find declaration
                declared_var = find_declare_scope(token)
                if declared_var["lr"] is not None:
                    right_node.last_read |= declared_var["lr"]
                if declared_var["lw"] is not None:
                    right_node.last_write |= declared_var["lw"]
                declared_var["lr"] = {right_node}

            # deal with the left node
            for left_node in left_nodes:
                token = left_node.get_text(text)
                if node.node.type == "init_declarator":
                    table[-1][(token, "l")] = {"lr": {left_node}, "lw": {left_node}}
                else:
                    declared_var = find_declare_scope(token)
                    if declared_var["lr"] is not None:
                        left_node.last_read |= declared_var["lr"]
                    if declared_var["lw"] is not None:
                        left_node.last_write |= declared_var["lw"]
                    declared_var["lw"] = {left_node}
                    declared_var["lr"] = {left_node}

        elif node.node.type == "call_expression":
            assert node.direct_next[1].node.type == "argument_list"
            argument_list = node.direct_next[1]
            arguments = [n for n in argument_list.get_leaf() if n.node.type == "identifier"]
            for argument in arguments:
                token = argument.get_text(text)
                declared_var = find_declare_scope(token)
                if declared_var["lr"] is not None:
                    argument.last_read |= declared_var["lr"]
                if declared_var["lw"] is not None:
                    argument.last_write |= declared_var["lw"]
                declared_var["lr"] = {argument}

        elif node.node.type == "update_expression":
            identifiers = [n for n in node.get_leaf() if n.node.type == "identifier"]
            for identifier in identifiers:
                token = identifier.get_text(text)
                declared_var = find_declare_scope(token)
                if declared_var["lr"] is not None:
                    identifier.last_read |= {declared_var["lr"], identifier}
                if declared_var["lw"] is not None:
                    identifier.last_write |= {declared_var["lw"], identifier}
                declared_var["lr"] = {identifier}
                declared_var["lw"] = {identifier}
        else:
            expression = node.get_text(text)
            valid = False
            for compare in COMPARE:
                if compare in expression:
                    valid = True
                    break
            if valid or node.node.type == 'return_statement':
                terminals = [t_node for t_node in node.get_leaf() if t_node.node.type == "identifier"]
                for terminal in terminals:
                    token = terminal.get_text(text)
                    declared_var = find_declare_scope(token)
                    if declared_var["lr"] is not None:
                        terminal.last_read |= declared_var["lr"]
                    if declared_var["lw"] is not None:
                        terminal.last_write |= declared_var["lw"]
                    declared_var["lr"] = {terminal}

    else:
        for child_node in node.direct_next:
            simulate_data_flow(child_node, text, table)
