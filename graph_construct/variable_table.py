SPECIAL_TOKENS = ("cin", "cout", "endl")
UPDATE_MODE = {"ro", "wo", "rw"}  # read-only, write-only, read-and-write


class VariableTable:
    """
    There's one thing to be mention that all of the variables references follow the lazy strategy, which means only
    inherent from the outer scope when truly mentioned
    """
    def __init__(self, father=None):
        """
        One basic unit is a key-value pair: (token, {"lr": set of "lr ref", "lw": set of "lw ref"}
        token is a string, reference are cpp parser's node

        One variable table is for one sequential process, for example, a compound statement, or a single statement

        :param father: the direct father scope
        """
        self.father = father
        self.child = None
        self.local_variables = dict({})
        self.outer_variables = dict({})

    def find_reference(self, token):
        if token in self.local_variables:
            return self.local_variables[token]
        elif token in self.outer_variables:
            return self.outer_variables[token]
        elif self.father is None:
            return None
        else:
            unit = self.father.find_reference(token)
            if unit is not None:  # be sure to deep copy the elements
                self.outer_variables[token] = {"lr": set([]) | unit["lr"], "lw": set([]) | unit["lw"]}
                unit = self.outer_variables[token]
            return unit

    def add_reference(self, token, node):
        assert token == node.token
        if token in SPECIAL_TOKENS:
            return 1
        self.local_variables[token] = {"lr": {node}, "lw": {node}}
        return self.local_variables[token]

    def find_and_update(self, token: str, node, mode: str):
        assert token == node.token
        if node.token in SPECIAL_TOKENS:
            return 1

        unit = self.find_reference(token)
        if unit is None:
            raise ValueError(f"{token} not found")

        if mode not in UPDATE_MODE:
            raise ValueError(f"{mode} is not an available update mode")

        node.last_read |= unit["lr"]
        node.last_write |= unit["lw"]

        if mode == "ro":
            unit["lr"] = {node}
        elif mode == "wo":
            unit["lw"] = {node}
        else:
            unit["lr"] = {node}
            unit["lw"] = {node}

        return 0

    """
    def find_or_add_reference(self, token):
        unit = self.find_reference(token)
        if unit is None:
            unit = self.add_reference(token)
        return unit
    """

    def add_variable_table(self):
        self.child = VariableTable(self)
        return self.child

    def pop_self(self):
        self.father.child = None
        for token in self.outer_variables:
            if token in self.father.local_variables:
                self.father.local_variables[token] = self.outer_variables[token]
            elif token in self.outer_variables:
                self.father.outer_variables[token] = self.outer_variables[token]
            else:
                raise ValueError(f"Outer variable {token} is not found in outer tables")

    def merge_and_pop_self(self):
        self.father.child = None
        for token in self.outer_variables:
            if token in self.father.local_variables:
                self.father.local_variables[token]["lr"] |= self.outer_variables[token]["lr"]
                self.father.local_variables[token]["lw"] |= self.outer_variables[token]["lw"]
            elif token in self.father.outer_variables:
                self.father.outer_variables[token]["lr"] |= self.outer_variables[token]["lr"]
                self.father.outer_variables[token]["lw"] |= self.outer_variables[token]["lw"]
            else:
                raise ValueError(f"Outer variable {token} is not found in outer tables")

    def __str__(self):
        return f" local: {str(self.local_variables)}\n global: {str(self.outer_variables)}"
