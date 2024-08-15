import ast
import sys
import os
import json


def main():
    tree = ast.parse(open(sys.argv[1]).read())
    DumpAsJson([ConvertToAnf(sub_tree.value) for sub_tree in tree.body])


def ConvertToAnf(tree):
    if isinstance(tree, ast.Call):
        if tree.func.id == "Let":
            return ConvertLetToAnf(tree)
        converter = NestedCallArgsConverter(ConvertToAnf(tree.func))
        return converter(tree.args)
    if isinstance(tree, ast.Lambda):
        return ConvertLambdaToAnf(tree)
    if isinstance(tree, ast.Name):
        return tree.id
    if isinstance(tree, ast.List):
        return NestedCallArgsConverter("list")(tree.elts)
    if isinstance(tree, ast.Constant):
        if isinstance(tree.value, str):
            return dict(str=tree.value)
        if isinstance(tree.value, int):
            return tree.value
        if isinstance(tree.value, bool):
            return tree.value
    raise NotImplementedError(f"not implemented. ast: {tree}. lineno: {tree.lineno}")


class NestedCallArgsConverter:
    def __init__(self, f):
        self.func = f
        self.bindings = []
        self.seq_no = 0

    def __call__(self, args):
        body_args = self.RecursiveConvertCallArgs(args)
        if len(self.bindings) == 0:
            return [self.func, *body_args]
        return ["let", self.bindings, [self.func, *body_args]]

    def RecursiveConvertCallArgs(self, args):
        return [self.RecursiveConvertCallArg(arg) for arg in args]

    def RecursiveConvertCallArg(self, tree):
        if not isinstance(tree, ast.Call):
            return ConvertToAnf(tree)
        if tree.func.id == "Let":
            return ConvertLetToAnf(tree)
        var = self.get_tmp_var()
        self.bindings.append(
            [var, [ConvertToAnf(tree.func), *self.RecursiveConvertCallArgs(tree.args)]]
        )
        return var

    def get_tmp_var(self):
        return f"__lambda_expr_tmp{self.get_seq_no()}"

    def get_seq_no(self):
        ret = self.seq_no
        self.seq_no += 1
        return ret


def ConvertLetToAnf(let):
    raise NotImplementedError(f"not implemented. {let}")


def ConvertLambdaToAnf(lmbd):
    return ["lambda", [arg.arg for arg in lmbd.args.args], ConvertToAnf(lmbd.body)]


def DumpAsJson(data):
    with open(sys.argv[2], "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
