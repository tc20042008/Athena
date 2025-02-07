import ast
import sys
import os
import json
from dataclasses import dataclass
import typing as t
import itertools
import functools
import operator

def main():
    tree = ast.parse(open(sys.argv[1]).read())
    parser = PyToAnfParser()
    parser(tree).ConvertToAnfExpr().DumpToFileAsJson(sys.argv[2])

@dataclass
class AnfExpr:

    def DumpToFileAsJson(self, file_name):
        with open(file_name, "w") as f:
            json.dump(self.value, f, indent=2)

@dataclass
class AtomicAnfExpr(AnfExpr):
    value : t.Any

@dataclass
class CombinedAnfExpr(AnfExpr):
    value : t.Any

@dataclass
class AnfParseResult:
    bindings: t.List["GeneratedCode"]
    body_atomic_anf_expr: AtomicAnfExpr

    def __add__(self, other):
        return AnfParseResult(
            bindings=[*self.bindings, *other.bindings],
            body_atomic_anf_expr=other.body_atomic_anf_expr
        )

    def ConvertToAnfExpr(self):
        ret = self.body_atomic_anf_expr
        if len(self.bindings) == 0:
            return ret
        assert isinstance(ret, AtomicAnfExpr)
        ret = CombinedAnfExpr(["__builtin_identity__", self.body_atomic_anf_expr.value])
        return CombinedAnfExpr(["__builtin_let__", self.bindings, ret.value])


class PyToAnfParser:
    def __init__(self, seq_no_counter = None, return_count_constraint = None):
        self.bindings = []
        self.seq_no_counter = (
            seq_no_counter if seq_no_counter is not None else itertools.count()
        )
        self.return_count_constraint = (
            return_count_constraint
            if return_count_constraint is not None
            else ReturnCounterConstraint(limits=1)
        )

    def __call__(self, tree):
        ret = self.Parse(tree)
        return AnfParseResult(
            bindings=self.bindings,
            body_atomic_anf_expr=ret
        )

    def Parse(self, tree):
        method_name = f"Parse{type(tree).__name__}"
        return getattr(self, method_name)(tree)

    def ParseImport(self, tree):
        for alias in tree.names:
            assert isinstance(alias, ast.alias)
            name = alias.name
            asname = alias.asname if alias.asname is not None else name
            self.Bind(asname, ["import", dict(str=name)])
        return AtomicAnfExpr(None)

    def ParseClassDef(self, tree : ast.ClassDef):
        assert len(tree.keywords) == 0
        class_name = tree.name
        def GetBases():
            bases = [self.Parse(base) for base in tree.bases]
            return self.BindToTmpVar(['__builtin_list__', *[x.value for x in bases]])
        def GetFunctions():
            body_name_and_method_pair = []
            for func_def in tree.body:
                if isinstance(func_def, ast.Pass):
                    continue
                assert isinstance(func_def, ast.FunctionDef), f"only method supported in class definition, {type(func_def)} were given."
                func_code = self.BindToTmpVar([
                    '__builtin_getattr__',
                    self.Parse(func_def).value,
                    dict(str='__function__')
                ])
                pair = self.BindToTmpVar([
                    "__builtin_list__",
                    dict(str=func_def.name),
                    func_code.value
                ])
                body_name_and_method_pair.append(pair)
            positional_args = self.BindToTmpVar(['__builtin_list__'])
            keyword_args = self.BindToTmpVar(['__builtin_list__', *[
                x.value for x in body_name_and_method_pair
            ]])
            packed_args = self.BindToTmpVar([
                '__builtin_PackedArgs__',
                positional_args.value,
                keyword_args.value
            ])
            return self.BindToTmpVar(['BuiltinSerializableAttrMap', packed_args.value])
        class_anf_expr = self.BindToTmpVar([
            'type',
            dict(str=class_name),
            GetBases().value,
            GetFunctions().value
        ])
        for elt in reversed(tree.decorator_list):
            decorator = self.Parse(elt)
            class_anf_expr = self.BindToTmpVar([decorator.value, class_anf_expr.value])
        self.Bind(class_name, class_anf_expr)
        return class_anf_expr

    def Parsekeyword(self, tree):
        value = self.Parse(tree.value)
        return self.BindToTmpVar(["__builtin_list__", dict(str=tree.arg), value.value])

    def ParseBinOp(self, tree):
        left = self.Parse(tree.left)
        op = self.Parse(tree.op)
        right = self.Parse(tree.right)
        return self.BindToTmpVar([op.value, left.value, right.value])

    def ParseUnaryOp(self, tree):
        op = self.Parse(tree.op)
        operand = self.Parse(tree.operand)
        return self.BindToTmpVar([op.value, operand.value])

    def ParseCompare(self, tree):
        assert len(tree.ops) == 1
        op = self.Parse(tree.ops[0])
        left = self.Parse(tree.left)
        assert len(tree.comparators) == 1
        right = self.Parse(tree.comparators[0])
        return self.BindToTmpVar([op.value, left.value, right.value])

    def ParseAdd(self, tree):
        return AtomicAnfExpr("__builtin_Add__")

    def ParseSub(self, tree):
        return AtomicAnfExpr("__builtin_Sub__")

    def ParseMult(self, tree):
        return AtomicAnfExpr("__builtin_Mul__")

    def ParseDiv(self, tree):
        return AtomicAnfExpr("__builtin_Div__")

    def ParseMod(self, tree):
        return AtomicAnfExpr("__builtin_Mod__")

    def ParseUSub(self, tree):
        return AtomicAnfExpr("__builtin_Neg__")

    def ParseEq(self, tree):
        return AtomicAnfExpr("__builtin_EQ__")

    def ParseNotEq(self, tree):
        return AtomicAnfExpr("__builtin_NE__")

    def ParseGt(self, tree):
        return AtomicAnfExpr("__builtin_GT__")

    def ParseGtE(self, tree):
        return AtomicAnfExpr("__builtin_GE__")

    def ParseLt(self, tree):
        return AtomicAnfExpr("__builtin_LT__")

    def ParseLtE(self, tree):
        return AtomicAnfExpr("__builtin_LE__")

    def ParseNot(self, tree):
        return AtomicAnfExpr("__builtin_Not__")

    def ParseModule(self, module: ast.Module):
        parse_result = AnfParseResult(
            bindings=[],
            body_atomic_anf_expr=AtomicAnfExpr(None)
        )
        if len(module.body) > 0:
            seq_no_counter = itertools.count()
            return_count_constraint = ReturnCounterConstraint(limits=0)
            parse_result = functools.reduce(operator.add, (
                PyToAnfParser(seq_no_counter, return_count_constraint)(tree)
                for tree in module.body
            ))
        return parse_result.ConvertToAnfExpr()

    def ParseFunctionDef(self, function_def: ast.FunctionDef):
        if len(function_def.body) > 0:
            return_count_constraint = ReturnCounterConstraint(limits=1)
            return_stmt_idx = self.GetStmtSizeUntilReturn(function_def.body)
            parse_result = functools.reduce(operator.add, [
                PyToAnfParser(self.seq_no_counter, return_count_constraint)(tree)
                for tree in function_def.body[0:return_stmt_idx]
                if not isinstance(tree, ast.Pass)
            ] + [
                AnfParseResult(
                    bindings=[],
                    body_atomic_anf_expr=AtomicAnfExpr(None)
                )
            ])
        else:
            parse_result = AnfParseResult(
                bindings=[],
                body_atomic_anf_expr=AtomicAnfExpr(None)
            )
        args = [
            arg.arg
            for arg in function_def.args.args
        ]
        lmbd = AtomicAnfExpr(['lambda', args, parse_result.ConvertToAnfExpr().value])
        for elt in reversed(function_def.decorator_list):
            decorator = self.Parse(elt)
            lmbd = self.BindToTmpVar([decorator.value, lmbd.value])
        func_name = function_def.name
        self.Bind(func_name, lmbd)
        return AtomicAnfExpr(func_name)

    def ParseLambda(self, function_def: ast.Lambda):
        return_count_constraint = ReturnCounterConstraint(limits=0)
        parser = PyToAnfParser(self.seq_no_counter, return_count_constraint)
        parse_result = parser(function_def.body)
        args = [
            arg.arg
            for arg in function_def.args.args
        ]
        return AtomicAnfExpr(['lambda', args, parse_result.ConvertToAnfExpr().value])

    def ParseAssign(self, tree):
        assert len(tree.targets) == 1
        if isinstance(tree.targets[0], ast.Name):
            val = self.Parse(tree.value)
            var = tree.targets[0].id
            self.Bind(var, val)
            return AtomicAnfExpr(var)
        elif isinstance(tree.targets[0], ast.Attribute):
            val = self.Parse(tree.value)
            attr = tree.targets[0]
            f = self.BindToTmpVar([
                '__builtin_setattr__',
                self.Parse(attr.value).value,
                dict(str=attr.attr)
            ])
            return self.BindToTmpVar([
                f.value,
                dict(str=attr.attr),
                val.value
            ])
        elif isinstance(tree.targets[0], ast.Subscript):
            val = self.Parse(tree.value)
            subscript = tree.targets[0]
            slice_val = self.Parse(subscript.slice).value
            f = self.BindToTmpVar([
                '__builtin_setitem__',
                self.Parse(subscript.value).value,
                slice_val
            ])
            return self.BindToTmpVar([
                f.value,
                slice_val,
                val.value
            ])            
        else:
            raise NotImplementedError(tree.targets)

    def ParseSubscript(self, tree):
        val = self.Parse(tree.value)
        slc = self.Parse(tree.slice)
        return self.BindToTmpVar(["__builtin_getitem__", val.value, slc.value])

    def ParseExpr(self, tree):
        return self.BindToTmpVar(self.Parse(tree.value))

    def BindToTmpVar(self, value):
        tmp_var = self.get_tmp_var()
        self.Bind(tmp_var, value)
        return AtomicAnfExpr(tmp_var)

    def GetStmtSizeUntilReturn(self, stmts):
        for idx, stmt in enumerate(stmts):
            if isinstance(stmt, ast.Return):
                return idx + 1
        return len(stmts)

    def ParseReturn(self, tree: ast.Return):
        self.return_count_constraint.CountAndCheck()
        value = self.Parse(tree.value)
        return self.BindToTmpVar(["__builtin_return__", value.value])

    def ParseStarred(self, tree: ast.Starred):
        value = self.Parse(tree.value)
        return self.BindToTmpVar(["__builtin_starred__", value.value])

    def ParseCall(self, tree: ast.Call):
        func = self.Parse(tree.func)
        assert isinstance(func, AtomicAnfExpr)
        def ParseArg(arg):
            parsed_arg = self.Parse(arg)
            assert isinstance(parsed_arg, AtomicAnfExpr)
            return parsed_arg
        args = [ParseArg(arg).value for arg in tree.args]
        kwargs = None
        if len(tree.keywords) > 0:
            keywords = [ParseArg(arg).value for arg in tree.keywords]
            kwargs = self.BindToTmpVar(["__builtin_list__", *keywords])
        if kwargs is None:
            if any(isinstance(arg, ast.Starred) for arg in tree.args):
                l = self.BindToTmpVar(["__builtin_list__", *args])
                return self.BindToTmpVar(["__builtin_apply__", func.value, l.value])
            else:
                return self.BindToTmpVar([func.value, *args])
        else:
            args = self.BindToTmpVar(["__builtin_list__", *args])
            packed_args = self.BindToTmpVar([
                "__builtin_PackedArgs__",
                args.value,
                kwargs.value
            ])
            return self.BindToTmpVar([func.value, packed_args.value])


    def ParseList(self, lst: ast.List):
        return self._ParseCall('__builtin_list__', lst.elts)

    def _ParseCall(self, func, ast_args):
        def ParseArg(arg):
            parsed_arg = self.Parse(arg)
            assert isinstance(parsed_arg, AtomicAnfExpr)
            return parsed_arg
        args = [ParseArg(arg).value for arg in ast_args]
        ret_var = self.get_tmp_var()
        self.Bind(ret_var, [func, *args])
        return AtomicAnfExpr(ret_var)

    def ParseAttribute(self, attr: ast.Attribute):
        ret_var = self.get_tmp_var()
        self.Bind(ret_var, [
            '__builtin_getattr__',
            self.Parse(attr.value).value,
            dict(str=attr.attr)
        ])
        return AtomicAnfExpr(ret_var)

    def ParseName(self, name: ast.Name):
        return AtomicAnfExpr(name.id)

    def ParseConstant(self, constant: ast.Constant):
        if isinstance(constant.value, str):
            return AtomicAnfExpr(dict(str=constant.value))
        if isinstance(constant.value, (bool, int, float)):
            return AtomicAnfExpr(constant.value)
        if constant.value is None:
            return AtomicAnfExpr(None)
        raise NotImplementedError(f"{constant} not supported by anf_expr")

    def ParseJoinedStr(self, tree: ast.JoinedStr):
        if len(tree.values) == 0:
            return AtomicAnfExpr(dict(str=""))
        def ToString(elt):
            parsed_elt = self.Parse(elt)
            parsed_elt = self.BindToTmpVar(['__builtin_ToString__', parsed_elt.value])
            return parsed_elt
        ret = ToString(tree.values[0])
        for elt in tree.values[1:]:
            parsed_elt = ToString(elt)
            ret = self.BindToTmpVar(['__builtin_Add__', ret.value, parsed_elt.value])
        return ret

    def ParseFormattedValue(self, tree: ast.FormattedValue):
        return self.Parse(tree.value)

    def Bind(self, var_name, anf_expr):
        return getattr(self, f"Bind{type(anf_expr).__name__}")(var_name, anf_expr)

    def BindAtomicAnfExpr(self, var_name, anf_expr):
        self.bindings.append(
            [var_name, ["__builtin_identity__", anf_expr.value]]
        )

    def Bindlist(self, var_name, anf_expr):
        self.bindings.append([var_name, anf_expr])

    def get_tmp_var(self):
        return f"___{next(self.seq_no_counter)}"


class ReturnCounterConstraint:
    def __init__(self, limits):
        self.counter = itertools.count()
        self.limits = limits

    def CountAndCheck(self):
        return_stmt_id = next(self.counter)
        assert return_stmt_id < self.limits

if __name__ == "__main__":
    main()
