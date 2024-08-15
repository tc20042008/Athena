import typing as t
import numpy as np
from athena.rp_expr.rp_expr import Tokenize, PrimitiveId, LetsListTokenRpExpr
from athena.rp_expr.rp_expr_passes import (
    FlattenTokenListPass,
    FoldTokensPass,
    RecursiveFoldTokensPass,
    FoldIfTokenIdGreatEqualPass,
    UnflattenAndSubThresholdPass,
)


class RpExprParser:
    def __init__(self, window_size=8):
        self.window_size = window_size

    def __call__(self, primitive_id_lists: t.List[t.List[PrimitiveId]]):
        token_list, id_allocator, token_id2primitive_id = Tokenize(primitive_id_lists)
        flatten_pass = FlattenTokenListPass(id_allocator)
        success, flattened_rp_expr = flatten_pass(token_list)
        assert success
        fold_pass = RecursiveFoldTokensPass(id_allocator, self.window_size)
        success, fold_rp_expr = fold_pass(flattened_rp_expr.flattened_tensor)
        assert success
        threshold = len(primitive_id_lists)
        unflatten_pass = UnflattenAndSubThresholdPass(
            id_allocator=id_allocator,
            threshold_start_token_id=threshold,
        )
        success, threshold_fold_rp_expr = unflatten_pass(fold_rp_expr)
        assert success
        return threshold_fold_rp_expr, token_id2primitive_id
