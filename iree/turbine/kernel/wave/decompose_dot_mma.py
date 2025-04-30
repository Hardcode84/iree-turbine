# Copyright 2025 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .._support.tracing import CapturedTrace
from .constraints import (
    Constraint,
    GenericDot,
)
import torch.fx as fx
from ..ops.wave_ops import (
    get_custom,
    MMA,
    Add,
    Mul,
    Sum,
    CastOp,
    GetResult,
    CustomOp,
    Reshape,
    ShuffleOp,
)
from .utils.general_utils import (
    get_hardware_constraint,
    get_largest_index_and_size,
)
from .utils.classes import ShuffleMode
from ..lang.global_symbols import THREAD_0
from copy import copy


def fixup_index_sizes(trace: CapturedTrace, constraints: list[Constraint]):
    hardware_constraint = get_hardware_constraint(constraints)

    def get_index(custom: CustomOp):
        if not custom.indexing_dims:
            return None
        if isinstance(custom, MMA):
            return custom.acc.index
        return custom.index

    # Resolve thread shapes conflicts for the args
    for op in trace.walk(lambda node: True):
        custom = get_custom(op)
        if isinstance(custom, GetResult):
            continue

        for i, arg in enumerate(op.args):
            if not isinstance(arg, fx.Node):
                continue

            arg_custom = get_custom(arg)
            if isinstance(custom, MMA) and isinstance(arg_custom, MMA):
                continue

            arg_index = get_index(arg_custom)
            expected_index = custom.operand_index(i)
            if arg_index is None or expected_index is None:
                continue

            # print("----")
            # print(custom)
            # print(arg_index)
            # print(expected_index)

            _, size1 = get_largest_index_and_size(arg_index)
            _, size2 = get_largest_index_and_size(expected_index)
            if size1 == size2:
                continue

            if size1 != 1:
                breakpoint()
                raise NotImplementedError(
                    "Currently only support resolving discrepancies when the first shape is 1"
                    f", got {size1} and {size2} for {custom} and {arg}"
                )

            subgroup_size = hardware_constraint.threads_per_wave
            graph = custom.graph
            index = custom.index
            vector_shapes = custom.vector_shapes
            with graph.inserting_before(custom.fx_node):
                parts = []
                for j in range(size2):
                    idx = (((THREAD_0 % subgroup_size) // size2) * size2) + j
                    val = ShuffleOp(
                        arg, idx, subgroup_size, ShuffleMode.IDX
                    ).add_to_graph(graph)
                    val.index = index
                    val.vector_shapes = vector_shapes
                    parts.append(val)

                reshape = Reshape(parts, vector_shapes).add_to_graph(graph)
                reshape.index = index
                custom.update_arg(i, reshape)


def decompose_dot_mma(trace: CapturedTrace, constraints: list[Constraint]):
    return
    hardware_constraint = get_hardware_constraint(constraints)

    def get_mma_type(mma_op: MMA) -> GenericDot:
        mma_type = mma_op.mma_type
        if mma_type is None:
            mma_type = hardware_constraint.mma_type

        return mma_type

    def is_dot_mma(node: fx.Node) -> bool:
        custom = get_custom(node)
        if not isinstance(custom, MMA):
            return False

        mma_type = get_mma_type(custom)
        return isinstance(mma_type, GenericDot)

    mma_nodes = trace.walk(is_dot_mma)
    for node in mma_nodes:
        mma_op = get_custom(node)
        mma_type = get_mma_type(mma_op)
        if mma_type.out_vec_size != 1:
            raise ValueError("Only support dot product with output vector size 1")

        with mma_op.graph.inserting_before(mma_op.fx_node):
            lhs = mma_op.lhs
            rhs = mma_op.rhs
            acc = mma_op.acc

            dtype = acc.type.dtype
            lhs_index = copy(lhs.index)
            rhs_index = copy(rhs.index)
            lhs = CastOp(lhs, dtype).add_to_graph(mma_op.graph)
            rhs = CastOp(rhs, dtype).add_to_graph(mma_op.graph)
            lhs.index = copy(lhs_index)
            rhs.index = copy(rhs_index)

            k_sym = get_custom(lhs).indexing_dims[1]

            mul = Mul(lhs, rhs).add_to_graph(mma_op.graph)
            sum = Sum(mul, None, k_sym).add_to_graph(mma_op.graph)
            red = Add(sum, acc).add_to_graph(mma_op.graph)

            mul.index = lhs_index | rhs_index
            del lhs_index[k_sym]
            del rhs_index[k_sym]
            ret_index = lhs_index | rhs_index
            sum.index = ret_index
            red.index = ret_index

            vector_shapes = mma_op.vector_shapes
            mul.vector_shapes = vector_shapes
            sum.vector_shapes = vector_shapes
            red.vector_shapes = vector_shapes

            mma_op.replace_all_uses_with(red)
