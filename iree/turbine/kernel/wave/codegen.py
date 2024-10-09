# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import operator
import sympy
import math
from typing import Any, Callable, ClassVar, Optional, List, Type, Dict
from dataclasses import dataclass
import torch.fx as fx
import torch.utils._pytree as pytree
from collections import namedtuple

from ..compiler.ir import (
    Attribute,
    DenseElementsAttr,
    FloatAttr,
    F32Type,
    IndexType,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    IrType,
    Location,
    MemRefType,
    OpResult,
    ShapedType,
    Value,
    VectorType,
    amdgpu_d,
    arith_d,
    func_d,
    gpu_d,
    math_d,
    memref_d,
    stream_d,
    scf_d,
    vector_d,
)
from iree.turbine.aot.support.ir_utils import _is_float_type, _is_integer_like_type

# TK infrastructure imports.
from iree.turbine.kernel.lang.global_symbols import *
from ..ops.wave_ops import (
    write,
    broadcast,
    register,
    mma,
    shuffle,
    read,
    reduction,
    exp2,
    maximum,
    get_custom,
    get_result,
    allocate,
    shared_memory_barrier,
    extract_slice,
    CustomOp,
)
from ..lang.wave_types import IndexMapping, IndexSymbol
from ..compiler.base import CodegenError, ValidationError, NDEBUG
from ..compiler.kernel_codegen import BoundKernelSignature
from .._support.tracing import CapturedTrace
from ..compiler.builder import IRProxyValue
from ..compiler.utils import strides_from_symbolic_shape
from ..compiler.vector_codegen import (
    cast_kernel_buffer,
    cast_py_literal,
    cast_py_value,
    cast_vector,
)
from .constraints import (
    Constraint,
    HardwareConstraint,
    MMAType,
    WorkgroupConstraint,
    TilingConstraint,
)
from .utils import subs_idxc, find_index_bounds, get_hardware_vector_map

# Indexing imports.
from .._support.indexing import IndexingContext, IndexExpr, IndexSequence


@dataclass
class WaveEmitter:
    """Emits a warp function as a `func` with a signature derived from the gm."""

    root_sig: BoundKernelSignature
    trace: CapturedTrace
    constraints: list[Constraint]
    dynamic_symbols: list[IndexSymbol]
    ip: InsertionPoint = None
    OP_HANDLERS: ClassVar[dict[str, Callable[["WaveEmitter", fx.Node], None]]] = {}
    _node_values: ClassVar[dict[fx.Node, List[IRProxyValue]]] = {}

    def __post_init__(self):
        self.ip = InsertionPoint(self.root_sig.entry_block)

    def emit_program_invariants(self):
        self.workgroup_ids = [
            stream_d.dispatch_workgroup_id(IntegerAttr.get(IndexType.get(), 0)),
            stream_d.dispatch_workgroup_id(IntegerAttr.get(IndexType.get(), 1)),
            stream_d.dispatch_workgroup_id(IntegerAttr.get(IndexType.get(), 2)),
        ]
        self.thread_ids = [
            gpu_d.thread_id(gpu_d.Dimension.x),
            gpu_d.thread_id(gpu_d.Dimension.y),
            gpu_d.thread_id(gpu_d.Dimension.z),
        ]
        self.induction_vars: dict[IndexSymbol, Value] = {}
        self.dynamic_dims: dict[IndexSymbol, Value] = {}
        symbol_iterator = iter(self.dynamic_symbols)
        for arg in self.root_sig.entry_block.arguments:
            if arg.type == IndexType.get():
                self.dynamic_dims[next(symbol_iterator)] = arg

    def emit(self, graph: Optional[fx.Graph] = None):
        with self.ip, Location.unknown():
            self.emit_program_invariants()
            self._emit_graph(
                graph if graph is not None else self.trace.get_root_graph()
            )

    def finish(self):
        with self.ip, Location.unknown():
            func_d.ReturnOp([])

    def _emit_graph(self, graph: fx.Graph):
        """Emits the given graph at the current insertion point."""
        for node in graph.nodes:
            if node.op == "call_function" or node.op == "call_method":
                self._emit_function_call_node(node)
            if node.op == "output":
                return node.args

    def _emit_function_call_node(self, node: fx.Node):
        target_op = node.tkw_op_name
        try:
            handler = self.OP_HANDLERS[target_op]
        except KeyError:
            raise CodegenError(f"No handler registered for op {target_op}")

        handler(self, node)

    def lookup_node_values(self, node: fx.Node) -> List[Value]:
        assert NDEBUG or isinstance(node, fx.Node)
        values = self._node_values.get(node)
        if values is None:
            values = [self.root_sig.resolve_by_reference(("node", node))]
            self._node_values[node] = values
        values = [v.ir_value if isinstance(v, IRProxyValue) else v for v in values]
        return values

    def bind_node_proxy(self, node: fx.Node, proxy: IRProxyValue):
        """Binds a node's result to a Python/IR proxy object."""
        assert NDEBUG or (isinstance(node, fx.Node) and isinstance(proxy, IRProxyValue))
        self._node_values[node] = [proxy]

    def bind_node_proxies(self, node: fx.Node, proxies: List[IRProxyValue]):
        assert NDEBUG or (
            isinstance(node, fx.Node)
            and all(isinstance(p, IRProxyValue) for p in proxies)
        )
        self._node_values[node] = proxies


def get_type_or_element_type(operand_type: IrType):
    assert isinstance(operand_type, IrType)
    if isinstance(operand_type, ShapedType):
        return operand_type.element_type
    else:
        return operand_type


def get_induction_vars(emitter: WaveEmitter) -> tuple[list[IndexExpr], list[OpResult]]:
    induction_var_syms = []
    induction_vars = []
    if emitter.induction_vars:
        for constraint in emitter.constraints:
            if isinstance(constraint, TilingConstraint):
                assert (
                    constraint.dim in emitter.induction_vars
                ), f"Could not find induction var for {constraint.dim} dimension"
                induction_var_syms.append(constraint.induction_var)
                induction_vars.append(emitter.induction_vars[constraint.dim])

    return induction_var_syms, induction_vars


def get_emitter_subs(emitter: WaveEmitter) -> dict[IndexSymbol, Any]:
    induction_var_syms, induction_vars = get_induction_vars(emitter)

    # TODO: factor this out
    all_symbols = emitter.thread_ids + emitter.workgroup_ids + induction_vars
    dynamics = dict(
        zip(
            [THREAD_0, THREAD_1, THREAD_2, WORKGROUP_0, WORKGROUP_1, WORKGROUP_2]
            + induction_var_syms,
            all_symbols,
        )
    )
    dynamics.update(emitter.dynamic_dims)
    return dynamics


_Rational = namedtuple("_Rational", ["numerator", "denominator"])


def gen_sympy_index(
    dynamics: dict[IndexSymbol, Any],
    expr: sympy.Expr,
    induction_var_syms: list[IndexExpr] = [],
) -> OpResult:
    stack: list[OpResult] = []

    def _get_ir_value(arg):
        if not isinstance(arg, (Value, OpResult)):
            arg = arg.result

        return arg

    def _check_vec_scalar(a, b):
        return isinstance(a.type, VectorType) and a.type.element_type == b.type

    def _broadcast(a, b):
        a = _get_ir_value(a)
        b = _get_ir_value(b)

        if a.type == b.type:
            return a, b

        if _check_vec_scalar(a, b):
            b = vector_d.splat(a.type, b)
            return a, b

        if _check_vec_scalar(b, a):
            a = vector_d.splat(b.type, a)
            return a, b

        raise CodegenError(f"Cannot broadcast {a.type} and {b.type}")

    def get_const_val(arg):
        if isinstance(arg, OpResult):
            arg = arg.owner.opview

        if isinstance(arg, arith_d.ConstantOp):
            value = arg.attributes["value"]
            if isinstance(value, IntegerAttr):
                return int(value)

        return None

    def muli_fold(lhs, rhs):
        if get_const_val(lhs) == 1:
            return rhs

        if get_const_val(rhs) == 1:
            return lhs

        return arith_d.muli(lhs, rhs)

    # `x + (a/b)` transformed into `(x*b + a) / b`
    def _add(lhs, rhs):
        is_rational_lhs = isinstance(lhs, _Rational)
        is_rational_rhs = isinstance(rhs, _Rational)
        if is_rational_lhs and not is_rational_rhs:
            numerator = muli_fold(*_broadcast(lhs.denominator, rhs))
            numerator = arith_d.addi(*_broadcast(numerator, lhs.numerator))
            return _Rational(numerator, lhs.denominator)
        elif not is_rational_lhs and is_rational_rhs:
            numerator = muli_fold(*_broadcast(lhs, rhs.denominator))
            numerator = arith_d.addi(*_broadcast(numerator, rhs.numerator))
            return _Rational(numerator, rhs.denominator)
        elif is_rational_lhs and is_rational_rhs:
            lhs_numerator = muli_fold(*_broadcast(lhs.numerator, rhs.denominator))
            rhs_numerator = muli_fold(*_broadcast(rhs.numerator, lhs.denominator))
            numerator = arith_d.addi(*_broadcast(lhs_numerator, rhs_numerator))
            denominator = muli_fold(*_broadcast(lhs.denominator, rhs.denominator))
            return _Rational(numerator, denominator)
        else:
            return arith_d.addi(*_broadcast(lhs, rhs))

    # `x * (a/b)` transformed into `(x * a) / b`
    def _mul(lhs, rhs):
        is_rational_lhs = isinstance(lhs, _Rational)
        is_rational_rhs = isinstance(rhs, _Rational)
        if is_rational_lhs and not is_rational_rhs:
            numerator = muli_fold(*_broadcast(lhs.numerator, rhs))
            return _Rational(numerator, lhs.denominator)
        elif not is_rational_lhs and is_rational_rhs:
            numerator = muli_fold(*_broadcast(lhs, rhs.numerator))
            return _Rational(numerator, rhs.denominator)
        elif is_rational_lhs and is_rational_rhs:
            numerator = muli_fold(*_broadcast(lhs.numerator, rhs.numerator))
            denominator = muli_fold(*_broadcast(lhs.denominator, rhs.denominator))
            return _Rational(numerator, denominator)
        else:
            return muli_fold(*_broadcast(lhs, rhs))

    def _floor(value):
        if isinstance(value, _Rational):
            value = arith_d.divsi(*_broadcast(value.numerator, value.denominator))

        return value

    def _ceiling(value):
        if isinstance(value, _Rational):
            value = arith_d.ceildivsi(*_broadcast(value.numerator, value.denominator))

        return value

    def _group_rationals(stack, args):
        """Group rationals and non-rationals args into 2 contiguous sets.

        This allows to mul/add all non-rationals first, reducing total number of ops.
        """
        terms = []
        for arg in args:
            terms.append((arg, stack.pop()))

        if len(induction_var_syms) > 0:
            # Group args based on induction vars usage to help LICM later.
            def _key(arg):
                return sum(
                    1 for sym in induction_var_syms if sym in arg[0].free_symbols
                )

            terms.sort(key=_key, reverse=True)

        rationals = []
        non_rationals = []
        for _, val in terms:
            if isinstance(val, _Rational):
                rationals.append(val)
            else:
                non_rationals.append(val)

        return non_rationals + rationals

    def _apply(args, func):
        assert len(args) > 0
        value = args[0]
        for val in args[1:]:
            value = func(value, val)

        return value

    def _enforce_non_rational(val, term):
        if isinstance(val, _Rational):
            raise CodegenError(f"Rational is not supported yet in '{type(term)}'")

    def _get_const(val):
        if isinstance(val, int):
            return arith_d.constant(IndexType.get(), val)

        if isinstance(val, (tuple, list)):
            vec_type = VectorType.get([len(val)], IndexType.get())
            vals = [IntegerAttr.get(IndexType.get(), v) for v in val]
            return arith_d.constant(vec_type, DenseElementsAttr.get(vals, vec_type))

        raise CodegenError(f"Unsupported const val {val} : {type(val)}")

    idxc = IndexingContext.current()
    # Substitute in frozen vars to simplify expression.
    if not isinstance(expr, sympy.Expr):
        expr = sympy.sympify(expr)
    expr = expr.subs(idxc.subs)
    # Why affine, for now simply create indexing expressions.
    # This can easily be adapted to affine expressions later.
    for term in sympy.postorder_traversal(expr):
        match term:
            case sympy.Symbol():
                res = idxc.get_val(term)
                if res is not None:
                    stack.append(_get_const(res))
                elif term in dynamics.keys():
                    stack.append(dynamics[term])
                else:
                    raise CodegenError(f"Unknown symbol {term}")
            case sympy.Integer():
                stack.append(_get_const(int(term)))
            case sympy.Mul():
                args = _group_rationals(stack, term.args)
                stack.append(_apply(args, _mul))
            case sympy.Add():
                args = _group_rationals(stack, term.args)
                stack.append(_apply(args, _add))
            case sympy.Mod():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                mod = arith_d.remsi(*_broadcast(lhs, rhs))
                stack.append(mod)
            case sympy.floor():
                stack.append(_floor(stack.pop()))
            case sympy.ceiling():
                stack.append(_ceiling(stack.pop()))
            case sympy.Rational():
                numerator = _get_const(term.p)
                denominator = _get_const(term.q)
                stack.append(_Rational(numerator, denominator))
            case sympy.StrictLessThan():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                res = arith_d.cmpi(arith_d.CmpIPredicate.slt, *_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.And():
                rhs = stack.pop()
                lhs = stack.pop()
                _enforce_non_rational(rhs, term)
                _enforce_non_rational(lhs, term)
                res = arith_d.andi(*_broadcast(lhs, rhs))
                stack.append(res)
            case sympy.logic.boolalg.BooleanFalse():
                res = arith_d.constant(IntegerType.get_signless(1), 0)
                stack.append(res)
            case sympy.logic.boolalg.BooleanTrue():
                res = arith_d.constant(IntegerType.get_signless(1), 1)
                stack.append(res)
            case sympy.UnevaluatedExpr():
                continue
            case _:
                raise CodegenError(f"Can not handle {type(term)} : {term}")

    if len(stack) != 1 or isinstance(stack[0], _Rational):
        raise CodegenError(f"Expected single result, got {len(stack)}")

    return stack[0]


def _gen_sympy_index_full(emitter: WaveEmitter, expr: sympy.Expr) -> OpResult:
    subs = get_emitter_subs(emitter)
    induction_var_syms, _ = get_induction_vars(emitter)
    return gen_sympy_index(subs, expr, induction_var_syms)


def get_constant_attr(value: Any, element_type: IrType) -> Attribute:
    if _is_integer_like_type(element_type):
        return IntegerAttr.get(element_type, int(value))
    if _is_float_type(element_type):
        return FloatAttr.get(element_type, float(value))
    raise CodegenError(f"Cannot create a constant attribute for type `{element_type}`")


def handle_op(op: Callable[..., Any]):
    def decorator(
        f: Callable[[WaveEmitter, fx.Node], None]
    ) -> Callable[[WaveEmitter, fx.Node], None]:
        WaveEmitter.OP_HANDLERS[op.__name__] = f
        return f

    return decorator


###############################################################################
# Memory Ops
###############################################################################


@handle_op(register)
def handle_register(emitter: WaveEmitter, node: fx.Node):
    try:
        shape, dtype, value = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    get_thread_shape = lambda index: max(x.size for x in index.values())
    shape = [get_thread_shape(get_custom(node).index)]
    vector_shape = cast_py_literal(emitter, shape)
    element_type = IrType.parse(dtype.ir_type_asm())
    vector_type = VectorType.get(vector_shape, element_type)
    register = arith_d.ConstantOp(
        vector_type,
        DenseElementsAttr.get_splat(
            vector_type, get_constant_attr(value, element_type)
        ),
    ).result
    emitter.bind_node_proxy(node, IRProxyValue(register))


@handle_op(allocate)
def handle_allocate(emitter: WaveEmitter, node: fx.Node):
    try:
        shape, distributed_shape, dtype, address_space = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    memref_shape = cast_py_literal(emitter, distributed_shape)
    element_type = IrType.parse(dtype.ir_type_asm())
    address_space = Attribute.parse("#gpu.address_space<workgroup>")
    memref_type = MemRefType.get(memref_shape, element_type, None, address_space)
    alloc = memref_d.alloc(memref_type, [], [])
    emitter.bind_node_proxy(node, IRProxyValue(alloc))


def _get_start_index(i: IndexSequence | IndexExpr) -> IndexExpr:
    if isinstance(i, IndexSequence):
        i = i.start

    return i


def _get_start_indices(
    src_indices: dict[IndexExpr, IndexSequence | IndexExpr]
) -> list[IndexExpr]:
    start_indices = []
    for dim_indexing in src_indices:
        i = _get_start_index(src_indices[dim_indexing])
        start_indices.append(i)

    return start_indices


def _build_start_indices(
    emitter: WaveEmitter, src_indices: dict[IndexExpr, IndexSequence | IndexExpr]
) -> list[OpResult]:
    return [_gen_sympy_index_full(emitter, i) for i in _get_start_indices(src_indices)]


def _compute_offset(indices: list[IndexExpr], strides: list[IndexExpr]) -> IndexExpr:
    return sum(i * s for i, s in zip(indices, strides))


def _get_symbolic_shape(node: fx.Node) -> tuple[IndexExpr]:
    return get_custom(node).type.symbolic_shape


def _is_identity_mapping(
    mapping: IndexMapping,
    input_shape: Optional[tuple[IndexExpr]] = None,
    output_shape: Optional[tuple[IndexExpr]] = None,
) -> bool:
    if not mapping.is_identity():
        return False

    if input_shape is not None and mapping.input_shape != input_shape:
        return False

    if output_shape is not None and mapping.output_shape != output_shape:
        return False

    return True


def _build_mask(
    emitter: WaveEmitter, index: Dict[IndexExpr, IndexExpr], elements_per_thread: int
) -> Optional[OpResult]:
    bounds = find_index_bounds(emitter.constraints, index)
    if bounds is None:
        return None

    idxc = IndexingContext.current()
    last_dim = tuple(index.keys())[-1]
    new_index = {k: _get_start_index(v) for k, v in index.items()}

    new_index[last_dim] = new_index[last_dim] + idxc.iota(elements_per_thread)

    mask_expr = functools.reduce(
        lambda a, b: sympy.And(a, b), (new_index[dim] < dim for dim in bounds)
    )
    mask = _gen_sympy_index_full(emitter, mask_expr)

    mask_vec_type = VectorType.get([elements_per_thread], IntegerType.get_signless(1))
    if mask.type != mask_vec_type:
        mask = vector_d.splat(mask_vec_type, mask)

    return mask


def _construct_gather_scatter_indices(
    emitter: WaveEmitter,
    symbolc_shape: tuple[IndexExpr],
    index: tuple[IndexExpr],
    mapping: IndexMapping,
    elements_per_thread: int,
    is_read: bool,
    thread_offset: int = 0,
) -> tuple[OpResult, OpResult, OpResult]:
    # Apply symbolc_shape order to indices, e.g. if original mapping is
    # {M: iter(0), N: iter(1)} and symbolc_shape is (N, M), result will
    # be (iter(1), iter(0))
    if is_read:
        assert (
            mapping.is_output_identity()
        ), "non-identity output mapping is not supported yet"
        index_mapping = mapping.map_input_indices(symbolc_shape)
    else:
        assert (
            mapping.is_input_identity()
        ), "non-identity input mapping is not supported yet"
        index_mapping = mapping.map_output_indices(symbolc_shape)

    idxc = IndexingContext.current()
    index_mapping = tuple(i.subs(idxc.subs) for i in index_mapping)

    iters = mapping.iters

    # As we only support identity input/output mapping for now, we can directly
    # substitute iterators with corresponding expanded index.
    subs = [
        (sym, expr.start) for sym, expr in zip(iters.keys(), index.values())
    ] + list(idxc.subs.items())

    # Contruct input/output index, substituting iterators in input mapping with
    # expanded index.
    result_index = {key: m.subs(subs) for key, m in zip(symbolc_shape, index_mapping)}

    strides = strides_from_symbolic_shape(idxc, symbolc_shape)
    offsets = []

    start_indices = _get_start_indices(result_index)
    start_indices_orig = _get_start_indices(index)

    need_dynamic_offsets = False
    start_indices_offset = _compute_offset(start_indices, strides)
    for i in range(elements_per_thread):
        # Update most-minor dim, i.e. in case of identity mapping it will
        # be equivalent to just vector.load
        subs = [(sym, idx) for sym, idx in zip(iters.keys(), start_indices_orig)]
        subs[-1] = (subs[-1][0], start_indices_orig[-1] + i + thread_offset)
        indices = [i.subs(subs) for i in index_mapping]

        # First, we build indices as if resulting gather/scatter `start_indices`
        # are 0 as mapping expression may depend on absolute value of index
        # (e.g. `index % 32`). Then we adjust for the non-0 `start_indices` by
        # subtracting computed previously linear `start_indices_offset`. For
        # simple cases like transpose, the resulting expression should fold into
        # simple constant while more complex expressions may requires actual
        # arith ops on dynamic values.
        offset = _compute_offset(indices, strides) - start_indices_offset
        offset = subs_idxc(offset)

        if offset.is_number:
            # If resulted offset sympy expr is convertible to int constant it
            # will be directly encoded into `arith.constant`.
            # For non-constant expressions, we will generate a real sequence of
            # arith ops and then `vector.insertelement` them into offsets vec.
            offset = int(offset)
        else:
            need_dynamic_offsets = True
            break

        offsets.append(IntegerAttr.get(IndexType.get(), offset))

    offsets_vec_type = VectorType.get([elements_per_thread], IndexType.get())
    if need_dynamic_offsets or True:
        result_index = {key: 0 for key in symbolc_shape}
        start_indices = _build_start_indices(emitter, result_index)
        subs = [(sym, idx) for sym, idx in zip(iters.keys(), start_indices_orig)]
        off = start_indices_orig[-1] + idxc.iota(elements_per_thread) + thread_offset
        subs[-1] = (
            subs[-1][0],
            off,
        )
        indices = [i.subs(subs) for i in index_mapping]
        offsets_vec = _gen_sympy_index_full(emitter, _compute_offset(indices, strides))
        if not isinstance(offsets_vec, VectorType):
            offsets_vec = vector_d.splat(offsets_vec_type, offsets_vec)
    else:
        start_indices = _build_start_indices(emitter, result_index)
        offsets_vec = arith_d.ConstantOp(
            offsets_vec_type, DenseElementsAttr.get(offsets, offsets_vec_type)
        )

    mask = _build_mask(emitter, index, elements_per_thread)
    if mask is None:
        mask_vec_type = VectorType.get(
            [elements_per_thread], IntegerType.get_signless(1)
        )
        mask = vector_d.constant_mask(mask_vec_type, [elements_per_thread])

    return start_indices, offsets_vec, mask


@handle_op(read)
def handle_read(emitter: WaveEmitter, node: fx.Node):
    # This is similar to tkl.store with fixed start indices for now.
    try:
        memory, elements_per_thread, mapping, _ = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_shape = cast_py_literal(emitter, (elements_per_thread,))
    # memory has no IR node yet.
    kb_src, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, memory)

    if not hasattr(node, "index"):
        raise ValidationError("codegen expected read to have index attr.")

    index = node.index

    element_type = kb_ir_type.element_type
    vector_type = VectorType.get(vector_shape, element_type)
    input_shape = _get_symbolic_shape(memory)
    if mapping is None or _is_identity_mapping(mapping, input_shape=input_shape):
        start_indices = _build_start_indices(emitter, index)
        mask = _build_mask(
            emitter, index, cast_py_literal(emitter, elements_per_thread)
        )
        if (
            mask is None
            or get_custom(node).memory_type.address_space == SHARED_ADDRESS_SPACE
        ):
            result = vector_d.load(vector_type, kb_src, start_indices)
        else:
            zero = get_constant_attr(0, element_type)
            zero = arith_d.ConstantOp(vector_type.element_type, zero)
            passthru = vector_d.splat(vector_type, zero)

            result = vector_d.maskedload(
                vector_type, kb_src, start_indices, mask, passthru
            )
    else:
        zero = get_constant_attr(0, element_type)
        zero = arith_d.ConstantOp(element_type, zero)
        result = vector_d.splat(vector_type, zero)
        vec_type1 = VectorType.get([1], element_type)
        passthru = vector_d.splat(vec_type1, zero)
        for i in range(elements_per_thread):
            start_indices, offsets_vec, mask = _construct_gather_scatter_indices(
                emitter=emitter,
                symbolc_shape=input_shape,
                index=index,
                mapping=mapping,
                elements_per_thread=1,
                is_read=True,
                thread_offset=i,
            )

            res = vector_d.gather(
                vec_type1, kb_src, start_indices, offsets_vec, mask, passthru
            )
            res = vector_d.extract(res, [], [0])
            result = vector_d.insert(res, result, [], [i])

        # start_indices, offsets_vec, mask = _construct_gather_scatter_indices(
        #     emitter=emitter,
        #     symbolc_shape=input_shape,
        #     index=index,
        #     mapping=mapping,
        #     elements_per_thread=cast_py_literal(emitter, elements_per_thread),
        #     is_read=True,
        # )

        # zero = get_constant_attr(0, element_type)
        # zero = arith_d.ConstantOp(vector_type.element_type, zero)
        # passthru = vector_d.splat(vector_type, zero)

        # result = vector_d.gather(
        #     vector_type, kb_src, start_indices, offsets_vec, mask, passthru
        # )

    emitter.bind_node_proxy(node, IRProxyValue(result))


@handle_op(write)
def handle_write(emitter: WaveEmitter, node: fx.Node):
    try:
        register, memory, elements_per_thread, mapping = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # memory has no IR node yet.
    kb_dest, kb_ir_type, kb_py_type = cast_kernel_buffer(emitter, memory)
    insert_vector = cast_vector(emitter, register, element_type=kb_ir_type.element_type)
    insert_type = VectorType(insert_vector.type)
    vector_shape = cast_py_literal(emitter, (elements_per_thread,))

    # TODO: Support elements_per_thread size mismatch and broadcasting

    assert (
        tuple(insert_type.shape) == vector_shape
    ), f"Shape doesn't match: {tuple(insert_type.shape)} and {(vector_shape)}"

    if not hasattr(node, "index"):
        raise ValidationError("codegen expected read to have index attr.")

    index = node.index
    input_shape = _get_symbolic_shape(register)
    output_shape = _get_symbolic_shape(memory)
    if mapping is None or _is_identity_mapping(
        mapping, input_shape=input_shape, output_shape=output_shape
    ):
        start_indices = _build_start_indices(emitter, index)
        mask = _build_mask(
            emitter, index, cast_py_literal(emitter, elements_per_thread)
        )
        if (
            mask is None
            or get_custom(node).memory_type.address_space == SHARED_ADDRESS_SPACE
        ):
            vector_d.store(insert_vector, kb_dest, start_indices)
        else:
            vector_d.maskedstore(kb_dest, start_indices, mask, insert_vector)
    else:
        assert (
            input_shape == mapping.input_shape
        ), "non-identity input mapping is not supported yet"

        start_indices, offsets_vec, mask = _construct_gather_scatter_indices(
            emitter=emitter,
            symbolc_shape=output_shape,
            index=index,
            mapping=mapping,
            elements_per_thread=cast_py_literal(emitter, elements_per_thread),
            is_read=False,
        )

        if elements_per_thread == 1:
            vector_d.maskedstore(kb_dest, start_indices, mask, insert_vector)
        else:
            vector_d.scatter(kb_dest, start_indices, offsets_vec, mask, insert_vector)


###############################################################################
# Contraction/MMA Ops
###############################################################################


def emit_mfma(
    m: int, n: int, k: int, vector_type: VectorType, acc: Value, values: list[Value]
):
    m = get_constant_attr(m, IntegerType.get_signless(32))
    n = get_constant_attr(n, IntegerType.get_signless(32))
    k = get_constant_attr(k, IntegerType.get_signless(32))
    blocks = get_constant_attr(1, IntegerType.get_signless(32))

    result = amdgpu_d.mfma(
        dest_d=vector_type,
        m=m,
        n=n,
        k=k,
        blocks=blocks,
        source_a=values[0],
        source_b=values[1],
        dest_c=acc,
    )
    return result


@handle_op(mma)
def handle_mma(emitter: WaveEmitter, node: fx.Node):
    try:
        lhs, rhs, acc = node.args
        acc = cast_vector(emitter, acc)
        values = [cast_vector(emitter, val) for val in [lhs, rhs]]
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    vector_type = VectorType(acc.type)

    hardware_constraints = [
        constraint
        for constraint in emitter.constraints
        if isinstance(constraint, HardwareConstraint)
    ]
    if not hardware_constraints:
        raise CodegenError("No hardware constraints found.")

    m, n, k = hardware_constraints[0].mma_matrix_shapes
    result = emit_mfma(m, n, k, vector_type, acc, values)
    emitter.bind_node_proxy(node, IRProxyValue(result))


@handle_op(shuffle)
def handle_shuffle(emitter: WaveEmitter, node: fx.Node):
    """
    Generate gpu shuffle instruction to enable communication
    between threads in a warp. Currently we only support
    float unit vector that is <= 32 bits.

    Translation to shuffle is done in 3 steps:
    1. Scalarize (vector<1xf16> -> f16)
    2. Pad to 32-bit if needed(f16 -> f32)
    3. Shuffle (gpu.shuffle xor src, offset, width -> f32)
    4. Reconstruct to original vector type (truncf f32 -> f16, broadcast -> vector<1xf16>)

    TODO: Handle non-unit vector types such as vector<4xF8> (useful for resolving layouts).
    """
    try:
        src, offset, width = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e
    if not isinstance(offset, int) or not isinstance(width, int):
        raise NotImplementedError(
            "Non-const width or offset is not yet implemented for shuffleOp."
        )
    src = cast_py_value(emitter, src).ir_value
    offset = cast_py_value(emitter, offset, IntegerType.get_signless(32)).ir_value
    width = cast_py_value(emitter, width, IntegerType.get_signless(32)).ir_value

    if not VectorType.isinstance(src.type):
        raise NotImplementedError("Scalar src is not implemented yet for shuffleOp.")

    if math.prod(src.type.shape) != 1:
        raise NotImplementedError("Currently only support unit vector for shuffleOp.")

    # Scalarize (vector<FLOAT_TYPE> -> FLOAT_TYPE).
    static_pos = [0 for i in range(src.type.rank)]
    element = vector_d.extract(src, static_position=static_pos, dynamic_position=[])
    element_original_type = element.type

    # Pad to 32 bit if needed.
    # TODO Handle and pack non-unit vector type. i.e enable shuffling of vector<4xF8>
    #      in one shuffle instruction.
    if not _is_float_type(element.type):
        raise NotImplementedError("Currently only support shuffle for floats.")
    if element.type.width > 32:
        raise ValueError("Cannot shuffle more than 32 bit.")
    elif element.type.width < 32:
        element = arith_d.extf(F32Type.get(), element)

    # Shuffle data between other threads in a warp.
    result = gpu_d.shuffle(element, offset, width, gpu_d.ShuffleMode.XOR)

    # Reconstruct shuffled value to original shape and dtype.
    shuffled_val = result[0]
    if element_original_type != shuffled_val.type:
        shuffled_val = arith_d.truncf(element_original_type, shuffled_val)
    vec_result = vector_d.broadcast(src.type, shuffled_val)

    emitter.bind_node_proxy(node, IRProxyValue(vec_result))


###############################################################################
# Binary math Ops
###############################################################################


def handle_binary_op(op):
    def decorator(binary_fn: Callable[[Value, Value], OpResult]):
        @handle_op(op)
        def handle_generic_binary(emitter: WaveEmitter, node: fx.Node):
            try:
                lhs, rhs = node.args
            except ValueError as e:
                raise ValidationError("Malformed arguments") from e
            lhs = cast_py_value(emitter, lhs)
            rhs = cast_py_value(emitter, rhs)

            if lhs.ir_value.type != rhs.ir_value.type:
                raise ValidationError("Expected lhs and rhs to have same type.")

            lhs = lhs.ir_value
            rhs = rhs.ir_value
            result = binary_fn(lhs, rhs)

            emitter.bind_node_proxy(node, IRProxyValue(result))

    return decorator


@handle_binary_op(operator.add)
def handle_add(lhs: Value, rhs: Value) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.addf(lhs, rhs)
    elif _is_integer_like_type(element_type):
        result = arith_d.addi(lhs, rhs)
    else:
        raise ValidationError(f"Found unhandled operand type for add: {element_type}")
    return result


@handle_binary_op(operator.sub)
def handle_sub(lhs: Value, rhs: Value) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.subf(lhs, rhs)
    elif _is_integer_like_type(element_type):
        result = arith_d.subi(lhs, rhs)
    else:
        raise ValidationError(f"Found unhandled operand type for sub: {element_type}")
    return result


@handle_binary_op(operator.mul)
def handle_mul(lhs: Value, rhs: Value) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.mulf(lhs, rhs)
    elif _is_integer_like_type(element_type):
        result = arith_d.muli(lhs, rhs)
    else:
        raise ValidationError(f"Found unhandled operand type for mul: {element_type}")
    return result


@handle_binary_op(operator.truediv)
def handle_div(lhs: Value, rhs: Value) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.divf(lhs, rhs)
    elif _is_integer_like_type(element_type) and (
        element_type.is_signed() or element_type.is_signless()
    ):
        result = arith_d.divsi(lhs, rhs)
    elif _is_integer_like_type(element_type) and element_type.is_unsigned():
        result = arith_d.divui(lhs, rhs)
    else:
        raise ValidationError(f"Found unhandled operand type for div: {element_type}")
    return result


@handle_binary_op(maximum)
def handle_maximum(lhs: Value, rhs: Value) -> OpResult:
    element_type = get_type_or_element_type(lhs.type)
    if _is_float_type(element_type):
        result = arith_d.maximumf(lhs, rhs)
    elif _is_integer_like_type(element_type) and (
        element_type.is_signed() or element_type.is_signless()
    ):
        result = arith_d.maxsi(lhs, rhs)
    elif _is_integer_like_type(element_type) and element_type.is_unsigned():
        result = arith_d.maxui(lhs, rhs)
    else:
        raise ValidationError(
            f"Found unhandled operand type for maximum: {element_type}"
        )
    return result


###############################################################################
# Unary math Ops
###############################################################################


def handle_unary_op(op):
    def decorator(unary_fn: Callable[[Value, Value], OpResult]):
        @handle_op(op)
        def handle_generic_unary(emitter: WaveEmitter, node: fx.Node):
            try:
                (src,) = node.args
            except ValueError as e:
                raise ValidationError("Malformed arguments") from e
            src = cast_py_value(emitter, src)

            src = src.ir_value
            result = unary_fn(src)
            emitter.bind_node_proxy(node, IRProxyValue(result))

    return decorator


@handle_unary_op(operator.neg)
def handle_neg(source: Value) -> OpResult:
    element_type = get_type_or_element_type(source.type)
    if _is_float_type(element_type):
        result = arith_d.negf(source)
    else:
        raise ValidationError(
            f"Found unhandled operand type for negate: {element_type}"
        )
    return result


@handle_unary_op(exp2)
def handle_exp2(source: Value) -> OpResult:
    element_type = get_type_or_element_type(source.type)
    if _is_float_type(element_type):
        result = math_d.exp2(source)
    else:
        raise ValidationError(f"Found unhandled operand type for exp2: {element_type}")
    return result


###############################################################################
# Control Flow ops
###############################################################################


@handle_op(reduction)
def handle_reduction(emitter: WaveEmitter, node: fx.Node):
    try:
        axis, init_args, subgraph, implicit_capture = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # Flatten init_args and get IR values for each of them.
    flat_init_args, _ = pytree.tree_flatten((init_args))
    flat_init_args = [cast_py_value(emitter, arg) for arg in flat_init_args]

    start = arith_d.constant(IndexType.get(), int(0))

    # For now, we assume that dimensions that have tiling constraints on them,
    # do not have any other constraints.
    end = arith_d.constant(IndexType.get(), int(node.count))

    # Since we divide the end by the tile size, we need to make sure that the
    # step is 1.
    step = arith_d.constant(IndexType.get(), int(1))

    if not step:
        raise CodegenError(
            "Could not determine step size for reduction due to missing tiling constraint."
        )

    forOp = scf_d.ForOp(
        start,
        end,
        step,
        [a.ir_value for a in flat_init_args],
    )
    emitter.induction_vars[axis] = forOp.induction_variable
    with InsertionPoint(forOp.body):
        # Add mapping for iter args.
        subgraph: fx.Graph = emitter.trace.get_subgraph(subgraph)
        iter_args: list[fx.Node] = get_custom(node).iter_args(subgraph)
        for i, v in enumerate(forOp.inner_iter_args):
            emitter.bind_node_proxy(iter_args[i], IRProxyValue(v))
        captured_vars: list[fx.Node] = get_custom(node).captured_vars(subgraph)
        for root_v, subgraph_v in zip(implicit_capture, captured_vars):
            emitter._node_values[subgraph_v] = emitter.lookup_node_values(root_v)
        # Emit the subgraph.
        return_values = emitter._emit_graph(subgraph)
        # Flattern return values.
        flat_ret_values, _ = pytree.tree_flatten((return_values))
        flat_ret_values = [
            cast_py_value(emitter, value).ir_value for value in flat_ret_values
        ]
        scf_d.YieldOp(flat_ret_values)

    emitter.bind_node_proxies(node, [IRProxyValue(v) for v in forOp.results_])


###############################################################################
# Synchronization ops
###############################################################################


@handle_op(shared_memory_barrier)
def handle_shared_memory_barrier(emitter: WaveEmitter, node: fx.Node):
    amdgpu_d.lds_barrier()


###############################################################################
# Slicing ops
###############################################################################


@handle_op(extract_slice)
def handle_extract_slice(emitter: WaveEmitter, node: fx.Node):
    try:
        register, offsets, sizes, strides = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    extract_vector = cast_vector(emitter, register)
    result_type = VectorType.get(sizes, extract_vector.type.element_type)
    element = vector_d.extract_strided_slice(
        result_type,
        extract_vector,
        offsets,
        sizes,
        strides,
    )

    emitter.bind_node_proxy(node, IRProxyValue(element))


###############################################################################
# Reshape ops
###############################################################################


@handle_op(broadcast)
def handle_broadcast(emitter: WaveEmitter, node: fx.Node):
    try:
        register, target_type = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    # Get thread_shape/size for broadcast.
    get_thread_shape = lambda index: max(x.size for x in index.values())
    bcast_dim_lane_dim_size = get_thread_shape(node.index)

    # Check MLIR shape
    vector_src = cast_vector(emitter, register)
    vector_type = vector_src.type
    # Only support broadcasting vector<1xdtype> for now.
    if not VectorType.isinstance(vector_type):
        raise NotImplementedError("Scalar src is not implemented yet for shuffleOp.")
    assert vector_type.rank == 1
    assert vector_type.shape[0] == 1

    # Extract and Splat
    # If by chance broadcast size  matches current size, we can return src.
    if bcast_dim_lane_dim_size == vector_type.shape[0]:
        emitter.bind_node_proxy(node, IRProxyValue(vector_src))

    result_type = VectorType.get([bcast_dim_lane_dim_size], vector_type.element_type)
    element = vector_d.extract(vector_src, static_position=[0], dynamic_position=[])
    splat = vector_d.splat(result_type, element)
    emitter.bind_node_proxy(node, IRProxyValue(splat))


###############################################################################
# Miscellanous ops
###############################################################################


@handle_op(get_result)
def handle_get_result(emitter: WaveEmitter, node: fx.Node):
    try:
        value, res_idx = node.args
    except ValueError as e:
        raise ValidationError("Malformed arguments") from e

    for_op = emitter.lookup_node_values(value)[0].owner
    emitter.bind_node_proxy(node, IRProxyValue(for_op.results[res_idx]))


@handle_op(operator.getitem)
def handle_getitem(emitter: WaveEmitter, node: fx.Node):
    raise NotImplementedError("getitem: Currently only stub implementation")
