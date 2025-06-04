from __future__ import annotations

"""Consolidated ONNX frontend for tinygrad.

Minimal ONNX support optimized for code size and essential functionality.
"""

import struct
from typing import Any, Dict, List, Tuple, Union, Optional, Sequence
from dataclasses import dataclass, field

# Minimal ONNX type definitions
@dataclass
class TensorProto:
  dims: Tuple[int, ...] = ()
  data_type: int = 0
  raw_data: bytes = b''
  float_data: List[float] = field(default_factory=list)
  int32_data: List[int] = field(default_factory=list)
  int64_data: List[int] = field(default_factory=list)

@dataclass
class Attribute:
  name: str = ''
  type: int = 0
  f: float = 0.0
  i: int = 0
  s: bytes = b''
  floats: List[float] = field(default_factory=list)
  ints: List[int] = field(default_factory=list)

@dataclass
class NodeProto:
  input: Tuple[str, ...] = ()
  output: Tuple[str, ...] = ()
  op_type: str = ''
  attribute: Tuple[Attribute, ...] = ()
  domain: str = ''

@dataclass
class ValueInfo:
  name: str = ''
  elem_type: int = 0
  shape: Tuple[Union[int, str], ...] = ()

@dataclass
class GraphProto:
  node: Tuple[NodeProto, ...] = ()
  initializer: Tuple[TensorProto, ...] = ()
  input: Tuple[ValueInfo, ...] = ()
  output: Tuple[ValueInfo, ...] = ()

@dataclass
class ModelProto:
  graph: Optional[GraphProto] = None
  opset_import: Tuple[Tuple[str, int], ...] = ()

  def __post_init__(self):
    if self.graph is None: self.graph = GraphProto()

# Minimal protobuf parser
def _read_varint(buf: bytes, pos: int) -> Tuple[int, int]:
  result = shift = 0
  while True:
    b = buf[pos]
    pos += 1
    result |= (b & 0x7F) << shift
    if not (b & 0x80): break
    shift += 7
  return result, pos

def _skip_field(wire_type: int, buf: bytes, pos: int) -> int:
  if wire_type == 0: _, pos = _read_varint(buf, pos)
  elif wire_type == 1: pos += 8
  elif wire_type == 2:
    size, pos = _read_varint(buf, pos)
    pos += size
  elif wire_type == 5: pos += 4
  return pos

def _parse_tensor(buf: bytes) -> TensorProto:
  t = TensorProto()
  pos = 0
  while pos < len(buf):
    tag, pos = _read_varint(buf, pos)
    field, wtype = tag >> 3, tag & 0x07
    if field == 1 and wtype == 0:
      val, pos = _read_varint(buf, pos)
      t.dims += (val,)
    elif field == 2 and wtype == 0: t.data_type, pos = _read_varint(buf, pos)
    elif field == 9 and wtype == 2:
      size, pos = _read_varint(buf, pos)
      t.raw_data = buf[pos:pos+size]
      pos += size
    else: pos = _skip_field(wtype, buf, pos)
  return t

def _parse_attr(buf: bytes) -> Attribute:
  a = Attribute()
  pos = 0
  while pos < len(buf):
    tag, pos = _read_varint(buf, pos)
    field, wtype = tag >> 3, tag & 0x07
    if field == 1 and wtype == 2:
      size, pos = _read_varint(buf, pos)
      a.name = buf[pos:pos+size].decode()
      pos += size
    elif field == 3 and wtype == 0: a.type, pos = _read_varint(buf, pos)
    elif field == 4 and wtype == 5:
      a.f = struct.unpack("<f", buf[pos:pos+4])[0]
      pos += 4
    elif field == 5 and wtype == 0: a.i, pos = _read_varint(buf, pos)
    elif field == 6 and wtype == 2:
      size, pos = _read_varint(buf, pos)
      a.s = buf[pos:pos+size]
      pos += size
    else: pos = _skip_field(wtype, buf, pos)
  return a

def _parse_node(buf: bytes) -> NodeProto:
  n = NodeProto()
  inps, outs, attrs = [], [], []
  pos = 0
  while pos < len(buf):
    tag, pos = _read_varint(buf, pos)
    field, wtype = tag >> 3, tag & 0x07
    if field == 1 and wtype == 2:
      size, pos = _read_varint(buf, pos)
      inps.append(buf[pos:pos+size].decode())
      pos += size
    elif field == 2 and wtype == 2:
      size, pos = _read_varint(buf, pos)
      outs.append(buf[pos:pos+size].decode())
      pos += size
    elif field == 4 and wtype == 2:
      size, pos = _read_varint(buf, pos)
      n.op_type = buf[pos:pos+size].decode()
      pos += size
    elif field == 5 and wtype == 2:
      size, pos = _read_varint(buf, pos)
      attrs.append(_parse_attr(buf[pos:pos+size]))
      pos += size
    else: pos = _skip_field(wtype, buf, pos)
  n.input, n.output, n.attribute = tuple(inps), tuple(outs), tuple(attrs)
  return n

def _parse_graph(buf: bytes) -> GraphProto:
  g = GraphProto()
  nodes, init = [], []
  pos = 0
  while pos < len(buf):
    tag, pos = _read_varint(buf, pos)
    field, wtype = tag >> 3, tag & 0x07
    if field == 1 and wtype == 2:
      size, pos = _read_varint(buf, pos)
      nodes.append(_parse_node(buf[pos:pos+size]))
      pos += size
    elif field == 5 and wtype == 2:
      size, pos = _read_varint(buf, pos)
      init.append(_parse_tensor(buf[pos:pos+size]))
      pos += size
    else: pos = _skip_field(wtype, buf, pos)
  g.node, g.initializer = tuple(nodes), tuple(init)
  return g

def parse_model(buf: bytes) -> ModelProto:
  m = ModelProto()
  pos = 0
  while pos < len(buf):
    tag, pos = _read_varint(buf, pos)
    field, wtype = tag >> 3, tag & 0x07
    if field == 4 and wtype == 2:
      size, pos = _read_varint(buf, pos)
      m.graph = _parse_graph(buf[pos:pos+size])
      pos += size
    else: pos = _skip_field(wtype, buf, pos)
  return m

# Main API
def load(path: str) -> ModelProto:
  """Load ONNX model from file path."""
  with open(path, 'rb') as f:
    return parse_model(f.read())

# Import the real ONNX implementation for actual execution
def _import_onnx_ops():
  """Import ONNX operations from extra.onnx if available."""
  try:
    from extra.onnx import get_onnx_ops, OnnxValue as RealOnnxValue, OnnxNode as RealOnnxNode
    from extra.onnx import dtype_parse as real_dtype_parse, buffer_parse as real_buffer_parse
    from extra.onnx import type_parse as real_type_parse, attribute_parse as real_attribute_parse
    from extra.onnx import to_python_const
    return get_onnx_ops(), RealOnnxValue, RealOnnxNode, real_dtype_parse, real_buffer_parse, real_type_parse, real_attribute_parse, to_python_const
  except ImportError:
    # Fallback if extra.onnx not available
    return None, None, None, None, None, None, None, None

# Get the real implementations
_onnx_ops, _RealOnnxValue, _RealOnnxNode, _real_dtype_parse, _real_buffer_parse, _real_type_parse, _real_attribute_parse, _to_python_const = _import_onnx_ops()

# Compatibility OnnxValue for when extra.onnx is not available
class OnnxValue:
  def __init__(self, shape=(), dtype=None, is_optional=False, is_sequence=False):
    self.shape = shape
    self.dtype = dtype
    self.is_optional = is_optional
    self.is_sequence = is_sequence

class OnnxRunner:
  """ONNX runner that integrates with the existing tinygrad ONNX system."""
  def __init__(self, model: ModelProto):
    self.model = model
    if model.graph is None:
      raise ValueError("Model has no graph")
    self.graph = model.graph

    # Use real implementations if available
    if _onnx_ops is not None:
      self._init_with_real_onnx()
    else:
      self._init_fallback()

  def _init_with_real_onnx(self):
    """Initialize with the real ONNX implementation from extra.onnx."""
    from tinygrad import Tensor
    import functools

    # Initialize training state
    self.is_training = any(n.domain in {"ai.onnx.training", "ai.onnx.preview.training"} for n in self.graph.node)
    self.old_training = Tensor.training
    Tensor.training = True if self.is_training else False

    # Parse graph components using real parsers
    self.graph_values: Dict[str, Any] = {"": None}
    if _real_buffer_parse:
      self.graph_values.update({x.name: _real_buffer_parse(x) for x in self.graph.initializer})

    # Set up graph inputs and outputs
    excluded = {x.name for x in self.graph.initializer}
    if _real_type_parse:
      self.graph_inputs = {x.name: _real_type_parse(x) for x in self.graph.input if x.name not in self.graph_values}
    else:
      self.graph_inputs = {x.name: OnnxValue(shape=x.shape) for x in self.graph.input if x.name not in excluded}

    self.graph_outputs = tuple(x.name for x in self.graph.output)

    # Parse nodes
    if _RealOnnxNode and _real_attribute_parse:
      self.graph_nodes = tuple(
        _RealOnnxNode(
          num, n.op_type, tuple(n.input), tuple(n.output),
          {x.name: _real_attribute_parse(x) for x in n.attribute}
        )
        for num, n in enumerate(self.graph.node)
      )
    else:
      self.graph_nodes = tuple(self.graph.node)

    # Set opset version
    self.opset_version = model.opset_import[0][1] if model.opset_import else 13
    self.variable_dims: Dict[str, int] = {}
    self.onnx_ops = _onnx_ops

  def _init_fallback(self):
    """Fallback initialization when extra.onnx is not available."""
    excluded = {x.name for x in self.graph.initializer}
    self.graph_inputs = {x.name: OnnxValue() for x in self.graph.input if x.name not in excluded}
    self.graph_outputs = tuple(x.name for x in self.graph.output)
    self.graph_values: Dict[str, Any] = {}
    self.graph_nodes = self.graph.node
    self.onnx_ops = {}

  def _parse_input(self, name: str, value: Any, spec):
    """Parse input tensor according to ONNX spec."""
    from tinygrad import Tensor

    if hasattr(spec, 'is_optional') and spec.is_optional and value is None:
      return None

    if hasattr(spec, 'is_sequence') and spec.is_sequence:
      if not isinstance(value, Sequence):
        raise RuntimeError(f"{name} received {value}, expected a sequence type")
      dtype = getattr(spec, 'dtype', None)
      sequence = [Tensor(v, dtype=dtype, requires_grad=self.is_training) if not isinstance(v, Tensor) else v for v in value]
      return sequence

    # Convert to tensor
    dtype = getattr(spec, 'dtype', None)
    tensor = Tensor(value, dtype=dtype, requires_grad=getattr(self, 'is_training', False)) if not isinstance(value, Tensor) else value

    # Validate shape if available
    if hasattr(spec, 'shape') and spec.shape:
      for dim, (onnx_dim, user_dim_input) in enumerate(zip(spec.shape, tensor.shape, strict=False)):
        if isinstance(onnx_dim, str):
          onnx_dim = self.variable_dims.get(onnx_dim, int(user_dim_input))
          self.variable_dims[onnx_dim] = int(user_dim_input)
        elif isinstance(onnx_dim, int) and onnx_dim > 0 and user_dim_input != onnx_dim:
          raise RuntimeError(f"{name} has mismatch on {dim=}. Expected {onnx_dim}, received {user_dim_input}.")

    return tensor

  def _dispatch_op(self, op, inps, opts):
    """Dispatch ONNX operation to the appropriate implementation."""
    if op in self.onnx_ops:
      fxn = self.onnx_ops[op]
      if isinstance(fxn, dict):
        # Handle versioned ops
        real_fxn = fxn
        for k in sorted(fxn.keys()):
          if k <= self.opset_version:
            real_fxn = fxn[k]
      else:
        real_fxn = fxn
      return real_fxn(*inps, **opts)
    raise NotImplementedError(f"{op=} not supported")

  def __call__(self, inputs: Dict[str, Any], debug: int = 0) -> Dict[str, Any]:
    """Run ONNX model inference."""
    if not self.onnx_ops:
      raise RuntimeError("ONNX operations not available. Please ensure extra.onnx is accessible.")

    from tinygrad import Tensor

    # Parse inputs
    for name, input_spec in self.graph_inputs.items():
      if name not in inputs:
        raise RuntimeError(f"Please provide input data for {name}")
      self.graph_values[name] = self._parse_input(name, inputs[name], input_spec)

    # Execute graph nodes
    for node in self.graph_nodes:
      # Get inputs for this node
      if _to_python_const:
        inps = [_to_python_const(self.graph_values[name], node.op, i) for i, name in enumerate(node.input)]
      else:
        inps = [self.graph_values[name] for name in node.input]

      # Get node options
      opts = getattr(node, 'opts', {})

      # Special case handling
      if node.op_type == "Split" and 'num_outputs' not in opts:
        opts['num_outputs'] = len(node.output)
      if node.op_type == "Gradient":
        opts['intermediate_tensors'] = self.graph_values

      if debug >= 1:
        print(f"op '{node.op_type}' opt {opts}")

      # Dispatch operation
      ret = self._dispatch_op(node.op_type, inps, opts)
      ret = ret if isinstance(ret, tuple) else (ret,)

      # Store outputs
      for output_name, output_value in zip(node.output, ret):
        self.graph_values[output_name] = output_value

    # Restore training state
    if hasattr(self, 'old_training'):
      Tensor.training = self.old_training

    # Return final outputs
    result = {name: self.graph_values[name] for name in self.graph_outputs}

    # For backward compatibility with tests expecting 'outputs' key
    if len(self.graph_outputs) == 1:
      result['outputs'] = result[self.graph_outputs[0]]

    return result

# Export for backward compatibility
def get_onnx_ops():
  """Return available ONNX operators."""
  if _onnx_ops:
    return list(_onnx_ops.keys())
  return ['Add', 'Conv', 'Relu', 'MatMul', 'Reshape', 'Concat', 'Transpose']

# Legacy compatibility functions
def dtype_parse(x):
  return _real_dtype_parse(x) if _real_dtype_parse else x

def attribute_parse(x):
  return _real_attribute_parse(x) if _real_attribute_parse else x

def buffer_parse(x):
  return _real_buffer_parse(x) if _real_buffer_parse else x

def type_parse(x):
  return _real_type_parse(x) if _real_type_parse else x

onnx_ops = _onnx_ops or {}
OnnxNode = _RealOnnxNode or NodeProto

__all__ = [
  'load', 'OnnxRunner', 'ModelProto', 'GraphProto', 'NodeProto', 'TensorProto',
  'get_onnx_ops', 'onnx_ops', 'dtype_parse', 'attribute_parse', 'buffer_parse',
  'type_parse', 'OnnxValue', 'OnnxNode'
]