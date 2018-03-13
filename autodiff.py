import numpy as np
from collections import OrderedDict
from queue import Queue

class Operator:
  _num_of_inputs = -1  # means no constrains of number of inputs

  def forward(self, x):
    """
    Operator forward.
    :param x: list of inputs
    :return: output value y according to input x
    """
    raise NotImplementedError("Abstract method.")

  def backward(self, x, y, dy):
    """
    Operator backward.
    :param x: list of inputs
    :param y: output value y according to input x
    :param dy: derivative of y
    :return: list of dx
    """
    raise NotImplementedError("Abstract method.")

  def __call__(self, *x):
    if self._num_of_inputs >= 0:
      assert len(x) == self._num_of_inputs
    x = [x_ if isinstance(x_, Variable) else Constant(x_) for x_ in x]
    return Variable(self, x)


def broadcast_grad(dx, x):
  # Since some operators in numpy have broadcast property,
  # we have some special treatment here.
  while np.ndim(dx) > np.ndim(x):
    dx = dx.sum(0)
    dx = np.array(dx)
  for (i, n) in enumerate(np.shape(dx)):
    if n == 1:
      dx = dx.sum(i, keepdims=True)
  return dx


class Sin(Operator):
  _num_of_inputs = 1

  def forward(self, x):
    return np.sin(x[0])

  def backward(self, x, y, dy):
    return [dy * np.cos(x[0])]


class Cos(Operator):
  _num_of_inputs = 1

  def forward(self, x):
    return np.cos(x[0])

  def backward(self, x, y, dy):
    return [dy * -np.sin(x[0])]


class Tan(Operator):
  _num_of_inputs = 1

  def forward(self, x):
    return np.tan(x[0])

  def backward(self, x, y, dy):
    return [dy * (1 + y ** 2)]


class Exp(Operator):
  _num_of_inputs = 1

  def forward(self, x):
    return np.exp(x[0])

  def backward(self, x, y, dy):
    return [dy * y]


class Log(Operator):
  _num_of_inputs = 1

  def forward(self, x):
    return np.log(x[0])

  def backward(self, x, y, dy):
    return [dy / x]


class Add(Operator):
  _num_of_inputs = 2

  def forward(self, x):
    return np.add(x[0], x[1])

  def backward(self, x, y, dy):
    return [broadcast_grad(dy, x_) for x_ in x]


class Multiply(Operator):
  _num_of_inputs = 2

  def forward(self, x):
    return np.multiply(x[0], x[1])

  def backward(self, x, y, dy):
    return [broadcast_grad(dy * x[1], x[0]),
            broadcast_grad(dy * x[0], x[1])]


class MatMul(Operator):
  _num_of_inputs = 2

  def forward(self, x):
    return np.matmul(x[0], x[1])

  def backward(self, x, y, dy):
    extend_dim = [False, False]
    if np.ndim(x[0]) == 1:
      x[0] = np.expand_dims(x[0], axis=0)
      dy = np.expand_dims(dy, axis=0)
      extend_dim[0] = True
    if np.ndim(x[1]) == 1:
      x[1] = np.expand_dims(x[1], axis=1)
      dy = np.expand_dims(dy, axis=1)
      extend_dim[1] = True
    dx = [np.matmul(dy, np.swapaxes(x[1], axis1=-1, axis2=-2)),
          np.matmul(np.swapaxes(x[0], axis1=-1, axis2=-2), dy)]
    dx = [broadcast_grad(dx_, x_) for dx_, x_ in zip(dx, x)]
    if extend_dim[0]:
      dx[0] = np.squeeze(dx[0], axis=0)
    if extend_dim[1]:
      dx[1] = np.squeeze(dx[1], axis=1)
    return dx


class RELU(Operator):
  _num_of_inputs = 1

  def forward(self, x):
    return np.maximum(x[0], 0)

  def backward(self, x, y, dy):
    return [dy * np.where(np.greater(x[0], 0.0), 1.0, 0.0)]


class Average(Operator):
  _num_of_inputs = 1

  def forward(self, x):
    return np.average(x)

  def backward(self, x, y, dy):
    return [dy / np.size(x[0]) * np.ones_like(x[0])]


class ConstantOperator(Operator):
  def __init__(self, constant):
    constant = np.array(constant)
    self.constant = constant

  def forward(self, x):
    return self.constant

  def backward(self, x, y, dy):
    return []


sin = Sin()
cos = Cos()
tan = Tan()
exp = Exp()
log = Log()
add = Add()
multiply = Multiply()
matmul = MatMul()
relu = RELU()
average = Average()


class Variable:
  def __init__(self, operator=None, variables=None):
    self.operator = operator
    self.variables = [] if variables is None else variables

  def __add__(self, other):
    return add(self, other)

  def __radd__(self, other):
    return add(other, self)

  def __mul__(self, other):
    return multiply(self, other)

  def __rmul__(self, other):
    return multiply(other, self)


class Constant(Variable):
  def __init__(self, constant):
    super().__init__(ConstantOperator(constant))


def func(output, feed_dict=None, get_gradient=False):
  """
  Compute the value of output with the given feed_dict.
  If get_gradient is True, the function will also compute the gradient of
  all the involved parameters. When output is not a scalar, the output
  gradient will be the sum of the gradients of all categories of output,
  which performs the same as tf.gradient in TensorFlow.
  :param output: the needed output value
  :param feed_dict: value of inputs, formated as a dict
  :param get_gradient: whether needs gradients
  :return: the value of output given feed_dict, and a dict of gradients of
           all variables if get_gradient is True
  """
  feed_dict = {} if feed_dict is None else feed_dict
  for i in feed_dict:
    feed_dict[i] = np.array(feed_dict[i])
  if output in feed_dict:
    return feed_dict[output]
  # store all vars with their values in reversed topology order
  var_dict = OrderedDict([(output, None)])
  var_queue = Queue()
  var_queue.put(output)
  while not var_queue.empty():
    var = var_queue.get()
    for child in var.variables:
      if child not in var_dict:
        if child in feed_dict:
          var_dict[child] = feed_dict[child]
        else:
          var_dict[child] = None
          var_queue.put(child)
  for var in reversed(var_dict):
    if var_dict[var] is None:
      child_values = [var_dict[child] for child in var.variables]
      for child_value in child_values:
        if child_value is None:
          raise ValueError("Loops in computational graph "
                           "or uninitialized inputs")
      var_dict[var] = var.operator.forward(child_values)
  if not get_gradient:
    return var_dict[output]
  grad_dict = {output: np.ones_like(var_dict[output])}
  for var in var_dict:
    if var not in feed_dict and var.operator is not None:
      child_values = [var_dict[child] for child in var.variables]
      dx = var.operator.backward(child_values, var_dict[var], grad_dict[var])
      for i, child in enumerate(var.variables):
        if child in grad_dict:
          grad_dict[child] += dx[i]
        else:
          grad_dict[child] = dx[i]
  return var_dict[output], grad_dict
