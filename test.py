import numpy as np

import autodiff as ad

t = 1e-8

def noise_like(x):
  return np.random.uniform(-1, 1, np.shape(x))

def run_problem3():
  print("-" * 18 + " Problem 3 " + "-" * 18)
  x = ad.Variable()
  w1 = ad.Variable()
  w2 = ad.Variable()
  y = ad.average(ad.matmul(ad.relu(ad.matmul(x, w1)), w2) + x)

  x_f = np.random.randn(1, 64)
  w1_f = np.random.randn(64, 128)
  w2_f = np.random.randn(128, 64)

  x_v = noise_like(x_f)
  w1_v = noise_like(w1_f)
  w2_v = noise_like(w2_f)

  f, grad = ad.func(y, {x: x_f, w1: w1_f, w2: w2_f}, get_gradient=True)
  f_np = np.average(np.matmul(np.maximum(np.matmul(x_f, w1_f), 0), w2_f) + x_f)
  print("Function value by autodiff =", f)
  print("Function value by numpy    =", f_np)

  lhs = (ad.func(y, {x: x_f + t * x_v, w1: w1_f + t * w1_v,
                     w2: w2_f + t * w2_v}) - f) / t
  rhs = (np.sum(grad[x] * x_v) + np.sum(grad[w1] * w1_v)
         + np.sum(grad[w2] * w2_v))
  print("(V(w + tv)-V(w)) / t       =", lhs)
  print("<dV(w), v>                 =", rhs)
  print("|lfs - rhs| / lhs          =", np.abs(lhs - rhs) / lhs)

def run_problem4():
  print("-" * 18 + " Problem 4 " + "-" * 18)
  x1 = ad.Variable()
  x2 = ad.Variable()
  x3 = ad.Variable()
  y = ((ad.sin(x1 + 1) + ad.cos(2 * x2)) * ad.tan(ad.log(x3))
       + (ad.sin(x2 + 1) + ad.cos(2 * x1)) * ad.exp(1 + ad.sin(x3)))

  x1_f = np.random.rand()
  x2_f = np.random.rand()
  x3_f = np.random.rand()

  x1_v = noise_like(x1_f)
  x2_v = noise_like(x2_f)
  x3_v = noise_like(x3_f)

  f, grad = ad.func(y, {x1: x1_f, x2: x2_f, x3: x3_f}, get_gradient=True)
  f_np = ((np.sin(x1_f + 1) + np.cos(2 * x2_f)) * np.tan(np.log(x3_f))
          + (np.sin(x2_f + 1) + np.cos(2 * x1_f)) * np.exp(1 + np.sin(x3_f)))
  print("Function value by autodiff =", f)
  print("Function value by numpy    =", f_np)

  lhs = (ad.func(y, {x1: x1_f + t * x1_v, x2: x2_f + t * x2_v,
                     x3: x3_f + t * x3_v}) - f) / t
  rhs = (np.sum(grad[x1] * x1_v) + np.sum(grad[x2] * x2_v)
         + np.sum(grad[x3] * x3_v))
  print("(V(w + tv)-V(w)) / t       =", lhs)
  print("<dV(w), v>                 =", rhs)
  print("|lfs - rhs| / lhs          =", np.abs(lhs - rhs) / lhs)
run_problem3()
run_problem4()
