# test_compiler.py
import pytensor
print("PyTensor config:")
print(f"Compiledir: {pytensor.config.compiledir}")
print(f"CXX: {pytensor.config.cxx}")

# Test simple de compilation
import pytensor.tensor as pt
x = pt.dscalar('x')
f = pytensor.function([x], x**2)
print("Compilation test successful!")
print(f"Test result: {f(4.0)}")