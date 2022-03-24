from multiprocessing import parent_process
from shutil import ExecError
import sys

from circulo import circulo

cir  = circulo(radio=10, color="Red")

print(cir.radio)

cir.agrandarRadio(2)

print(cir.radio)