import numpy as np
import grafica_Grad as gr
import math

# z = 3*x^2 + y^2;

[x, y, h] = gr.graficoGradientePy(1)

z = 3*x**2 + y**2

# Guardamos el pto. actual
PtoAnt = [x, y, z]
x = x-1
y = y-1

z = 3*x**2 + y**2
gr.graficarPaso(PtoAnt, [x, y, z], h)
    