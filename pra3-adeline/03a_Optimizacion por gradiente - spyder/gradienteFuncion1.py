import numpy as np
import grafica_Grad as gr
import math

# z = 3*x^2 + y^2;

[x, y, h] = gr.graficoGradientePy(1)

# z = 3*x**2 + y**2
# PtoAnt = [x, y, z]
# x = x-1  
# y = y-2  #cambiamos x e y
# z = 3*x**2 + y**2;
# gr. graficarPaso(PtoAnt, [x, y, z], h)

alfa = 0.05
MAX_ITE = 100  
ite = 1
z = 1
z_new = 3*x**2 + y**2

while ((ite<MAX_ITE) and (math.fabs(z - z_new)>0.0001)):
    z = z_new
    PtoAnt = [x, y, z]
    grad_x = 6*x   # derivada respecto de x
    grad_y = 2*y   # derivada respecto de y
    
    x = x - alfa * grad_x
    y = y - alfa * grad_y
    z_new = 3*x**2 + y**2
    
    gr.graficarPaso(PtoAnt, [x, y, z_new], h)
    ite = ite + 1

