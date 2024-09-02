import numpy as np
from grafica_Grad import *
import math

#z = -3/(x^2 + y^2 + 1)

[x, y, dibu] = graficoGradientePy(3)
x = 1.5
y=1.5

alfa = 0.5

MAX_ITE = 100

ite = 0
z = 1
z_new = -3/(x**2 + y**2 + 1)
z=z_new-1

while ((ite<MAX_ITE) and (math.fabs(z - z_new)>10e-06)):
    z = z_new
    PtoAnt = [x, y, z]
    grad_x = 6*x/(x**2+y**2+1)**2
    grad_y = 6*y/(x**2+y**2+1)**2
    
    x = x - alfa * grad_x
    y = y - alfa * grad_y
    z_new = -3/(x**2 + y**2 + 1)
    
    graficarPaso(PtoAnt, [x, y, z_new], dibu)
    ite = ite + 1

    print ("ite= %d   x= %8.5f   y=%8.5f   z=%.8f" % (ite,x,y,z_new))