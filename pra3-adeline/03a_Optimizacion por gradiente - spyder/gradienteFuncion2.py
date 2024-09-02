import numpy as np
import grafica_Grad as gr
import math

# Z = (X**2 * Y * math.pi)/3 

[x, y, h] = gr.graficoGradientePy(2)

alfa = 0.1

MAX_ITE = 100  
ite = 1
z = 1
z_new = x**2 * y * math.pi/3

while ((ite<MAX_ITE) and (math.fabs(z - z_new)>10e-6)):
    z = z_new
    PtoAnt = [x, y, z]
    grad_x = 2 * x * y * math.pi/3   # derivada respecto de x
    grad_y = x**2 * math.pi/3   # derivada respecto de y
    
    x = x - alfa * grad_x
    y = y - alfa * grad_y
    z_new = x**2 * y * math.pi/3
    
    gr.graficarPaso(PtoAnt, [x, y, z_new], h)
    ite = ite + 1

print(f" fdsafdsa fasdfdas {x},{y}")

#print ("ite = %d   x= %.5f   y=%.5f   z=%.8f" % (ite,x,y,z_new))

