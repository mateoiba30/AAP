from grafica_Grad import *
import math

# z = 3*w0^2 + w1^2;

[w0, w1, dibu] = graficoGradientePy(4)

alfa = 0.05
MAX_ITE = 100
ite = 1
E_ant = 1
E = (1/3)*((3-2*w1-w0)**2+(1-w1-w0)**2+(-3+w1-w0)**2)

while ((ite<MAX_ITE) and (math.fabs(E_ant - E)>0.001)):
    E_ant=E
    PtoAnt = [w0, w1, E]

    grad_w0 = -(2/3)*((3-2*w1-w0)+(1-w1-w0)+(-3+w1-w0))
    grad_w1 = (2/3)*((-2)*(3-2*w1-w0)-(1-w1-w0)+(-3+w1-w0))
    
    w0 = w0 - alfa * grad_w0
    w1 = w1 - alfa * grad_w1
    E = (1/3)*((3-2*w1-w0)**2+(1-w1-w0)**2+(-3+w1-w0)**2)
   
    graficarPaso(PtoAnt, [w0, w1, E], dibu)
    ite = ite + 1

