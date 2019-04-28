
import  random
import  numpy as np
import matplotlib.pyplot as pt
import csv
X = np.linspace(1, 15, 100, endpoint=True)

Y=1.477*X+0.089+random.gauss(0,1)+random.gauss(0, 1)

# 画出这些点
pt.plot(X, Y, linestyle='', marker='.')
pt.show()
#‘a’是追加模式，可以改成'w'——覆盖写模式
with open('data.csv','w') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(0, len(X)):
            writer.writerow([X[i],Y[i]])

def err_function(b,w,points):
    totalError=0
    for i in  range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        totalError+=(y-(w*x+b))**2
    return totalError/float(len(points))

def step_gradient(b,w,points,learningRate):
    b_gradient=0
    w_gradient=0
    N=float(len(points))
    for i in range(0,len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += (2/N)*((w*x+b)-y)
        w_gradient += (2 / N)*x * ((w * x + b) - y)
    new_b=b-(learningRate*b_gradient)
    new_w=w-(learningRate*w_gradient)
    return [new_b,new_w]
def gradient_run(points,start_b,start_w,learningRate,num_iterations):
    b=start_b
    w=start_w
    for i in range(num_iterations):
        b,w=step_gradient(b,w,np.array(points),learningRate)
    return [b,w]

def run():
    points=np.genfromtxt("data.csv",delimiter=",")
    learningRate=0.0001
    init_b=0
    init_w=0
    num_iterations=100000
    print("Starting gradient dencent at b={0},w={1},error={2}".format(init_b,init_w,err_function(init_b,init_w,points)))

    print("running...")
    [b,w]=gradient_run(points,init_b,init_w,learningRate,num_iterations)
    print("after {3}  iterations b={0},w={1},error={2}".format(b,w,err_function(b,w,points),num_iterations))
if __name__=='__main__':
    run()



