__author__ = 'naveen'
# Libs import
import numpy as np
import theano
from theano import tensor as T
import matplotlib.pyplot as plt
import matplotlib.animation as animation


a = T.scalar()
b = T.scalar()

y = a*b

multiply = theano.function(inputs=[a, b], outputs=y)

print multiply(2,3)
print multiply(3.4,9.5)


X_tr = np.linspace(-1,1,101)
y_tr = 2 * X_tr + np.random.randn(*X_tr.shape)*.33

def model(X,w):
    return X*w

X = T.scalar()
Y = T.scalar()

w = theano.shared(np.asarray(0., dtype=theano.config.floatX))
y = model(X,w)

cost = T.mean(T.sqr(y - Y))
gradient = T.grad(cost = cost, wrt = w)
updates = [[w, w-gradient * 0.01 ]]

train  = theano.function(inputs=[X,Y], outputs=cost, updates = updates, allow_input_downcast=True)

def init():
    line.set_data([], [])
    return line,

def update_w(i):
    train(X_tr[i],y_tr[i])

    global w
    line.set_data(X_tr,X_tr*w.get_value())
    return line,

fig1 = plt.figure()
ax = plt.axes()
line, = ax.plot([], [], lw=2)

plt.scatter(X_tr,y_tr)
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['Fitted model','Input'])
plt.title('Plot of $y = 2x+0.33 \eta (0,1)$')

line_ani = animation.FuncAnimation(fig1, update_w,init_func=init, frames=100, interval=25, blit=True)
plt.show()
