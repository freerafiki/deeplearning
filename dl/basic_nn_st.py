import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import random

st.title("Basic exercise for machine learning")

st.markdown("""
we create some random data with the shape (300, 2) so we can easily plot in the x, y plane

we assign some of them the *label* of 1 (or blue, or whatever you want) and the others are 0 (or orange, or different to whatever you wanted before).

The simplest exercise to start with is labelling based on the x axis.
""")
np.random.seed(17)
random_data = np.random.rand(300,2)
labels = np.round(random_data[:,0]).astype(int)

fig = plt.figure()
plt.scatter(random_data[labels==True,0], random_data[labels==True,1])
plt.scatter(random_data[labels==False,0], random_data[labels==False,1])
st.pyplot(fig)

st.markdown("""
## Classification example
One if the simplest form of classification is taking a line and classifying the values
based on whether they are above or below the line.

Can you find the slope to achieve 100% accuracy?
""")
c1, c2, c3, c4 = st.columns(4)
with c1:
    m = st.number_input('slope', value=1.0, step=0.1)

with c2:
    q = st.number_input('offset', value=0.01)

with c3:
    st.markdown('equation')
    st.write(f"y = {m:.02f}x+{q:0.02f}")

with c4:
    classification = np.squeeze(random_data[:,1]>random_data[:,0]*m+q).astype(int)
    accuracy = np.sum(np.abs(classification - labels))/300
    st.write('Accuracy')
    st.write(f'{accuracy*100:.02f} %')

x = np.linspace(0,1,300)
y = m*x + q
y2 = y[y<=1]
x2 = x[y<=1]
y3 = y2[y2>=0]
x3 = x2[y2>=0]
fig = plt.figure()

plt.scatter(random_data[random_data[:,1]>random_data[:,0]*m+q,0], random_data[random_data[:,1]>random_data[:,0]*m+q,1])
plt.scatter(random_data[random_data[:,1]<random_data[:,0]*m+q,0], random_data[random_data[:,1]<random_data[:,0]*m+q,1])
plt.plot(x3,y3)
st.pyplot(fig)

st.markdown("""
## A harder example
If we assign random (or pseudo random) labels, the estimation is harder.
""")
np.random.seed(73)
random_data2 = np.random.rand(300,2)
labels2 = np.round(np.random.rand(300)).astype(bool)

fig = plt.figure()
plt.scatter(random_data2[labels==True,0], random_data2[labels==True,1])
plt.scatter(random_data2[labels==False,0], random_data2[labels==False,1])
st.pyplot(fig)

st.markdown("""
Can you find the slope to achieve 100% accuracy?
""")
c1, c2, c3, c4 = st.columns(4)
with c1:
    m2 = st.number_input('new slope', value=1.0, step=0.1)

with c2:
    q2 = st.number_input('new offset', value=0.01)

with c3:
    st.markdown('equation')
    st.write(f"y = {m2:.02f}x+{q2:0.02f}")

with c4:
    classification2 = np.squeeze(random_data2[:,1]>random_data2[:,0]*m2+q2).astype(int)
    accuracy2 = np.sum(np.abs(classification2 - labels2))/300
    st.write('Accuracy')
    st.write(f'{accuracy2*100:.02f} %')

x = np.linspace(0,1,300)
y = m2*x + q2
y2 = y[y<=1]
x2 = x[y<=1]
y3 = y2[y2>=0]
x3 = x2[y2>=0]
fig = plt.figure()

plt.scatter(random_data2[random_data2[:,1]>random_data2[:,0]*m2+q2,0], random_data2[random_data2[:,1]>random_data2[:,0]*m2+q2,1])
plt.scatter(random_data2[random_data2[:,1]<random_data2[:,0]*m2+q2,0], random_data2[random_data2[:,1]<random_data2[:,0]*m2+q2,1])
plt.plot(x3,y3)
st.pyplot(fig)

st.markdown(r"""
## How do you get higher accuracies?
We cannot get to 100% with only a line, we will need a curve. But that's behind the scope of this page.

The simplest way is trying. You put some value in the slope and offset, try and see the accuracy.
Then you change it, see if it got better or worse and change it again, accordingly.

This is the basic mechanism behind the neural network idea (at least the basic feed forward one).

### Neural network
A neural network consists of input, layers, and output.
Usually, you have $y = Wx +b$, where $W$ is a matrix and $b$ is a vector
($x$ is a vector as well, if you are wondering).
The training of the network consists in finding the optimal parameters $W$ and $b$.

In our simple example, we have instead $y = mx + q$, so we ahve only 2 parameters (2 scalars!).
We can consider it a a super small neural network with one neuron.
To train it, you have three fundamental steps:

#### Forward propagation
What part would be the forward propagation for our model?
""")
show_forward = st.checkbox('show forward solution')
if show_forward:
    st.write(r'computing $y$! given $m$ and $q$, calculating $y = mx + q$ is actually the forward step')

st.markdown("""
#### Loss function
What is the loss function for our model?
""")
show_loss = st.checkbox('show loss solution')
if show_loss:
    st.write(r'''
computing the accuracy! Comparing the prediction $y$ which we got from the forward step
is the calculation of the loss.
''')

st.markdown("""
#### Back Propagation
What part would be the backward propagation for our model?
""")
show_loss = st.checkbox('show back prop solution')
if show_loss:
    st.write(r'''
updating the parameter! The back propagation takes the loss calculated from the loss function
and propagates it back to our parameters, updating them for a new trial.
How exactly the parameters are updated it's a bit more complex and will be seen later.
''')

st.write(r'''
## Improving the parameters
How can we update the parameter in the right direction and avoid changing randomly and never getting to the optimal solution?
''')
candidates_m1 = np.linspace(50, 150, 100)
c_q1 = -50
accuracies_1 = np.zeros((candidates_m1).shape)

for k, candidate_m1 in enumerate(candidates_m1):
    classification_1 = np.squeeze(random_data[:,1]>random_data[:,0]*candidate_m1+c_q1).astype(int)
    accuracies_1[k] = np.sum(np.abs(classification_1 - labels))/300
fig, ax = plt.subplots()
plt.plot(candidates_m1,1-accuracies_1)
ax.set_ylabel(r'loss')
ax.set_xlabel(r'$m$ values for the first case (assuming q=50)')
st.pyplot(fig)
#st.write(candidates_m1, accuracies_1)

st.write(r'''
It is possible in the first case, if we plot the loss function we can clearly see
there is a point, we just need to follow the pendence of the curve.
This is done using derivatives, which can be seen as small lines tangent to the curve.
Using them, we can get to lower values of the loss function.

But this is guarantee to succeed only if the loss shape is convex, meaning it only has one global minimum.

In the second case, we assign the label randomly, so the loss will always oscillates around 0.5 (random, in fact),
and there is no guarantee to converge to the optimal solution.
''')

candidates_m2 = np.linspace(50, 150, 100)
c_q2 = -50
accuracies_2 = np.zeros((candidates_m2).shape)

for k, candidate_m2 in enumerate(candidates_m2):
    classification_2 = np.squeeze(random_data2[:,1]>random_data2[:,0]*candidate_m2+c_q2).astype(int)
    accuracies_2[k] = np.sum(np.abs(classification_2 - labels2))/300
fig, ax = plt.subplots()
plt.plot(candidates_m2,1-accuracies_2)
ax.set_ylabel(r'loss')
ax.set_xlabel(r'$m$ values for the second case (assuming q=50)')
st.pyplot(fig)
#st.write(candidates_m1, accuracies_1)
