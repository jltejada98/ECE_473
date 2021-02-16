
import numpy as np

P = [[0.6,0.3, 0.1],[0.2,0.8,0],[0.15,0.35,0.5]]

P = np.array(P)

#Section 2 Question 8
number_steps = 5

P_n = np.linalg.matrix_power(P,number_steps)

initial_distribution = [5,0.5,2]

print(np.dot(initial_distribution,P_n))

#We want to find the solution to the system pi*P = pi, where pi is the steady state matrix.

def steady_state_prob(p): #Function to calculate steady state probabilities of a transition probability matrix
    dim = p.shape[0]
    q = (p-np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q,ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    sol = np.linalg.solve(QTQ,bQT)
    if np.allclose(np.dot(QTQ, sol),bQT) : #Check solution
        return sol
    else:
        return False

print(steady_state_prob(P))



