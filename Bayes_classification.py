import numpy as np

def discriminant_fun(x, mean, cov, prior):
    exponent = -0.5*np.dot(np.dot((x-mean).T,np.linalg.inv(cov)),(x-mean))
    return exponent-0.5*np.log(np.linalg.det(cov))+np.log(prior)

#set seed
np.random.seed(1)

mean0 = np.array([3, 0])
cov0 = np.array([[0.10, 0],[0, 0.75]])
sample0 = 250

mean1 = np.array([6, 0])
cov1 = np.array([[0.10, 0],[0, 0.75]])
sample1 = 250

classA_data = np.random.multivariate_normal(mean0, cov0, sample0)
classB_data = np.random.multivariate_normal(mean1, cov1, sample1)

X = np.vstack((classA_data, classB_data))
y = np.hstack((np.zeros(sample0), np.ones(sample1)))

P_classA = len(classA_data)/(len(classA_data)+len(classB_data))
P_classB = len(classB_data)/(len(classA_data)+len(classB_data))

mean_classA = np.mean(classA_data, axis= 0)
mean_classB = np.mean(classB_data, axis= 0)

cov_classA = np.cov(classA_data.T)
cov_classB = np.cov(classB_data.T)

predic = []
for i in X:
    classA = discriminant_fun(i, mean_classA, cov_classA, P_classA)
    classB = discriminant_fun(i, mean_classB, cov_classB, P_classB)
    if classA > classB:
        predic.append(0)
    else:
        predic.append(1)
        
predic = np.array(predic)

true_positive = np.sum((predic==1)&(y==1))
true_negative = np.sum((predic==0)&(y==0))
false_positive = np.sum((predic==1)&(y==0))
false_negative = np.sum((predic==0)&(y==1))

print(f'true_positive = {true_positive}')
print(f'true_negative = {true_negative}')
print(f'false_positive = {false_positive}')
print(f'false_negative = {false_negative}')