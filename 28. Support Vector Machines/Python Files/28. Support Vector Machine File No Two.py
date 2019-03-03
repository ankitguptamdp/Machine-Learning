
# coding: utf-8

# # 01. SVM - Introduction

# ## Support Vector Machines
# - SVM is a powerful classifier that works both on linearly and non-linearly separable data
# - It is a classification technique which uses kernel trick
# <img src="../Pictures/linearly_separable.png" alt="Linear Separable" style="width: 500px;"/>
# 
# 
# 
# - Finds an optimal hyperplane, that best separates our data so that the distance from nearest points in space to itself(also called margin) is maximized
# - These nearest points are called **Support Vectors**
# 
# <figure>
# <img src="../Pictures/svm_margin.png" alt="Pizza-1" style="width: 300px;"/>
# 
# <figcaption>
# Image By - <a href="//commons.wikimedia.org/w/index.php?title=User:Larhmam&amp;action=edit&amp;redlink=1" class="new" title="User:Larhmam (page does not exist)">Larhmam</a> - <span class="int-own-work" lang="en">Own work</span>, <a href="https://creativecommons.org/licenses/by-sa/4.0" title="Creative Commons Attribution-Share Alike 4.0">CC BY-SA 4.0</a>, <a href="https://commons.wikimedia.org/w/index.php?curid=73710028">Link</a>
# </figcaption>
# 
# - For the non-linearly separable case, it uses something called 'Kernel Trick' which will go through in the next part.

# ## What does hyperplane mean ?
# 
# <img src="../Pictures/hyperplanes.jpg" alt="Hyperplanes" style="width: 250px;"/>
# 
# A hyperplane is plane of _n-1_ dimensions in _n_ dimensional feature space, that separates the two classes. 
# For a 2-D feature space, it would be a line and for a 3-D Feature space it would be plane and so on.
# 
# <img src="../Pictures/3d_hyperplane.png" alt="Hyperplanes" style="width: 200px;"/>
# 
# 
# 
# A hyperplane is able to separate classes if for all points -
# 
# #### _wx=_w1x1 + _w2x2
# #### Distance of a point from the plane is :( _w1x1 + _w2x2 + c)/(sqrt(_w1^2+ _w2^2))
# #### Or ( _wx+ c)/(sqrt(_w^2))
# 
# #### **_w_ x** + b > 0 
# (For data points in class 1)  
# #### **_w_ x** + b < 0 
# (For data points in  class 0)

# ## Maximum Margin Hyperplane 
# 
# An optimal hyperplane best separates our data so that the distance/margin from nearest points(called Support Vectors) in space to itself is maximized.
# 
# L1 is not a hyperplane
# 
# L2 is not a good hyperplane because nearest point to this plane is quite close
# 
# L3 is a good hyperplane
# 
# <img src="../Pictures/maximum_margin.png" alt="Hyperplanes" style="width: 400px;"/>
# 

# ### SVM Implementation using Pegasos
# 
# **Formulating SVM as Unconstrainted Optimization Problem**
# 
# Paper - [Pegasos: Primal Estimated sub-GrAdient SOlver for SVM](http://www.ee.oulu.fi/research/imag/courses/Vedaldi/ShalevSiSr07.pdf)
# 
# The final SVM Objective we derived was -
# Hinge Loss
# 
# <img src="../Pictures/loss.png" alt="Hinge Loss" style="width: 400px;"/>
# 
# 

# # 07. SVM Implementation 1 - Hinge Loss Function

# ### Generate Dataset

# In[1]:


from sklearn.datasets import make_classification


# In[4]:


X,Y=make_classification(n_classes=2,n_samples=400,n_features=2,n_informative=2,n_redundant=0)


# In[5]:


import matplotlib.pyplot as plt


# In[8]:


plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[10]:


X,Y=make_classification(n_classes=2,n_samples=400,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,random_state=5)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[12]:


X,Y=make_classification(n_classes=2,n_samples=400,n_features=2,n_informative=2,n_redundant=0,n_clusters_per_class=1,random_state=3)
plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[58]:


# Convert Y-Labels into {-1,1} otherwise we will lose information when value of Y will be 0
print(Y)
print(Y==0)
Y[Y==0]=-1
print(Y)


# In[59]:


print(X.shape)


# In[60]:


class SVM:
    
    def __init__(self,C=1.0):        
        self.C=C # Penalty
        self.W=0 # Weight
        self.b=0 # Bias Term
        
    def hingeLoss(self,W,b,X,Y):
        
        import numpy as np
        
        loss=0.0
        loss+=0.5*np.dot(W,W.T) # W.T => W Transpose
        
        m=X.shape[0]
        
        for i in range(m):
            ti=Y[i]*(np.dot(W,X[i].T)+b) # We are trying to make W 1x2 and X[i].T is 2x1 so finally it will be 1x1
            loss+=self.C*max(0,(1-ti))
            
        return loss[0][0]
    
    def fit(self,X,Y,batch_size=100,learning_rate=0.001):
        # batch_size is the Batch Size in Batch Gradient Descent Algorithm
        # Batch gradient descent is a variation of the gradient descent algorithm that calculates the error for each example in the training dataset, but only updates the model after all training examples have been evaluated. 
        # One cycle through the entire training dataset is called a training epoch.
        
        import numpy as np
        
        no_of_features=X.shape[1]
        no_of_samples=X.shape[0]
        
        n=learning_rate
        c=self.C
        
        # Initialise the model parameter W and b
        W=np.zeros((1,no_of_features))
        bias=0
        print(self.hingeLoss(W,bias,X,Y))
        
        # Training from here
        # Weight and Bias update rule


# In[61]:


mySVM=SVM()


# In[62]:


mySVM.fit(X,Y)


# # 08. SVM Implementation 2 - Training Using Mini-Batch Gradient Descent

# In[63]:


# Convert Y-Labels into {-1,1} otherwise we will lose information when value of Y will be 0
print(Y)
print(Y==0)
Y[Y==0]=-1
print(Y)


# In[64]:


plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[106]:


class SVM:
    
    def __init__(self,C=1.0):        
        self.C=C # Penalty
        self.W=0 # Weight
        self.b=0 # Bias Term
        
    def hingeLoss(self,W,b,X,Y):
        
        import numpy as np
        
        loss=0.0
        loss+=0.5*np.dot(W,W.T) # W.T => W Transpose
        
        m=X.shape[0]
        
        for i in range(m):
            ti=Y[i]*(np.dot(W,X[i].T)+b) # We are trying to make W 1x2 and X[i].T is 2x1 so finally it will be 1x1
            loss+=self.C*max(0,(1-ti))
            
        return loss[0][0]
    
    def fit(self,X,Y,batch_size=100,learning_rate=0.001,maxItr=300):
        # batch_size is the Batch Size in Batch Gradient Descent Algorithm
        # Batch gradient descent is a variation of the gradient descent algorithm that calculates the error for each example in the training dataset, but only updates the model after all training examples have been evaluated. 
        # One cycle through the entire training dataset is called a training epoch.
        
        import numpy as np
        
        no_of_features=X.shape[1]
        no_of_samples=X.shape[0]
        
        n=learning_rate
        c=self.C
        
        # Initialise the model parameter W and b
        W=np.zeros((1,no_of_features))
        bias=0
        #print(self.hingeLoss(W,bias,X,Y))
        
        # Training from here
        # Weight and Bias update rule
        losses=[]
        
        for i in range(maxItr):
            # Training Loop
            
            l=self.hingeLoss(W,bias,X,Y)
            losses.append(l)
            ids=np.arange(no_of_samples)
            np.random.shuffle(ids)
                        
            # Batch Gradient Descent(Paper) with random shuffling
            for batch_start in range(0,no_of_samples,batch_size):
                # Assume 0 gradient for the batch
                gradw=0
                gradb=0
                
                # Iterate over all examples in the mini batch
                for j in range(batch_start,batch_start+batch_size):
                    if j<no_of_samples:
                        i=ids[j]                    
                        ti=Y[i]*(np.dot(W,X[i].T)+bias) # T means Transpose
                        
                        if ti>1:
                            gradw+=0
                            gradb+=0
                        else:
                            gradw+=c*Y[i]*X[i]
                            gradb+=c*Y[i]
                
                # Gradient for the batch is ready! Update W,B
                W = W - n*W + n*gradw
                bias = bias + n*gradb
            
        self.W=W
        self.b=bias
        return W,bias,losses                    


# In[107]:


import numpy as np

ids=np.arange(100)
print(ids)
np.random.shuffle(ids)
print(ids)


# In[108]:


mySVM=SVM()
W,b,losses=mySVM.fit(X,Y)


# In[109]:


print(losses)


# In[110]:


plt.plot(losses)
plt.show()


# In[112]:


mySVM=SVM()
W,b,losses=mySVM.fit(X,Y,maxItr=50)
print(losses[0])
print(losses[-1])


# In[81]:


plt.plot(losses)
plt.show()


# In[113]:


mySVM=SVM()
W,b,losses=mySVM.fit(X,Y,maxItr=100)
print(losses[0])
print(losses[-1])
plt.plot(losses)
plt.show()


# In[85]:


W,B=mySVM.W,mySVM.b
print(W)
print(B)


# In[88]:


def plotHyperplane(w1,w2,b):
    x_1=np.linspace(-2,4,10)
    x_2= -( w1 * x_1 + b ) / w2
    
    plt.plot(x_1,x_2)
    plt.scatter(X[:,0],X[:,1],c=Y)
    plt.show()


# In[89]:


plotHyperplane(W[0,0],W[0,1],B)


# In[90]:


def plotHyperplane(w1,w2,b):
    plt.figure(figsize=(12,12))
    x_1=np.linspace(-2,4,10)
    x_2= -( w1 * x_1 + b ) / w2
    
    plt.plot(x_1,x_2)
    plt.scatter(X[:,0],X[:,1],c=Y)
    plt.show()


# In[91]:


plotHyperplane(W[0,0],W[0,1],B)


# # 09. SVM - Visualizing Hyperplanes, Effect Of Penalty Constant

# ### Visualising Support Vectors, Positive And Negative Hyperplanes

# In[93]:


# Effect Of Changing 'C' - Penalty Constant


# In[96]:


def plotHyperplane(w1,w2,b):
    plt.figure(figsize=(12,12))
    x_1=np.linspace(-2,4,10)
    x_2= -( w1*x_1 + b ) / w2 
    # w*x + b =0
    # w1*x1 + w2*x2 + b = 0
    
    x_p= -( w1*x_1 + b + 1) / w2 
    # w*x + b = -1
    # w1*x1 + w2*xp + b = -1
    
    x_n= -( w1*x_1 + b - 1) / w2 
    # w*x + b = 1
    # w1*x1 + w2*xn + b = 1
    
    plt.plot(x_1,x_2)
    plt.plot(x_1,x_p)
    plt.plot(x_1,x_n)
    plt.scatter(X[:,0],X[:,1],c=Y)
    plt.show()


# In[97]:


plotHyperplane(W[0,0],W[0,1],B)


# In[100]:


def plotHyperplane(w1,w2,b):
    plt.figure(figsize=(12,12))
    x_1=np.linspace(-2,4,10)
    x_2= -( w1*x_1 + b ) / w2 
    # w*x + b =0
    # w1*x1 + w2*x2 + b = 0
    
    x_p= -( w1*x_1 + b + 1) / w2 
    # w*x + b = -1
    # w1*x1 + w2*xp + b = -1
    
    x_n= -( w1*x_1 + b - 1) / w2 
    # w*x + b = 1
    # w1*x1 + w2*xn + b = 1
    
    plt.plot(x_1,x_2,label='Hyperplane WX+B=0')
    plt.plot(x_1,x_p,label='Positive Hyperplane WX+B=-1')
    plt.plot(x_1,x_n,label='Negative Hyperplane WX+B=1')
    plt.legend()
    
    plt.scatter(X[:,0],X[:,1],c=Y)
    plt.show()


# In[101]:


plotHyperplane(W[0,0],W[0,1],B)


# In[102]:


# Points above orange line contributes 0 error
# Points below green line contributes 0 error excepts theat purple point
# For the region in between them they will have an error of (1-Ei)


# In[119]:


# Changing C (Penalty)
mySVM=SVM(C=1000)
W,b,losses=mySVM.fit(X,Y,maxItr=100)
print(losses[0])
print(losses[-1])
plt.plot(losses)
plt.show()
# Final loss is very high


# In[120]:


# Weights and Bias is high
W,B=mySVM.W,mySVM.b
print(W)
print(B)


# In[123]:


plotHyperplane(W[0,0],W[0,1],B)
# The margin has been reduced here because penalty for adding any point near to the hyperplane was very high here.


# In[124]:


# For a optimal value of C we will study grid search later on


# In[125]:


# A Support Vector Machine (SVM) performs classification by finding the hyperplane that maximizes the margin between the two classes. The vectors (cases) that define the hyperplane are the support vectors.


# # 10. Handling Non-Linearly Separable Data

# ### Non-Linear Classification
# 
# 
# - In many real life problems, the data is not linearly separable,but we need to classify the data. This can be done using by projecting the data to higer dimesions so that it becomes linearly separable.
# <img src="../Pictures/linearly_separable.png" alt="Linear Separable" style="width: 600px;"/>

# ### Projecting data to higher dimensions!
# When working with non-linear datasets, we can project orginal feature vectors into higher dimensional space where they can be linearly separated!  
# 
# ### Let us see one example
# 
# 
# Data in 2-Dimensional Space
# <img src="../Pictures/circles_low.png" alt="Linear Separable" style="width: 400px;"/>
# 
# Data Projected in 3-D Dimensional Space, after processing the original data using a non-linear function.
# <img src="../Pictures/circles_3d.png" alt="Linear Separable" style="width: 400px;"/>

# ### Code

# In[130]:


from sklearn.datasets import make_circles
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# In[131]:


X,Y=make_circles(n_samples=500,noise=0.05)


# In[132]:


print(X.shape,Y.shape)


# In[133]:


plt.scatter(X[:,0],X[:,1],c=Y)
plt.show()


# In[134]:


def phi(X):
    """Non Linear Transformation"""
    X1=X[:,0]
    X2=X[:,1]
    X3=X1**2 + X2**2
    
    X_=np.zeros((X.shape[0],3))
    print(X_.shape)
    
    X_[:,:-1]=X
    X_[:,-1]=X3
    
    return X_


# In[135]:


X_=phi(X)


# In[136]:


print(X[:3,:])


# In[137]:


print(X_[:3,:])


# In[152]:


def plot3d(X,Y):
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection='3d')
    
    """https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html"""
    
    """Either a 3-digit integer or three separate integers describing the position of the subplot.
    If the three integers are nrows, ncols, and index in order,
    the subplot will take the index position on a grid with nrows rows and ncols columns.
    index starts at 1 in the upper left corner and increases to the right.
    pos is a three digit integer, where the first digit is the number of rows,
    the second the number of columns, and the third the index of the subplot.
    i.e. fig.add_subplot(235) is the same as fig.add_subplot(2, 3, 5).
    Note that all integers must be less than 10 for this form to work."""
    
    X1=X[:,0]
    X2=X[:,1]
    X3=X[:,2]
    
    ax.scatter(X1,X2,X3,zdir='z',s=20,c=Y,depthshade=True)
    
    """https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html"""
    # zdir	Which direction to use as z (‘x’, ‘y’ or ‘z’) when plotting a 2D set.
    # s	Size in points^2. It is a scalar or an array of the same length as x and y.
    # depthshade	Whether or not to shade the scatter markers to give the appearance of depth. Default is True.
    
    plt.show()
    return ax


# In[153]:


plot3d(X_,Y)


# In[154]:


# Reducing noise
X,Y=make_circles(n_samples=500,noise=0.02)
X_=phi(X)
plot3d(X_,Y)


# In[155]:


ax=plot3d(X_,Y)


# ### Logistic Classifier

# In[168]:


from sklearn.linear_model import LogisticRegression


# In[171]:


lr=LogisticRegression(solver='lbfgs')
# Adding solver to avoid warning

#/usr/local/lib/python3.6/dist-packages/sklearn/linear_model/logistic.py:433:
#FutureWarning: Default solver will be changed to 'lbfgs' in 0.22.
#Specify a solver to silence this warning. FutureWarning)"""


# In[172]:


from sklearn.model_selection import cross_val_score


# In[173]:


#cross_val_score?


# In[174]:


accuracy=cross_val_score(lr,X,Y,cv=5).mean()
"""cv : int, cross-validation generator or an iterable, optional
    Determines the cross-validation splitting strategy."""
print("Accuracy X(2D) is %.4f"%(accuracy*100))


# In[175]:


# It is a pretty bad accuracy


# ### Logistic Classifier On Higher Dimension Space

# In[178]:


accuracy=cross_val_score(lr,X_,Y,cv=5).mean()
print("Accuracy X(3D) is %.4f"%(accuracy*100))


# In[177]:


# It is best possible accuracy


# ### Visualise The Decision Surface

# In[179]:


lr.fit(X_,Y)


# In[188]:


wts=lr.coef_
print(wts)
# These are the weights


# In[189]:


bias=lr.intercept_
print(bias)
# This is the bias


# In[187]:


xx,yy=np.meshgrid(range(-2,2),range(-2,2))
print(xx)
print()
print(yy)


# In[191]:


z=-(wts[0,0]*xx+wts[0,1]*yy+bias)/wts[0,2]
print(z)


# In[202]:


def plot3d(X,Y,show=True):
    fig=plt.figure(figsize=(10,10))
    ax=fig.add_subplot(111,projection='3d')
    
    """https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html"""
    
    """Either a 3-digit integer or three separate integers describing the position of the subplot.
    If the three integers are nrows, ncols, and index in order,
    the subplot will take the index position on a grid with nrows rows and ncols columns.
    index starts at 1 in the upper left corner and increases to the right.
    pos is a three digit integer, where the first digit is the number of rows,
    the second the number of columns, and the third the index of the subplot.
    i.e. fig.add_subplot(235) is the same as fig.add_subplot(2, 3, 5).
    Note that all integers must be less than 10 for this form to work."""
    
    X1=X[:,0]
    X2=X[:,1]
    X3=X[:,2]
    
    ax.scatter(X1,X2,X3,zdir='z',s=20,c=Y,depthshade=True)
    
    """https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html"""
    # zdir	Which direction to use as z (‘x’, ‘y’ or ‘z’) when plotting a 2D set.
    # s	Size in points^2. It is a scalar or an array of the same length as x and y.
    # depthshade	Whether or not to shade the scatter markers to give the appearance of depth. Default is True.
    if show==True:
        plt.show()
    return ax


# In[203]:


ax=plot3d(X_,Y,show=False)
ax.plot_surface(xx,yy,z,cmap='coolwarm')
plt.show()


# In[204]:


ax=plot3d(X_,Y,show=False)
ax.plot_surface(xx,yy,z,alpha=0.4)
# alpha for opacity adjustment
plt.show()


# In[197]:


#ax.plot_surface?

