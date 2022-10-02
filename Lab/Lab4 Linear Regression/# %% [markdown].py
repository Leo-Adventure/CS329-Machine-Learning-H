# %% [markdown]
# # <center>LAB4 tutorial for Machine Learning <br > Linear Regression</center>
# 
# > The document description are designed by JIa Yanhong in 2022. Sept. 22th
# 
# ## Objective
# 
# - Master the  linear regression algorithm.
# - Understanding Gradient Descent
# - Polynomial Regression
# - Learn how to evaluate regression models
# - Complete the LAB assignment and submit it to BB.
# 

# %% [markdown]
# ## 1 Linear Regression
# ### 1.1 Conceptual Overview
# 
# Linear Regression is **“supervised” “regression”** algorithm.
# 
# <center><img src="images/classical-ml.png" alt=" " style="zoom:80%;" /></center>
# 
# Supervised meaning we use labeled data to train the model.
# 
# <center><img src="images/supervised-vs-unsupervised.png" alt="supervised-vs-unsupervised" style="zoom:150%;" /> </center>
# 
# Regression meaning we predict a numerical value, instead of a “class”.
# <center><img src="images/p1.png" alt="p1 " style="zoom:150%;" /> </center>
# 
# 
# 
# 

# %% [markdown]
# ### 1.2 Linear Regression
# 
# In statistics, linear regression is a linear approach to modelling the relationship between a dependent variable and one or more independent variables. 
# 
# We will define a linear relationship between these two variables as follows:
# $$ y=\theta _{0}+\theta _{1}x_{1}$$ 
# 
# 
# This is the equation for a line that you studied in high school. $\theta _{1}$ is the slope of the line and $\theta _{0}$ is the y intercept.
# 
# 
# <center><img src="images/p12.png" alt="p12 " style="zoom:100%;" /> </center>
# When there are multiple independent variables, the linear relationship becomes as follows:
# 
# $$y=\theta _{0}+\theta _{1}x_{1}+\theta _{2}x_{2}+...+\theta _{n}x_{n}$$
# 
# 
# <center><img src="images/p6_.png" alt="p6 " style="zoom:70%;" /> </center>
# 
# Let $\vec{x}$ be the independent variable and **y** be the dependent variable. The general form of the model's prediction:
# 
# $$\hat{y}=h_{\theta }(x)=\theta _{0}+\theta _{1}x_{1}+\theta _{2}x_{2}+...+\theta _{n}x_{n}=\theta ^{T}\cdot x$$
# $\theta$ is the model’s parameter vector, containing the bias term $\theta _{0}$ and the feature weights $\theta _{1}$ to $\theta _{n}$. (n = number of features).
# 
# $\theta ^{T}\cdot x$ is the dot product of the vectors $\theta$ and $x$, which is, of course, equal to $\theta _{0}+\theta _{1}x_{1}+\theta _{2}x_{2}+...+\theta _{n}x_{n}$.
# 
# There is an error between the real y and the predicted $\hat{y}$:
# 
# $$y=\hat{y}+\varepsilon =h_{\theta }(x)+\varepsilon =\theta ^{T}\cdot x+\varepsilon $$
# 
# Our challenge today is to determine the value of $\theta$. 
# 
# For a single variable, such that the line corresponding to those values is the best fitting line .
# 
# For multiple variables, such as 2 variables, the surface corresponding to these values is the surface of best fit.
# 
# <center><img src="images/p8.png" alt="p8 " style="zoom:120%;" /></center>

# %% [markdown]
# ### 1.3 Cost Function
# 
# 
# Cost Function evaluates the model’s predictions and tells us how accurate are the model’s predictions. The lower the value of the cost function, better accurate the predictions of the model. They are many cost functions to choose, but we will use the Mean Squared Error (MSE) cost function.
# 
# The MSE function calculates the average of the squared difference between the prediction and the actual value (y).
# 
# $$J(\theta )=MSE=\frac{1}{2m}\sum_{j=1}^{m}(h_{\theta }(x^{j})-y^{j})^{2} = \frac{1}{2m}\sum_{j=1}^{m}(\theta _{0}+\theta _{1}x_{1}^{j}+\theta _{2}x_{2}^{j}+...+\theta _{n}x_{n}^{j}-y^{j})^{2}$$
# > $x_{i}^{j}$ :The $i_{th}$ feature of the $j^{th}$ sample
# > 
# > $y_{i}^{j}$ : the $i_{th}$ label of the $j^{th}$ sample
# > 
# > $(x^{j},y^{j})$: the $j^{th}$ sample
# > 
# > $m$ : total number of instances in your dataset
# 
# 

# %% [markdown]
# ### 1.4 Least-squares estimation
# 
# Now that we have determined the cost function, the only thing left to do is minimize it. 
# This is done by finding the partial derivative of $J(\Theta )$, equating it to 0 and then finding an expression for $\Theta$ .
# 
# The loss function can be written as:
# 
# \begin{aligned} 
# J(\theta )&=\frac{1}{2m}\sum_{j=1}^{m}(h_{\theta }(x^{j})-y^{j})^{2}\\
# &=\frac{1}{2m}\sum_{j=1}^{m}(\theta ^{T}\cdot x^{j}-y^{j})^{2} \\
# &= \frac{1}{2m}(X\theta - y)^{T}(X\theta -y )
# \end{aligned}
# 
# 
# As the loss is convex the optimum solution lies at gradient zero. The gradient of the loss function is ：
# \begin{aligned} 
# \frac{\partial  J(\theta )}{\partial  \theta } 
# &= \frac{\partial  \frac{1}{2m}(X\theta - y)^{T}(X\theta -y )}{\partial  \theta } \\
# &= \frac{\partial  \frac{1}{2m}(\theta^{T}X^{T} - y^{T})(X\theta -y )}{\partial  \theta } \\
# &= \frac{\partial  \frac{1}{2m}(\theta^{T}X^{T}X\theta-\theta^{T}X^{T}y-y^{T}X\theta+y^{T}y)}{\partial  \theta } \\
# &= \frac{1}{2m}(2X^{T}X\theta-X^{T}y-(y^{T}X)^{T}) \\
# &= \frac{1}{m}X^{T}X\theta - X^{T}y 
# \end{aligned}
# 
# 
# When X is a matrix of full rank, setting the gradient to zero produces the optimum parameter:
# \begin{aligned} 
# \theta ^{*}=(X^{T}X)^{-1}X^{T}y
# \end{aligned}
# 
# 
# \begin{aligned} 
# X=
# \begin{Bmatrix}
#  &1  &x_{1}^{1}  &x_{2}^{1}  &... &x_{n}^{1} \\ 
#  &1  &x_{1}^{2}  &x_{2}^{2}  &... &x_{n}^{2} \\ 
#  &...  &  &  & &...\\ 
#  &1  &x_{1}^{m}  &x_{2}^{m}  &... &x_{n}^{m}
# \end{Bmatrix} ,\ y = \begin{Bmatrix} y_{0}\\ y_{1}\\ y_{2}\\ ...\\ y_{m} \end{Bmatrix},\theta = \begin{Bmatrix} \theta _{0}\\ \theta _{1}\\ \theta _{2}\\ ...\\ \theta _{n} \end{Bmatrix}
# 
# \end{aligned}

# %% [markdown]
# #### Hands-on Coding

# %%
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
np.random.seed(42) # 随机生成
X = np.random.randn(1000, 1)
# y = 5 + 3 * X + Gaussian noise -> because in real-world it is very unlikely to get data that has a perfect linear relationship
y = 5 + 3 * X + np.random.randn(1000, 1)
plt.scatter(X, y)
plt.show()


# %%
X_b = np.c_[np.ones((1000, 1)), X]  # Adding the bias term which is equal to 1

# Dividing the data into train and test sets    
X_train, X_test, y_train, y_test = train_test_split(X_b, y, test_size=0.2, random_state=42)

theta_optimize = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train) # 最优的Theta/weight

#Output
#obtained feature weights  w0 = 5.04, w1 = 2.94
print(theta_optimize)

# Predicting new data with the obtained feature weights
y_pred = X_test.dot(theta_optimize)
r2_score(y_test, y_pred)


# %%
plt.scatter(X, y) 
plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')  # regression line
plt.show()

# %% [markdown]
# You have seen it has predicted the feature weights very close to the actual values ($y = 5 + 3*X + Gaussian noise$), but due to the noise in the data it is unable to predict the exact values, but the predictions were close enough.
# 
# #### Disadvantages
# 
# - It is computationally expensive if you have a large number of features.
# - If there are any redundant features in your data, they the matrix inversion in the normal equation is not possible. In that case, the inverse can be replaced by the pseudo inverse.

# %% [markdown]
# ### 1.5 Gradient Descent
# 
# In this section you can learn how the gradient descent algorithm works and implement it from scratch in python.
# 
# Gradient Descent minimizes the cost function by iteratively moving in the direction of steepest descent, updating the parameters along the way.
# 
# <center><img src="images/gradient-descent-algorithm.jpg" alt="gradient-descent-algorithm " style="zoom:80%;" /></center>
# 
# In a real world example, it is similar to find out a best direction to take a step downhill.
# 
# <center><img src="images/downhill.png" alt="img " style="zoom:100%;" /></center>
# 
# 
# 
# We take a step towards the direction to get down. From the each step, you look out the direction again to get down faster and downhill quickly. 
# 
# 
# 
# 
# To find the best minimum, repeat steps to apply various values for $\theta$. In other words, repeat steps until convergence.
# 
# <center><img src="images/p13.png" alt="img " style="zoom:110%;" /></center>
# 
# 
# The choice of correct learning rate is very important as it ensures that Gradient Descent converges in a reasonable time. 
#  
# 
# - If we choose α to be very small, Gradient Descent will take small steps to reach local minima and will take a longer time to reach minima.  
# 
# - If we choose **α to be very large**, Gradient Descent can overshoot the minimum. It may fail to converge or even diverge. 
# 
# <center><img src="images/learning-rate.png" alt="img " style="zoom:50%;" /></center>

# %% [markdown]
# #### Hands-on Coding
# #####  implement Linear Regression from scratch
# 

# %%
import numpy as np
from sklearn.metrics import r2_score

np.random.seed(42) # 确保每次运行结果一样

learning_rate = 0.1
iterations = 50
m = 100 # total number of samples
theta = np.random.randn(2,1) # random initialization
for iteration in range(iterations):
    gradient = 2/m * X_train.T.dot(X_train.dot(theta) - y_train)
    theta = theta - learning_rate * gradient

# Output
print(theta)


# array([[5.08703256],
#        [2.93815133]])

# Predicting new values with gradient descent
y_pred = X_test.dot(theta)
r2_score(y_test, y_pred)

# %%
plt.scatter(X, y) 
plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')  # regression line

plt.show()

# %% [markdown]
# ##### Implementing Linear Regression in Scikit-Learn

# %%
from sklearn.linear_model import LinearRegression # 直接导入 LinearRegression 库进行使用

X_train = X_train[:, 1] # The sklearn model will automatically add the bias term so we donot have to add it
X_test = X_test[:, 1]


linear_regression = LinearRegression()
linear_regression.fit(X_train.reshape(-1, 1), y_train)
print(linear_regression.intercept_)
# Output
#array([5.08703256])

print(linear_regression.coef_)
# Ouput
#array([[2.93815133]])

# Predicting new values and calculating the r2_score
linear_regression.score(X_test.reshape(-1, 1), y_test)

# output
#0.8961012486926588

# %% [markdown]
# As you can see the values we got from the normal equation, gradient descent, sklearn are nearly the same.
# 
# #### Disadvantages
# 
# - There are possibilities for the gradient descent to stuck in a local minimum if you use another cost function that is not of a convex shape.
# - You should find the appropriate value for the learning rate.
# 
# 
# Well if you have read this far and everything makes sense pat yourself on the back!. You have learned all the underlying concepts of linear regression.
# A

# %% [markdown]
# ### 1.6 [Polynomial Regression](https://en.wikipedia.org/wiki/Polynomial_regression)
# 
# We use polynomial regression when the relationship between  the independent and dependent variables is nonlinear.
# 
# 
# <center><img src="images/p2.png" alt="p2 " style="zoom:120%;" /></center>
# 
# This is accomplished by "exponentiating" our variable by taking it to powers greater than 1.
# 
# <center><img src="images/p3.png" alt="p3 " style="zoom:60%;" /></center>

# %% [markdown]
# #### Hands-on Coding

# %%
#step 1: Import the required libraries
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import train_test_split

# %%
#step2: Load the data set. Here, we're generating data 
sample_cnt= 100
#X = np.linspace(start = -8, stop = 8, num = sample_cnt)
X = np.random.uniform(-3,3,size=100)

# curve using polynomial
θ0, θ1, θ2, θ3 = 0.2, 1, 0.5, -0.4
y = θ0 + θ1*X + θ2*(X**2) + θ3*(X**3) + np.random.normal(0,1,size = sample_cnt) # 加入高斯噪声

plt.scatter(X,y)
plt.xlabel('X')
plt.ylabel('y')
plt.savefig('regu-0.png', dpi=200)
plt.show()


# %%
#split data 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,random_state=1)

# step 3: using a linear regression model to predict
lin_reg = LinearRegression()
lin_reg.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

y_pred = lin_reg.predict(X_train.reshape(-1,1))
print("lin_reg.coef = ", lin_reg.coef_)
print("lin_reg.intercept = ", lin_reg.intercept_)
a = lin_reg.coef_[0][0]
b = lin_reg.intercept_[0]

plt.plot(X, y, 'b.')
# plt.plot([min(X_train), max(X_train)], [min(y_pred), max(y_pred)], color='red')  # regression line
plt.plot(X_train, y_pred, 'r-')
plt.xlabel('X')
plt.ylabel('y')
plt.savefig('regu-1.png', dpi=200)
print("The equation: y = {}x+{}".format(a,b))
#Calculate the error and evaluate the model
print(lin_reg.intercept_, lin_reg.coef_) 
print(r2_score(y_pred, y_train)) 
plt.show()


# %%
# Use a polynomial with degree of 2
ploy  = PolynomialFeatures(degree=2, include_bias=False)
X_train2 = ploy.fit_transform(X_train.reshape(-1,1))

lin_reg = LinearRegression()
lin_reg.fit(X_train2, y_train.reshape(-1,1))
print(lin_reg.intercept_, lin_reg.coef_)  # [ 2.60996757] [[-0.12759678  0.9144504 ]]
a = lin_reg.coef_[0][0]
b = lin_reg.coef_[0][1]
c = lin_reg.intercept_[0]



y_pred = np.dot(X_train2, lin_reg.coef_.T) + lin_reg.intercept_

plt.plot(X_train, y_train, 'b.')
plt.plot(np.sort(X_train),y_pred[np.argsort(X_train)],color='r') # 注意此处的绘图方式

plt.savefig('regu-2.png', dpi=200)
plt.show()
print("The equation: y = {:.3f}x^2+{:.3f}x+{:.3f}".format(a,b,c))
print(r2_score(y_pred, y_train)) 
print(r2_score(np.dot(ploy.transform(X_test.reshape(-1,1)), lin_reg.coef_.T) + lin_reg.intercept_, y_test)) 

# %% [markdown]
# As the data extending to polynomial features, the value would be extremely large or small because of the power operation. That will influence the use of gradient descent which runs in background when we call fit(). So a normalization or standardization is necessary. See `StandardScaler` in preprocessing.
# 
# `Pipeline` can help us assemble several preprocessing functions and the learning process together.

# %%
# Use a polynomial with degree of 3
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

poly_reg = Pipeline([
    ('poly', PolynomialFeatures(degree=3)),
    ('std_scaler', StandardScaler()), # 标准化
    ('lin_reg', LinearRegression()) # 再进行线性回归
])

poly_reg.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))


y_pred = poly_reg.predict(X_train.reshape(-1,1))


plt.plot(X_train, y_train, 'b.')
plt.plot(X_test, y_test, 'g.')
plt.plot(np.sort(X_train),y_pred[np.argsort(X_train)],color='r')

plt.savefig('regu-3.png', dpi=200)
plt.show()
params =poly_reg.named_steps['lin_reg']
print("The equation:",poly_reg.named_steps['lin_reg'].coef_,poly_reg.named_steps['lin_reg'].intercept_)
print("The equation: y = {:.3f}x^3+{:.3f}x^2+{:.3f}x+{:.3f}".format(params.coef_[0][0],params.coef_[0][1],params.coef_[0][2],params.coef_[0][3],params.intercept_[0]))
print(r2_score(y_pred, y_train)) 
print(r2_score(poly_reg.predict(X_test.reshape(-1,1)), y_test)) 
#print(mean_squared_error(y_pred, y)) 

# %% [markdown]
# In practice,  we  need to select the "degree" of the polynomial.
# 
# <center><img src="images/degree-of-polynomial.png" alt="degree-of-polynomial" style="zoom:120%;" /></center>
# 
# Our selection should seek to fit the current data well, and generalize to new data.
# 
# <center><img src="images/fit.png" alt="fit" style="zoom:120%;" /></center>
# 
# This challenge is known as the "bias variance tradeoff".
# 
# <center><img src="images/bias-variance-tradeoff.png" alt="bias-variance-tradeoff " style="zoom:100%;" /></center>
# 
# <font color = "red">Bias usually caused by underfitting, Variance caused by overfitting. </font> What can we do to solve this problem? 
# - 1. Choose a better polynomial degree.
# As we increase the degree, the bias decreases, but the variance increases. We want to stop where these two factors are minimized.
# <center><img src="images/optimal-capacity.png" alt="optimal-capacity" style="zoom:120%;" /></center>
# 
# > Note:  polynomial regression is very sensitive to outliers, and we must take care when selecting the degree to avoid overfitting.
# 
# - 2. Regularization .Regularization in scikit learn is [`RidgeRegression`](https://en.wikipedia.org/wiki/Ridge_regression#:~:text=Ridge%20regression%20is%20a%20method%20of%20estimating%20the,in%20many%20fields%20including%20econometrics%2C%20chemistry%2C%20and%20engineering.) ,which is  in [linear_model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html). Use it if you need regularization in your model.
#   
#     Ridge regression is also known as L2 regularization. It is a regularized version of linear regression to find a better fitting line. It adds l2 penalty terms in the cost function and thereby reducing coefficients lower towards zero and minimizing their impact on the training data. It is useful to avoid over-fitting of the data in a model. The alpha parameter of a model is used to calculate cost function to manage bias-variance tradeoff and reduce an error in prediction.
# 
# \begin{aligned} 
# J(\theta )&=\frac{1}{2m}\sum_{j=1}^{m}(h_{\theta }(x^{j})-y^{j})^{2}+\, \alpha \sum_{i=1}^{n}\theta _{i}\\
# &= \frac{1}{2m}(X\theta - y)^{T}(X\theta -y )+\alpha \theta ^{T}\theta 
# \end{aligned}
# 

# %% [markdown]
#  - 3. Use training set, validation set and test set or [cross validation](https://scikit-learn.org/stable/modules/cross_validation.html) to acquire a better model.
#  ## 2 [Cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html)
# 
#     We often use cross validation to select parameters or to select models.
#     
# <center><img src="images/grid_search_workflow.png" alt="grid_search_workflow " style="zoom:30%;" /></center><br/>
# 
# <center><img src="images/grid_search_cross_validation.png" alt="grid_search_cross_validation " style="zoom:80%;" /></center>
# 
# For more detail: 
# - https://scikit-learn.org/stable/modules/cross_validation.html
# - https://en.wikipedia.org/wiki/Cross-validation_(statistics)
# - https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
#  
# 

# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

sample_cnt= 100
X = np.random.uniform(-3,3,size=100).reshape(-1, 1)

# curve using polynomial
θ0, θ1, θ2, θ3 = 2, 1, 0.5, 0
y = θ0 + θ1*X + θ2*(X**2) + θ3*(X**3)
y +=  np.random.normal(0,1,size = sample_cnt).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

plt.scatter(X,y)
plt.show()

# %%


model = Pipeline([
    ('poly', PolynomialFeatures()),
    ('std_scaler', StandardScaler()),
    ('ridge', Ridge())
])
params = { 'poly__degree':[1,2,3,4],
'ridge__alpha':list(x / 10 for x in range(0, 101)),
          'ridge__max_iter':[5,10,100,1000]}


grid_search = GridSearchCV(model, param_grid=params, cv=5)
grid_search.fit(X_train, y_train)
print('The highest score of cross validation:{:.3f}\n'.format(grid_search.best_score_))
print('The best parameters:{}\n'.format( grid_search.best_params_))
print('The highest score:{:.3f}\n'.format(grid_search.score(X_test,y_test)))

# y_hat = grid_search.predict(np.array(X_test))
# t = np.arange(len(X_test))
# plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
# plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
# plt.legend(loc='upper right')
# plt.grid()
# plt.show()

# %% [markdown]
# ## 3 Evaluation for Regression Models
# 
# Metrics commonly used to evaluate regression models are:
# 
# - Mean Absolute Error
# - Mean Squared Error (MSE)
# - Root Mean Squared Error (RMSE)
# - R-Squared
# 
# 
# 
# ### 1）Mean Absolute Error
# 
# ![img](images/MAE.png)
# 
# ```python
# def MAE(y, y_pre):
#     return np.mean(np.abs(y - y_pre))
# ```
# 
# **Advantages**
# 
# - MAE is not sensitive to outliers. Use MAE when you do not want outliers to play a big role in error calculated.
# 
# **Disadvantages**
# 
# - MAE is not differentiable globally. This is not convenient when we use it as a loss function, due to the gradient optimization method.
# 
# ### 2) Mean Squared Error (MSE)
# 
# ![img](images/MSE.png)
# ```python
# def MSE(y, y_pre):
#     return np.mean((y - y_pre) ** 2)
# ```
# 
# **Advantages**
# 
# - Graph of MSE is differantiable which means it can be easily used as a loss function.
# - MSE can be decomposed into variance and bias squared. This helps us understand the effect of variance or bias in data to the overall error.
# 
# ![img](images/dF42UODmyd4-qX_m-lPTqw.png)
# 
# ** Disadvantages**
# 
# - The value calculated MSE has a different unit than the target variable since it is squared. (Ex. meter → meter²)
# - If there exists outliers in the data, then they are going to result in a larger error. Therefore, MSE is not robust to outliers (this can also be an advantage if you are looking to penalize outliers).
# 
# 
# 
# 
# 
# ### 3) Root Mean Squared Error (RMSE)
# 
# ![img](images/RMSE.png)
# 
# ```python
# def RMSE(y, y_pre):
#     return np.sqrt(np.mean((y - y_pre) ** 2))
# ```
# 
# 
# 
# **Advantages**
# 
# - The error calculated has the same unit as the target variables making the interpretation relatively easier.
# 
# **Disadvantages**
# 
# - Just like MSE, RMSE is also susceptible to outliers.
# 
# ### 4) R-Squared
# 
# ![image-20220923123410702](images/image-20220923123410702.png)
# 
# ![image-20220923123435789](images/image-20220923123435789.png)
# 
# ```python
# def R2(y, y_pre):
#     u = np.sum((y - y_pre) ** 2)
#     v = np.sum((y - np.mean(y)) ** 2)
#     return 1 - (u / v)
# ```
# 
# Value closer to 1 is better.

# %% [markdown]
# 
# 
#   
# 

# %% [markdown]
# ## 4 LAB Assignment  
# Now it's time to implement linear regression techniques in practice. In this lab, you will use linear regression to fit a house price model. You will use some real-world data as the test set to evaluate your model. 
# 
# ### 4.1 Before Assignment
# #### 4.1.1 Load dataset & Import the required libraries
# **Datasets**: scikit-learn provides a number of datasets which can be directly loaded by using a function. First we load a  dataset as an example.

# %%
import warnings
from sklearn import datasets
boston = datasets.load_boston()
print(boston.DESCR)

# %% [markdown]
# See [sklearn website](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets) for details. To do this you have to import right packages and modules. 
# 
# #### 4.1.2 Preprocessing data
# This is a small dataset containing 506 samples and 13 attributes. We need to use proper visualization methods to have an intuitive understanding. We choose the sixth attribute and draw a scattering plot to see the distribution of samples. We use *matplotlib* for data visualization.

# %%
# Use one feature for visualization
x = boston.data[:,5]

# Get the target vector
y = boston.target

# Scattering plot of prive vs. room number
from matplotlib import pyplot as plt
plt.scatter(x,y)
plt.show()

# %% [markdown]
# It can be seen that the samples have some exceptional distributions at the top of the plot. They may be outliers owing to some practical operation during the data input (e.g., convert any price larger than 50 into 50). However, these data are harmful to the model training, and should be removed.

# %%
x = x[y<50.0]
y = y[y<50.0]

plt.scatter(x,y)
plt.show()

# %% [markdown]
# Now it can be seen that the data is nearly linear, although just in one dimension. Now we use X to denote all attributes

# %%
X = boston.data
y = boston.target

X = X[y<50.0]
y = y[y<50.0]

X.shape

# %% [markdown]
# #### 4.1.3 Split data
#  Now we divide the whole dataset into a training set and a test set using the the scikit-learn model_selection module.

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y)
y_train.shape

# %% [markdown]
# Usually we also use a validation set. When we use the test set for evaluation, the model will not be changed after the evaluation. However, sometime we want to optimize our model by changing its parameters according to prediction results. The solution is to split a validation set from the training set for adjusting our model. When we believe that the model is good enough, then we evaluate our model on the test set. A more rigorous and costly way is cross validation. With that method, the training set is divided into several pieces in the same size and take every piece as a validation set in turn.

# %% [markdown]
# #### 4.1.4 Training 
# ##### 1) Linear Regression
# Now we try to implement a simple linear regression model because the dataset seems linear.

# %%
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# %% [markdown]
# The model has been trained just by using a few lines of codes. Now let’s make a prediction for testing

# %%
# Make a prediction
y_0_hat = lin_reg.predict(X_test[0].reshape(1,-1))
y_0_hat

# %% [markdown]
# Notice that in scikit-learn, the standard interface for machine learning is
# 1) instantiate a learner with super parameters or none; 
# 2) use `fit()` method and feed the learner with training data; 
# 3) use `predict()` for prediction. 
# 
# Moreover, the data preprocessing algorithms also have the same interface, they just use `transform()` instead of `predict()`.
# 
# Below are the trained parameters.

# %%
y_test[0]

# %%
lin_reg.coef_

# %%
lin_reg.intercept_

# %% [markdown]
# Use the evaluation method to see if it is a good model. The `score()` method uses R-square.

# %%
lin_reg.score(X_test, y_test)

# %% [markdown]
# ##### 2) Polynomial Regression
# If you have understood the concept of linear regression, you can easily implement polynomial regression. 
# 
# #### 4.1.5 Evaluation model
# Checking the results on test set。

# %% [markdown]
# ### 4.2 LAB Assignment
# Please use the real world dataset, **California housing price**, for model training and evaluate the model’s prediction performance. You can use simple linear regression, polynomial regression or more complicated base functions such as Gaussian function or use regularization methods. Make sure at least **20% data for testing** and choose one evaluation method you think good. **Please do not just train your model and say that is good enough, you need to give your analysis**. For that end, validation or cross validation is needed. Compare the score in the training set and the validation set. If they are both good enough, then use the model on the test set.
# 
# **Your test set can only be used for final evaluation!**

# %%
# 导入加州房价
import warnings
from sklearn import datasets
california = datasets.fetch_california_housing()

print(california.DESCR)

# %%
# 预处理数据 并且按照8:2划分训练集以及测试集，并且选出40% 作为验证集
X = california.data
y = california.target
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8,random_state=1)
X_train, X_validation, y_train, y_validation = train_test_split(X, y, train_size = 0.6, random_state = 1)


# %%
# 使用 Linear Regression 训练
from sklearn.model_selection import cross_val_score


lin_reg = Pipeline([("std_scaler", StandardScaler()), 
                     ("lin_reg", LinearRegression()),                                             
])
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_train)

print("In train set")

mse = np.mean((y_pred - y_train)**2)
print("The MSE of the train set of Linear Regression model is", mse)


R2_score = lin_reg.score(X_train, y_train)
print("The R2 score of the train set of Linear Regression is ", R2_score)
cross_scores = cross_val_score(lin_reg, X_train, y_train, cv=5)
print("The mean cross validation score of Linear Regression is ", cross_scores.mean())

print()
print("In validation set")

y_pred = lin_reg.predict(X_validation)

mse = np.mean((y_pred - y_validation)**2)
print("The MSE of the validation set of Linear Regression model is", mse)

R2_score = lin_reg.score(X_validation, y_validation)
print("The R2 score of the validation set of Linear Regression is ", R2_score)

cross_scores = cross_val_score(lin_reg, X_validation, y_validation, cv=5)
print("The mean cross validation score of Linear Regression is ", cross_scores.mean())

# %%
# 使用 Polynomial Regression 进行训练
model = Pipeline([
    ('poly', PolynomialFeatures()),
    ('std_scaler', StandardScaler()), # 进行标准化，防止因为维度过大而对预测结果产生较大影响
    ('ridge', Ridge())
])
params = { 'poly__degree':[1,2,3,4],
'ridge__alpha':list(x / 10 for x in range(0, 101)),
          'ridge__max_iter':[5,10,100,1000]}


grid_search = GridSearchCV(model, param_grid=params, cv=5)
grid_search.fit(X_train, y_train)

print('The highest score of cross validation:{:.3f}\n'.format(grid_search.best_score_))
print('The best parameters:{}\n'.format( grid_search.best_params_))
print()
print('The highest score in train set :{:.3f}\n'.format(grid_search.score(X_train,y_train)))

cross_scores = cross_val_score(grid_search, X_train, y_train, cv=5)
print("The cross validation score of Polynomial Regression is ", cross_scores.mean())

print()
print('The highest score in validation set :{:.3f}\n'.format(grid_search.score(X_validation, y_validation)))
# print('The highest score in validation set :{:.3f}\n'.format(grid_search.score(X_train,y_validation)))

cross_scores = cross_val_score(grid_search, X_validation, y_validation, cv=5)
print("The cross validation score of Polynomial Regression is ", cross_scores.mean())


# %%
from random import gauss
from sklearn.gaussian_process import GaussianProcessRegressor
import mlxtend
from mlxtend.evaluate import bias_variance_decomp
gau_reg = GaussianProcessRegressor()
gau_reg.fit(X_train, y_train)

print("In train set")
# y_pred = gau_reg.predict(X_train)
# mse = np.mean((y_pred - y_train)**2)
# print("The mse of Gaussian Process Regression is ", mse)
R2_score = gau_reg.score(X_train, y_train)
print("The R2_score of Gaussian Process Regression is ", R2_score)
cross_scores = cross_val_score(gau_reg, X_train, y_train, cv=5)
print("The cross validation score of Gaussian Process Regression is ", cross_scores.mean())


print()
print("In validation set")
# y_pred = gau_reg.predict(X_validation)
# mse = np.mean((y_pred - y_validation)**2)
# print("The mse of Gaussian Process Regression is ", mse)
R2_score = gau_reg.score(X_validation, y_validation)
print("The R2_score of Gaussian Process Regression is ", R2_score)
cross_scores = cross_val_score(gau_reg, X_validation, y_validation, cv=5)
print("The cross validation score of Gaussian Process Regression is ", cross_scores.mean())

mse, bias, var = bias_variance_decomp(gau_reg, X_train, y_train, X_validation, y_validation, loss = 'mse', num_rounds = 10,random_seed=1)
print("The mse of Gaussian Process Regression is ", mse)

# %% [markdown]
# ## 分析与选择模型
# 通过对于线性回归、多项式回归以及高斯线性回归模型的比较和分析，发现对于均方差（MSE）的评估指数而言，线性回归模型的均方差相对最小，在训练集上为0.5269388232881851，在验证集上为0.521251943252734，，在验证集上的均方差比在训练集上还要低，这意味着该模型的预测误差较小，表现较好；同时，线性回归模型的R2指数在训练集和验证集的平均也更接近1，也说明了模型泛化的能力相对比较强；通过交叉熵的验证，发现线性回归模型无论在训练集还是在验证集上面，预测的结果和真实的结果都相对更加相似，表现的相对更好。
# 
# 所以在加州房价这个数据集上，我使用线性回归模型进行预测。
# 

# %%
lin_reg = Pipeline([("std_scaler", StandardScaler()), 
                     ("lin_reg", LinearRegression()),                                             
])
lin_reg.fit(X_train, y_train)
R2_score = lin_reg.score(X_test, y_test)
print("The R2 score of Linear Regression model is ", R2_score)


# %% [markdown]
# ### 4.3 Questions
# 1) Describe another real-world application where the regression method can be applied
# 
#     The regression method also can be used in predicting the yield of corns according to multiple cause such as rainfall, fertilizer, ect.
# 
# 2) What are the strengths of the linear/polynomial regression methods; when do they perform well?
# 
#     The strengths of the linear/polynomial regression methods is that it can be understanded and implemented easily, and its interpretability is good. They perform well when the feature is of great independency, and when the data is not complex.
# 
# 3) What are the weaknesses of the linear/polynomial regression methods; when do they perform poorly?
# 
#     The weaknesses of the linear/polynomial regression methods is that It is difficult to model nonlinear data or polynomial regression with correlation between data features. When the data is nonlinear, they will perform poorly.
# 
# 4) What makes the linear regression method a good candidate for the regression problem, if you have enough knowledge about the data?
# 
#     When the data is linear and have little dependency, the linear regression method is a good candidate for yhe regression problem.
# 

# %% [markdown]
# <center><font size=5 color='red'>Please complete lab4 Assignment  , and submit the result to bb  as required (Lab02_Assignment_Template.ipynb ) </font></center>

# %% [markdown]
#  <center><font size=10> Well done!👏 You have made it. </font></center>


