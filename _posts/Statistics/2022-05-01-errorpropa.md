---
title:  "Error Propagation for Statistical Modeling"
categories:
  - Stat-num
tags:
  - Error propagation
  - Statistics
  
classes: wide
toc: false

---

Let's say you have a dataset with $$x$$ and $$y$$ features,
and then you want to build a model, 

$$u=f(x,y).$$

Also, you want to calculate not only the value $u$, but also want to estimate its **error**.
The problem is that you know how to calculate the mean and error of $$x$$ and $$y$$ using their data points,
how to calculate the mean of $$u$$, which is $$f(\bar{x},\bar{y})$$, 
where $$\bar{x}$$ and $$\bar{y}$$ are mean values of $$x$$ and $$y$$, respectively,
but you don't know how to use them to calculate the **error** of $$u$$.
After reading this post, you will be able to do that. It's actually easy.


# Propagation of error

Let's say you have mean values of $$x$$ and $$y$$, $$\bar{x}$$ and $$\bar{y}$$,
and their errors, $$\sigma_x$$ and $$\sigma_y$$.
The error of $$u$$ is defined as

$$\sigma_u^2 = E[(u-\bar{u})^2], \tag{1}$$

and from the **first order Taylor expansion**,

$$u-\bar{u} \approx \frac{\partial u}{\partial x}(x-\bar{x}) +  \frac{\partial u}{\partial y}(y-\bar{y}). \tag{2}$$


By inserting Eq.(2) to Eq.(1),

$$\begin{aligned} \sigma_u^2 
&= E[(u-\bar{u})^2] \\
&= E[(\frac{\partial u}{\partial x}(x-\bar{x}) +  \frac{\partial u}{\partial y}(y-\bar{y}))^2] \\
&= (\frac{\partial u}{\partial x})^2\sigma_x^2 + (\frac{\partial u}{\partial y})^2\sigma_y^2 
+ 2E[(x-\bar{x})(y-\bar{y})]\frac{\partial u}{\partial x}\frac{\partial u}{\partial y} \\
\end{aligned} \tag{3}$$

The last term of Eq.(3) contains covariance, $$E[(x-\bar{x})(y-\bar{y})]=cov(x,y)$$,
which becomes 0 if $$x$$ and $$y$$ are independent. Then finally, 

$$\sigma_u = \sqrt{(\frac{\partial u}{\partial x})^2\sigma_x^2 + (\frac{\partial u}{\partial y})^2\sigma_y^2 }.$$



## Example
Let's say you are trying to assign an error of $$u=x/y$$.
Then

$$\begin{aligned} \sigma_u 
&= \sqrt{(\frac{\partial u}{\partial x})^2\sigma_x^2 + (\frac{\partial u}{\partial y})^2\sigma_y^2 }\\
&= \sqrt{(\frac{1}{y})^2\sigma_x^2 + (\frac{x}{y^2})^2\sigma_y^2 }\\
&= |u|\sqrt{(\frac{\sigma_x}{x})^2 + (\frac{\sigma_y}{y})^2 }\\
\end{aligned}$$


This method requires an assumption that Eq.(2) is valid.
For example, if $$u=f(x,y)$$ is not a continuous function for the given $$x, y$$ range,
this method is not applicable.
Then you might try althernative way.

# Alternative way: Simulation
Now, think about when 

$$ u = \sqrt{x^2+y^2}. $$

Then

$$ \sigma_u = \sqrt{\frac{x^2\sigma_x^2+y^2\sigma_y^2}{x^2+y^2}}.$$


What happens when $$x$$ and $$y$$ are distributed around zero?
Then the error propagation formula will give you wrong answer, and you should try alternative way: simulation.

Check out details of calculation and code implementation at my Jupyter Notebook post.
[Link to my GitHub](https://github.com/minjung-mj-kim/minjung-mj-kim.github.io/blob/master/_posts/Statistics/error_propa.ipynb)

