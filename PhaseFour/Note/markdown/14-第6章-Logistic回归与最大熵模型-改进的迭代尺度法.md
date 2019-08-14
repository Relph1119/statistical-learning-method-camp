## 第6章-Logistic回归与最大熵模型-改进的迭代尺度法
### 改进的迭代尺度法
&emsp;&emsp;最大熵模型的形式是一个指数形式，需要对$w$进行参数估计，求解$w$的方法是最大似然估计，其似然函数为$$L(w)=\sum_{x, y} \big[\tilde{P}(x, y) \sum_{i=1}^{n} w_i f_i(x, y)\big]-\sum_{x} \big[ \tilde{p}(x) \ln z_w(x) \big]$$
&emsp;&emsp;其中$\tilde{P}(x, y)$是$x$和$y$的经验分布，根据训练集当中特定的$x$和$y$的个数占总训练集实例的比值；$w_i$为所求，一共需要求解$n$个$w$；$f_i(x,y)$是已经给定的特征函数，取值为0或1；$Z_w(x)$表示关于给定$x$的$y$条件分布的归一化系数，$\displaystyle Z_w(x)=\sum_y \exp \big[ \sum_i w_i f_i(x,y) \big]$，在该公式中，也存在$w$。  
&emsp;&emsp;这个函数$L(w)$是关于$w$的，函数形式比较复杂，有指数上面的$w$并还要取对数，直接对其求导比较难求，使用了迭代的方法求导。首先给$w$初值，然后更新$w$，使$L(w)$的值不断增大，从而求得$L(w)$的最大值。  

### 求解最大似然函数
&emsp;&emsp;假设存在一个$\delta$，使得$w \rightarrow w + \delta$。对于给定的经验分布$\tilde{P}(x,y)$，模型参数从$w$到$w+\delta$，对数似然函数的改变量为：$$L(w+\delta) - L(w) = \sum_{x,y} \tilde{P}(x, y) \sum_{i=1}^n \delta_i f_i(x,y) - \sum_x \tilde{P}(x)\ln \frac{Z_{w+\delta}(x)}{Z_w(x)}$$
&emsp;&emsp;通过更新$w$变为$w+\delta$，使得似然函数变大，变大的值为上述公式，故求解等号后面的最大值。  
$\because -\ln \alpha \geqslant 1 - \alpha, \alpha > 0$  
$\begin{aligned} 仅观察这项： - \sum_x \tilde{P}(x) \ln \frac{Z_{w+\delta}(x)}{Z_w(x)} 
& \geqslant \sum_x \tilde{P}(x) \big[ 1 - \frac{Z_{w+\delta}(x)}{Z_w(x)} \big] \\
& = 1 - \sum_x \tilde{P}(x) \frac{Z_{w+\delta}(x)}{Z_w(x)}
\end{aligned}$  
$\begin{aligned} \because \frac{Z_{w+\delta}(x)}{Z_w(x)} 
&= \frac{1}{Z_w(x)} \cdot \sum_y \exp \big(\sum_{i=1}^n(w_i+\delta)f_i(x,y) \big) \\
&= \frac{1}{Z_w(x)} \cdot \sum_y \exp \big[ \sum_{i=1}^n w_i f_i(x,y) + \sum_{i=1}^n \delta_i f_i(x,y) \big] \\
&= \frac{1}{Z_w(x)} \cdot \sum_y \big[ \exp \sum_{i=1}^n w_i f_i(x,y) \cdot \exp \sum_{i=1}^n \delta_i f_i(x,y) \big] \\
&= \sum_y \frac{1}{Z_w(x)} \big[ \exp \sum_{i=1}^n w_i f_i(x,y) \cdot \exp \sum_{i=1}^n \delta_i f_i(x,y) \big] \\
\end{aligned}$  
$\because $最大熵模型为$\displaystyle P_w(y|x) = \frac{1}{Z_w(x)} \exp \big( \sum_{i=1}^n w_i f_i(x,y) \big)$  
$\begin{aligned} \therefore \frac{Z_{w+\delta}(x)}{Z_w(x)}
&= \sum_y \frac{1}{Z_w(x)} \big[ \exp \sum_{i=1}^n w_i f_i(x,y) \cdot \exp \sum_{i=1}^n \delta_i f_i(x,y) \big] \\
&= \sum_y \big[ P_w(y|x) \exp \big( \sum_{i=1}^n \delta_i f_i(x,y) \big) \big]
\end{aligned}$  
$\begin{aligned}  将上式代入并整理：L(w+\delta) - L(w) 
&= \sum_{x,y} \tilde{P}(x, y) \sum_{i=1}^n \delta_i f_i(x,y) - \sum_x \tilde{P}(x)\ln \frac{Z_{w+\delta}(x)}{Z_w(x)} \\
&\geqslant \sum_{x,y} \tilde{P}(x, y) \sum_{i=1}^n \delta_i f_i(x,y) + 1 - \sum_x \tilde{P}(x) \sum_y \big[ P_w(y|x) \exp \big( \sum_{i=1}^n \delta_i f_i(x,y) \big) \big]  \\
&= A(\delta|w)
\end{aligned}$  
目前得到了改变量的下界，如果想让该值最大，就最大化$A(\delta|w)$值。  
$\because (e^{\sum \delta_i f_i})' = e^{\sum \delta_i f_i} \cdot f_i$，求导之后依然有其他的$\delta_i$分量，但是希望对$\delta_i$求导之后能得到只关于$\delta_i$的函数，使得$g(\delta_i)=0$，需要对$\displaystyle \exp \big( \sum_{i=1}^n \delta_i f_i(x,y) \big)$再进行变换，需要用到Jesson不等式。

> **Jesson不等式：**  
对一个凸函数$\phi(x)$，已知权重$a_i$，$\sum a_i = 1$，下列不等式成立：$$\phi(\sum_i a_i x_i) \leqslant \sum_i a_i \phi(x_i)$$

根据Jesson不等式，可得：
$$\begin{aligned}  \exp \big( \sum_i \delta_i f_i(x,y) \big) 
&= \exp (\sum_i \frac{f_i(x,y)}{f^\#(x,y)} f^\#(x,y) \delta_i) \\
& \leqslant \sum_i \frac{f_i(x,y)}{f^\#(x,y)} \exp(f^\#(x,y) \delta_i)
\end{aligned}$$   
$\displaystyle \therefore A(\delta |w) \geqslant \sum_{x,y} \tilde{P}(x, y) \sum_{i=1}^n \delta_i f_i(x,y) + 1 - \sum_x \tilde{P}(x) \sum_y \big[ P_w(y|x) \sum_i \frac{f_i(x,y)}{f^\#(x,y)} \exp(f^\#(x,y) \delta_i) \big] = B(\delta | w)$  
&emsp;&emsp;经过上述放缩，$B(\delta|w)$是对数似然函数改变量的一个新的下界。对$B(\delta|w)$求导，使得导数等于0。 

> &emsp;&emsp;再考虑迭代尺度法，收敛的条件是$L(w+\delta)$和$L(w)$的差值是接近0的，即最大化没有提升空间了，当$L(w + \delta) - L(w) = 0$，则$\delta=0$。  
&emsp;&emsp;会有下面思考：放缩了两次，怎样保证最大化下界，最后收敛时，可以使得$L(w + \delta) - L(w)$最大化呢？可将$\delta=0$带入下界公式中，观察得到的值是否为0。当$\delta=0$时，满足$A(\delta|w)=B(\delta|w)=0$。  

&emsp;&emsp;求$B(\delta|w)$对$\delta_i$的偏导数，并令偏导数为0可得：$$\sum_{x,y} \tilde{P}(x) P_w(y|x)f_i(x,y) \exp (\delta_i f^\#(x,y)) = E_{\tilde{P}}(f_i)$$
&emsp;&emsp;求解使得该等式成立的$\delta_i$，没有一个显示的形式，对于这样一个方程，要如何寻找零点？此时可以用牛顿迭代法。  
&emsp;&emsp;上述问题变为：已知$\displaystyle g(\delta_i) = \sum_{x,y} \tilde{P}(x) P_w(y|x)f_i(x,y) \exp (\delta_i f^\#(x,y)) - E_{\tilde{P}}(f_i)$，令$g(\delta_i)  = 0$，求解$\delta_i$。  
&emsp;&emsp;求解步骤：先给出$\delta_i$的初值，更新$\displaystyle \delta_i^{(k+1)} = \delta_i^{(k)} - \frac{g(\delta_i^{(k)})}{g'(\delta_i^{(k)})}$ 