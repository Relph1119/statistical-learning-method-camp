# 第9章-EM算法及推广-EM算法的推导{docsify-ignore-all}

&emsp;&emsp;EM算法本质上是估计一个密度函数，在估计密度函数时，通过观测值采用最大化似然函数估计参数$\theta$。EM算法针对的是含有隐变量的密度函数的估计问题，这个时候直接最大化似然函数会比较困难，借鉴的算法思路和第6章改进的迭代尺度法是类似的，通过不等式放缩，将最大化观测数据的对数似然函数转化为另外一个比较好实现的式子，在推导EM算法时，与书上的数学符号保持一致。  
&emsp;&emsp;首先有一个需要观测的向量$\theta$，观测数据$Y=(y_1,y_2,\cdots,y_N)$，隐变量$Z=(z_1,z_2,\cdots,z_N)$，当求解$\theta$时，似然函数为$$\begin{aligned} L(\theta)
&= \ln P(Y|\theta) \\
&= \ln \sum_Z P(Y,Z|\theta) \\
&= \ln \left( \sum_Z P(Z|\theta) P(Y|Z,\theta) \right)
\end{aligned}$$&emsp;&emsp;假设在第$i$次迭代后$\theta$的估计值为$\theta^{(i)}$，希望新估计值$\theta$能使$L(\theta)$增加，即$L(\theta) > L(\theta^{(i)})$，则可计算两者的差：  
$$L(\theta)-L(\theta^{(i)}) 
= \ln \left( \sum_Z P(Z|\theta) P(Y|Z,\theta) \right) - \ln P(Y|\theta^{(i)})$$  
&emsp;&emsp;一般来说，对$\ln P_1 P_2 \cdots P_N$比较好处理，但是如果是$\ln \sum P_1 P_2$就不好处理，为了将$\sum$求和符号去掉，用Jenson不等式进行缩放处理。 

> **Jenson不等式：** $$f(\sum_i \alpha_i x_i) \geqslant \sum_i \alpha_i f(x_i)$$其中函数$f$是凸函数，那么对数函数也是凸函数，$\displaystyle \sum_i \alpha_i = 1$，$\alpha_i$表示权值，$0 \leqslant \alpha_i \leqslant 1$

&emsp;&emsp;对于上述形式，对$Z$求和，要如何凑出来一个具有Jenson不等式中的$\alpha_i$呢？很容易想到，关于$Z$的密度函数，该密度函数取值求和为1，需要构造一个$Z$的概率分布。  
$\begin{aligned}L(\theta)-L(\theta^{(i)}) 
=& \ln \left( \sum_Z P(Z|\theta) P(Y|Z,\theta) \right) - \ln P(Y|\theta^{(i)}) \\
=& \ln \left( \sum_Z P(Z|Y,\theta^{(i)}) \frac{P(Z|\theta)P(Y|Z,\theta)}{P(Z|Y,\theta^{(i)})}  \right) - \ln P(Y|\theta^{(i)}) 
\end{aligned}$  
利用Jesson不等式，$\displaystyle \ln \left( \sum_Z P(Z|Y,\theta^{(i)}) \frac{P(Z|\theta)P(Y|Z,\theta)}{P(Z|Y,\theta^{(i)})}  \right) \geqslant \sum_Z P(Z|Y,\theta^{(i)}) \ln \frac{P(Z|\theta)P(Y|Z,\theta)}{P(Z|Y,\theta^{(i)})}$  
$\displaystyle \because \ln P(Y|\theta^{(i)}) = \sum_Z P(Z|Y,\theta^{(i)}) \ln P(Y|\theta^{(i)})$  
$\begin{aligned} \therefore L(\theta)-L(\theta^{(i)}) 
& \geqslant \sum_Z P(Z|Y,\theta^{(i)}) \ln \frac{P(Z|\theta)P(Y|Z,\theta)}{P(Z|Y,\theta^{(i)})} - \sum_Z P(Z|Y,\theta^{(i)}) \ln P(Y|\theta^{(i)}) \\
&= \sum_Z P(Z|Y,\theta^{(i)}) \ln \frac{P(Z|\theta)P(Y|Z,\theta)}{P(Z|Y,\theta^{(i)}) P(Y|\theta^{(i)})}
\end{aligned}$  
令$\displaystyle B(\theta,\theta^{(i)}) = L(\theta^{(i)}) + \sum_Z P(Z|Y,\theta^{(i)}) \ln \frac{P(Z|\theta)P(Y|Z,\theta)}{P(Z|Y,\theta^{(i)}) P(Y|\theta^{(i)})} $  
$\therefore L(\theta) \geqslant B(\theta,\theta^{(i)})$，也就是说$B(\theta,\theta^{(i)})$是$L(\theta)$的一个下界，要最大化$L(\theta)$，可换成最大化$B(\theta,\theta^{(i)})$，这个和之前的改进的迭代尺度法的思路是一致的。  
$\begin{aligned} \therefore \theta^{(i+1)} 
=& \mathop{\arg \max} \limits_{\theta} B(\theta,\theta^{(i)})  \\
=& \mathop{\arg \max} \limits_{\theta} \left( \sum_Z P(Z|Y,\theta^{(i)}) \ln P(Z|\theta)P(Y|Z,\theta) \right) \\
=& \mathop{\arg \max} \limits_{\theta} \left( \sum_Z P(Z|Y,\theta^{(i)}) \ln P(Y,Z|\theta) \right) 
\end{aligned}$  
$\displaystyle \because Q(\theta, \theta^{(i)}) = \sum_Z \ln P(Y,Z|\theta) P(Z|Y,\theta^{(i)}) $  
$\displaystyle \therefore \theta^{(i+1)} = \mathop{\arg \max} \limits_{\theta} Q(\theta, \theta^{(i)})$  
&emsp;&emsp;等价于EM算法的M步，E步等价于求$\displaystyle \sum_Z P(Z|Y,\theta^{(i)}) \ln P(Y,Z|\theta)$，以上就得到了EM算法，通过不断求解下界的极大化逼近求解对数似然函数极大化。