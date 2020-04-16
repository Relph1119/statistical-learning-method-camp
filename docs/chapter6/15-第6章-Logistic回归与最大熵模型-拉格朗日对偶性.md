# 第6章-Logistic回归与最大熵模型-拉格朗日对偶性{docsify-ignore-all}
&emsp;&emsp;拉格朗日对偶性是用在解优化问题中的一个性质，在第6章最大熵模型的推导过程和第7章SVM都用到了这个性质，主要是按照附录C中的内容来介绍。

## 原始问题
&emsp;&emsp;优化问题的一般形式，对于任意一个优化问题，都有一个需要优化的目标函数为$f$，需要优化的变量为$x$，最小化$f(x)$，会有一些约束条件，有不等式约束$c_i(x) \leqslant 0, i= 1, 2, \cdots, k$，也有等式约束$h_j(x)=0,j=1,2,\cdots,l$，很多问题并不是两个约束都有，比如第6章中，只有等式约束（即$k=0$），第7章SVM中，只有不等式约束（即$l=0$）。  
原始问题$P$：$$\begin{array}{ll}
\mathop{\min}\limits_{x \in R^n} & f(x) \\ 
\text { s.t. } & c_i(x) \leqslant 0, i= 1, 2, \cdots, k \\ 
{}& h_j(x)=0,j=1,2,\cdots,l
\end{array}$$  

## 拉格朗日函数
&emsp;&emsp;在不等式约束中，有多少个不等式就有多少个$\alpha$，有多少个等式就有多少个$\beta$。
$\displaystyle L(x, \alpha, \beta) = f(x) + \sum_{i=1}^k \alpha_i c_i(x) + \sum_{j=1}^l \beta_j h_j(x)$  
其中$\alpha_i \geqslant 0$目标函数$f$，优化变量$x$，可行域为满足所有约束条件的$x$，最优值为$x^*$，对应的目标函数为$p^*=f(x^*)$。</br>    
**结论1：** 可以用拉格朗日函数的极小极大问题表示为原始问题，记为$$P=\min_x \max_{\alpha, \beta} L(x, \alpha, \beta)= \min_x \left \{ \begin{array}{ll} f(x) & c_i(x) \leqslant 0, h_j(x)=0\\
\infty & \textrm{其他}
\end{array}  \right.$$
**结论2：** 对偶问题$D$可以用拉格朗日函数的极大极小问题表示，记为$$\begin{array}{ll}
\displaystyle \max_{\alpha, \beta} \min_x & L(x, \alpha, \beta) \\ 
\text { s.t. } & \alpha_i \geqslant 0, i= 1, 2, \cdots, k
\end{array}$$  
其中，最优解为$\alpha^*,\beta^*$，对应的值记为$d^*$。

## 总结
&emsp;&emsp;已知原始的最优化问题，根据拉格朗日函数定义出了原始问题的对偶问题，原始问题是关于$x$求目标函数的极小值，对偶问题是关于$\alpha，\beta$求目标函数的极大值，原始问题相当于极小极大化拉格朗日函数，对偶问题相当于极大极小化拉格朗日函数。

## 定理C.1
&emsp;&emsp;本定理是讲了原始问题$p^*$和对偶问题$d^*$的关系。若原始问题和对偶问题都有最优解，则：$$d^*=\max_{\alpha,\beta:\alpha \geqslant 0} \min_x L(x,\alpha,\beta) \leqslant \min_x \max_{\alpha,\beta:\alpha \geqslant 0} L(x, \alpha,\beta) = p^*$$  
**证明：**  
$\begin{aligned} \because d^*
&= \max_{\alpha,\beta} \min_x L(x,\alpha,\beta) \\ 
&\leqslant \max_{\alpha,\beta} \min_{x \in 可行域} L(x,\alpha,\beta) \\
&\leqslant \max_{\alpha,\beta} \min_{x \in 可行域} f(x) \\
&\leqslant \min_{x \in 可行域} f(x) = p^*
\end{aligned}$  
$\therefore d^* \leqslant p^*$  
**直观理解：**  对偶问题的最优解是一个原始问题极小值的下界。  

## 定理C.2  
&emsp;&emsp;要使$d^*=p^*$，需要满足两个条件：（1）原问题是凸优化问题；（2）原问题满足Slater条件，$d^*=p^*$称为强对偶性，$d^* \leqslant p^*$称为弱对偶性。定理C.2给出了对偶性的一个充分条件。  
&emsp;&emsp;优化问题就是在一个可行域里，求解目标函数的最优值，当可行域是一个凸集，目标函数$f(x)$为凸函数时，该问题就是**凸优化问题**，也就是说，在一个凸集上求解凸函数的极小值问题。  
&emsp;&emsp;**凸集**是从某空间中的任意取两个点，这两个点连成的线段上面的每一个点依然在这个点集中。  
&emsp;&emsp;**凸函数**是在这个函数上，任取两个点，所连成的线段在这个函数上方。当$C_i(x)$是一个凸函数，$h_j(x)$函数是关于$x$的线性函数（仿射函数）时，可行域是一个**凸集**。  
&emsp;&emsp;**Slater条件**是针对约束中，不等式约束的一个限制，之前所讲的每一个不等式约束，所满足的$x$可以看成一个集合，在凸优化问题中，每一个满足这个条件的$x$都是一个凸集，如果$x$同时满足所有不等式约束，那么所有不等式约束对应凸集的交集依然是一个凸集，Slater条件要求这些凸集的交集是有内点的，不仅仅只是边界，边界中还有一些点。  
&emsp;&emsp;Slater条件相对比较宽松，一般只需要确定原始问题是凸优化问题即可，那么强对偶性就是成立的，可以通过求解对偶问题，求解原问题的最优值。以上条件可以得到$d^*=p^*=L(x^*,\alpha^*,\beta^*)$。定理C.2告诉我们，在什么样的条件下可以通过对偶问题求解原始问题。  

## 定理C.3(KKT条件)
&emsp;&emsp;当原始问题满足强对偶性时，才可以使用KKT条件。  
KKT条件如下：$$\begin{array}{ll} 
\nabla_x L(x^*, \alpha^*, \beta^*) = 0 \\
\alpha_i^* c_i(x^*) = 0 \\
c_i(x^*) \leqslant 0 \\
\alpha_i^* \geqslant 0 \\
h_j(x^*) = 0
\end{array}$$
其中，$\nabla_x L(x^*, \alpha^*, \beta^*) = 0$为拉格朗日函数直观地对$x$求导等于0，$c_i(x^*) \leqslant 0$和$h_j(x^*) = 0$u是原始问题的约束，$\alpha_i^* \geqslant 0$是对偶问题的约束，$\alpha_i^* c_i(x^*) = 0$称为互补松弛条件，通过KKT条件，求出原始问题和对偶问题的解。