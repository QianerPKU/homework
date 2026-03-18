# 问题：求解泊松方程：

$$ \nabla ^ 2 \phi = -f $$

其中（转换为柱坐标）：

$$ f = \frac{0.8}{1+e^{(\sqrt{\rho^2+z^2}-r_0)/0.7}} [c \cdot fm^{-3}] $$

$$ r_0 = 10.0(1+1.0 \cdot \frac{1}{4} \sqrt{\frac{5}{\pi}} (3 \frac{z^2}{z^2+ \rho ^2} -  1) )$$

拉普拉斯算符展开为（z轴旋转对称）：

$$ \nabla ^ 2 \phi = \frac{\partial ^2 \phi}{\partial \rho^2} + \frac{1}{\rho} \frac{\partial \phi}{\partial \rho} + \frac{\partial ^2 \phi}{\partial z^2} $$

## 算法：

在 $(\rho,z)$ 空间中取格点 $\phi_{i,j}$ （i表示 $\rho$ 维度，j表示z维度） ，边长设为 $\Delta h$ ，用离散方法近似计算拉普拉斯算符：

$$\nabla^2 \phi_{i, j} \approx \frac{1}{(\Delta h)^2} \left[ \left(1 + \frac{1}{2i}\right)\phi_{i+1, j} + \left(1 - \frac{1}{2i}\right)\phi_{i-1, j} + \phi_{i, j+1} + \phi_{i, j-1} - 4\phi_{i, j} \right]$$

然后用梯度下降法优化迭代得到收敛解：

$$loss = \overline{ (\nabla ^ 2 \phi + f)^2 } $$

## 边界条件的处理：

当 $ \rho = 0 $ 时（即(0,0)点），拉普拉斯算符中的第二项需要做洛必达定理：

$$ \nabla ^ 2 \phi = 2 \frac{\partial ^2 \phi}{\partial \rho^2} + \frac{\partial ^2 \phi}{\partial z^2} \approx \frac{1}{(\Delta h)^2} \left[ 4\phi_{1, j} + \phi_{0, j+1} + \phi_{0, j-1} - 6\phi_{0, j} \right] $$


当 $ \rho \to +\infty$ 时，将电荷分布按多级展开近似到一阶（点电荷阶）：

$$Q_{total} = \sum_{i,j} f_{i,j} \Delta V_{i,j} = \sum_{i,j} f_{i,j} \cdot 2\pi (i\Delta h) (\Delta h)^2$$

并用：

$$ \phi = \frac{Q_{total}}{4\pi \sqrt{\rho^2 + z^2}} = \frac{Q_{total}}{4\pi \Delta h \sqrt{i^2+j^2}}$$

固定边界上的格点的f取值。

## 实验发现

### 优化器比较

sgd

adam

L-BFGS

### $\phi$ 的初值选择

zeros

randn

均匀带电球

先粗粒度solve，再细粒度微调（粗粒度按均匀带点球初始化）