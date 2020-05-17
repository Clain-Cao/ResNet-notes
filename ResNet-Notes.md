# ResNet（残差神经网络）

标签（空格分隔）： CNN

---

##Introduction
-  ResNet 是一个很神奇的神经网络（嗯，就这样）, 首先从深度来讲从 LeNet 的 5 层到 AlexNet 的 8 层，再到 VGG-Net 的 16 和 19 层，随着 layer 的增加，training error 在不断的减少，test 的 accuracy 在不断地增加，According to detection ，是不是随着 layer 的增加，accuracy 就不断减少呢？ that's not correct ，因为在随着网络深度的不断增加，经过 Backprop 的 Gradient 会随着 layer 的增加而减少，Finally ，会出现网络退化问题，下图说明了一切。
![][1]

> 上图里面记录了两个 plain Network 的情况，一个 20-layer 的 Network 和一个 56-layer 的 Network , test 的 error 随着 network 层数的增加而增加 , why ? Actually , we're going to think of over-fitting , But , According to Graph about traning error , 56-layer 并不是表现出了 over-fitting , 所以就考虑 gradient 消失 , 拿 sigmoid 做例子（如下图） , 对于幅度为1的信号，每向后传递一层，梯度就衰减为原来的 0.25 ，层数越多，衰减越厉害，导致无法对前面网络层的权重进行有效的调整。那么 , 如何又能加深网络层数、又能解决梯度消失问题、又能提升模型精度呢 ? So , 就有了 ResNet 这个神经网络啦~~
![plain network 的 误差表][2]

---
##ResNet Architecture##

`首先 ResNet 具有许多不同 layer 的 version 。选取一个34-layer 的 review 。`

###首先来 review 一下残差神经元 , 残差神经元（如下图）
> 前面描述了一个实验结果现象，在不断加神经网络的深度时，模型准确率会先上升然后达到饱和，再持续增加深度时则会导致准确率下降，那么我们作这样一个假设：假设现有一个比较浅的网络（Shallow Net）已达到了饱和的准确率，这时在它后面再加上几个恒等映射层（ Identity mapping ，也即 `y = x` ，输出等于输入），这样就增加了网络的深度，并且起码误差不会增加，也即更深的网络不应该带来训练集上误差的上升。而这里提到的使用恒等映射直接将前一层输出传到后面的思想，便是著名深度残差网络ResNet的灵感来源。
 ResNet 引入了残差网络结构（residual network），通过这种残差网络结构，可以把网络层弄的很深（据说目前可以达到 `1000` 多层），并且最终的分类效果也非常好，残差网络的基本结构如下图所示，很明显，该图是带有跳跃结构的：
![残差神经单元结构][3]
残差网络借鉴了高速网络 ( Highway Network ) 的跨层链接思想，但对其进行改进 ( 残差项原本是带权值的，但 ResNet 用恒等映射代替之) 。假定某段神经网络的输入是 x ，期望输出是 H(x) ，即 H(x) 是期望的复杂潜在映射，如果是要学习这样的模型，则训练难度会比较大。
回想前面的假设，如果已经学习到较饱和的准确率 (或者当发现下层的误差变大时) , 那么接下来的学习目标就转变为恒等映射的学习，也就是使输入 x 近似于输出 H(x) , 以保持在后面的层次中不会造成精度下降。
在上图的残差网络结构图中 , 通过 “ shortcut connections (捷径连接)” 的方式，直接把输入 x 传到输出作为初始结果，输出结果为 `H(x) = F(x) + x` ，当 `F(x) = 0` 时，那么`H(x) = x`，也就是上面所提到的恒等映射。于是 , ResNet 相当于将学习目标改变了，不再是学习一个完整的输出，而是目标值 H(x) 和 x 的差值，也就是所谓的残差:
    $$F(x) = H(x)-x$$  因此，后面的训练目标就是要将残差结果逼近于 0 ，使到随着网络加深，准确率不下降。这种残差跳跃式的结构，打破了传统的神经网络 `n - 1` 层的输出只能给 n 层作为输入的惯例，使某一层的输出可以直接跨过几层作为后面某一层的输入，其意义在于为叠加多层网络而使得整个学习模型的错误率不降反升的难题提供了新的方向。
至此，神经网络的层数可以超越之前的约束，达到几十层、上百层甚至千层，为高级语义特征提取和分类提供了可行性。

###关于 残差的公式推导
> 为什么残差学习相对更容易，从直观上看残差学习需要学习的内容少，因为残差一般会比较小，学习难度小点。不过我们可以从数学的角度来分析这个问题，首先残差单元可以表示为：
$\begin{align} & {{y}_{l}}=h({{x}_{l}})+F({{x}_{l}},{{W}_{l}}) \\ & {{x}_{l+1}}=f({{y}_{l}}) \\ \end{align}$
其中 $x_{l}$ 和 $x_{l+1}$ 分别表示的是第 l 个残差单元的输入和输出，注意每个残差单元一般包含多层结构。 F 是残差函数，表示学习到的残差，而 $h(x_{l})=x_{l}$ 表示恒等映射， f 是ReLU激活函数。基于上式，我们求得从浅层 l 到深层 L 的学习特征为：$${{x}_{L}}={{x}_{l}}+\sum\limits_{i=l}^{L-1}{F({{x}_{i}}},{{W}_{i}})$$
利用链式规则，可以求得反向过程的梯度：
$$\frac{\partial loss}{\partial {{x}_{l}}}=\frac{\partial loss}{\partial {{x}_{L}}}\cdot \frac{\partial {{x}_{L}}}{\partial {{x}_{l}}}=\frac{\partial loss}{\partial {{x}_{L}}}\cdot \left( 1+\frac{\partial }{\partial {{x}_{L}}}\sum\limits_{i=l}^{L-1}{F({{x}_{i}},{{W}_{i}})} \right)$$
式子的第一个因子 \frac{\partial loss}{\partial {{x}_{L}}} 表示的损失函数到达 L 的梯度，小括号中的1表明短路机制可以无损地传播梯度，而另外一项残差梯度则需要经过带有weights的层，梯度不是直接传递过来的。残差梯度不会那么巧全为-1，而且就算其比较小，有1的存在也不会导致梯度消失。所以残差学习会更容易。要注意上面的推导并不是严格的证明。

###再来看两个不同的残差神经单元
> ![两种不同的残差神经单元][4]
 左图是对于那些不太深的 ResNet 网络来讲的 , 而右图是相对于比较深的神经网络来讲的(也就是两种结构分别针对`ResNet34`（左图）和 `ResNet50/101/152` （右图）） why ? Don't worried , 让我慢慢道来 , 因为目的主要是降低 parameter 的数量 , 假设两个图的 input 和 output 都是 256-dimension 的 , 那么 左图 total 就是 3×3×256×256×2 = 1179648 个 weight parameter , 右图是先用 1×1 的卷积进行降维，再用 1×1 的卷积进行升维 , total 就是 1×1×256×64 + 3×3×64×64 + 1×1×64×256 = 69632 , 从而将 parameter 降低了 10 倍以上 , 因此右图是降低 parameter 从而减少计算量 。
 
---
##34-layer 的整体结构
![ResNet Architecture][5]
> 图片中的分别有虚线和实线的 shortcut connection 
 ○ 实线代表统一用$$H(x) = F(x) + x $$
 ○ 虚线用由于维度或者通道数不一定相同，所以用$$H(x) = F(x) + Wx$$ W 为卷积核 , 通过卷积操作将通道数变为相同，同时我认为在这里使用卷积操作也是为了再次提取特征的。

---
##ResNet不同层数的结构
![不同层数的ResNet][6]
 


  [1]: https://static.oschina.net/uploads/space/2018/0223/111417_u90M_876354.png
  [2]: https://static.oschina.net/uploads/space/2018/0223/111541_7sm1_876354.png
  [3]: https://static.oschina.net/uploads/space/2018/0223/111635_C81Q_876354.png
  [4]: https://static.oschina.net/uploads/space/2018/0223/111833_m5OE_876354.png
  [5]: https://static.oschina.net/uploads/space/2018/0223/111741_FjZa_876354.png
  [6]: https://upload-images.jianshu.io/upload_images/2228224-2ff6ccbf1b14840a.png?imageMogr2/auto-orient/