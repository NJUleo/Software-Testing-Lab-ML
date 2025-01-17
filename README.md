# Software-Testing-Lab-ML

### 算法思路和一些个人的思考

从整体上来说，目前这个攻击实际上的目标是，针对每一个输入，对他进行一个微小的扰动，让扰动尽可能小而结果的偏差尽可能大，而且在多分类模型的基础之上，应该试图让攻击数据的结果不但尽可能偏离正确结果，而应该尽可能以一个比较大的置信值落入一个错误的分类，依次达到“欺骗”模型的效果，真正攻击到关键的点上。

那么在网上查阅了一些资料，实际上基本都在本质上就是上面的一种具体实现。个人不太理解的是，这些对抗样本攻击的计算，似乎都是基于一个具体的、训练好的模型。针对这个特殊的模型进行攻击。实际上通过对这个具体模型的不断迭代攻击，我们肯定能找到一些特别特别强的攻击点。这个时候就要考虑到迁移性，如果这些点被放到别的模型（同样的训练集）当中，能否起到攻击作用？这可能是现在要解决的最重要、最难的问题，因为实际上我们很难知道针对于这个同样的训练集，实际的训练方式是怎么样的，也许是普通的分层神经网络，也许是卷积神经网络，层数可能不同，每层的神经元个数可能也不同，在这些都未知的情况之下，如何能够保证可迁移性？我认为这之中的数学原理恐怕远远超出本人的理解范畴。但是，无论如何，我个人认为一个粗略的想法是非常有道理的，那就是，对抗样本针对特定模型的攻击成功率和可迁移性应该是大体上互斥的，更高的攻击成功意味着更好的符合这个模型，同时也就意味着可迁移性的丧失。这是类似于过拟合的情况，太多的迭代计算最终会让结果太贴合于具体而失去泛化能力。

那么基于这样的一种想法，也许”简单的就是最好的“。更简单的模型（指生成对抗样本的模型）也许能具有刚好的泛化能力和足够的攻击成功率（因为成功率）。因此我尝试了一些算法，在后面列出。

具体来说，每一个算法的基本想法，都是对于每一个训练样本，给出一些扰动向量（根据不同的算法，可能是不同的扰动向量的给出方法），然后计算结果的偏差，取偏差最大的，依次进行计算。有一些算法可能是动态的给出扰动向量，也是类似ML模型的训练一样，以特定的步长去试图收敛到偏差最大的点上。

