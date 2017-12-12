### CS 744 Final Project
#### Greedy approach to maximizing gradient diversity for minibatch SGD

Minibatch SGD is a compromise between one-sample SGD and batch SGD in terms of stability, accuracy, and efficiency. However, minibatch SGD struggles to scale to beyond batch size in the tens before sacrificing model performance, with batch size inversely correlated with accuracy and generalization. Yin et al. show that batch size is bounded by gradient diversity, which affects convergence and stability. They propose that gradient diversity can explain the effectiveness of diversity-increasing heuristics, e.g. dropout [1]. 

This project sets out to review a couple of ways to tackle the scaling problem with the key insights from Yin et al. First we try simple methods like dropout and toggling between different batch size to see their effect on convergence and generalization. With the aim of increasing viable batch size, one could try to maximize gradient diversity when sampling for each batch. We attempt to greedily select for gradients that would maintain or increase gradient diversity while each example is processed in a batch.

The same dataset used for the Kaggle Display Advertising Challenge sponsored by Criteo Labs was used for this distributed logistic regression task. All implementation was done in TensorFlow 1.4.0, CPU version, with Python 2.7 back-end. The following are the different algorithms and experiments attempted.


##### Vanilla minibatch SGD

    python batch.py --batch_size=256


##### Dropout minibatch SGD

    python frankenstein.py --batch_size=256 --dropout_rate=0.5


##### Greedy diversity tracker minibatch SGD
Within each batch, a gradient for an example is calculated, and is only added to the current batch’s gradient if it would maintain or improve on the running gradient diversity. The batch size is capped at some large value (e.g. 16384) and this filtering mechanism would only pick up the subset of examples and their gradients that would help the overall cause. 

    python divesity.py --batch_size=256

Disappointingly, this method failed to add any (!) examples at all after the first couple of iterations, which unfortunately prevents the model from improving since no gradients were being applied. In order to investigate the cause of this issue, the vanilla minibatch SGD was re-run, this time tracking the gradient diversity as well as the validation set error. It was revealed that a faulty assumption was made when the greedy approach was attempted. The gradient diversity, as gradients are added, is monotonously increasing (in smaller batch size = 32) or decreasing (in larger batch sizes). The assumption of a fluctuating or stochastic gradient diversity was invalid, and cannot be taken advantage of by a greedy approach. 

[1] Yin, Dong, et al. "Gradient Diversity Empowers Distributed Learning." arXiv preprint arXiv:1706.05699 (2017).
