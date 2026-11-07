# References

1. K. He, X. Zhang, S. Ren, and J. Sun. **Deep Residual Learning for Image Recognition.** CVPR, 2016. https://arxiv.org/abs/1512.03385
2. R. Zhang. **Making Convolutional Networks Shift-Invariant Again.** ICML, 2019. https://arxiv.org/abs/1904.11486
3. T. DeVries and G. W. Taylor. **Improved Regularization of Convolutional Neural Networks with Cutout.** 2017. https://arxiv.org/abs/1708.04552
4. N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov. **Dropout: A Simple Way to Prevent Neural Networks from Overfitting.** JMLR 15, 2014, pp. 1929-1958. https://jmlr.org/papers/v15/srivastava14a.html
5. G. Huang, Y. Sun, Z. Liu, D. Sedra, and K. Q. Weinberger. **Deep Networks with Stochastic Depth.** ECCV, 2016. https://arxiv.org/abs/1603.09382
6. I. Loshchilov and F. Hutter. **SGDR: Stochastic Gradient Descent with Warm Restarts.** ICLR, 2017. https://arxiv.org/abs/1608.03983 (This implementation uses cosine decay without restarts.)
7. NYU Tandon. **Deep Learning Project 1, Fall 2026.** Course assignment and [competition rules](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/rules).

The model and training implementation are local project code inspired by these methods. No external implementation package, pretrained weights, external images, or pretrained teacher is used by the submitted training pipeline. The residual-drop implementation masks each sample after residual computation; it does not skip convolutional compute during training.
