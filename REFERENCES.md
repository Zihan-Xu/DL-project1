# References

Citation numbers match the eight-page report. The development studies also tested methods that were not selected for the submitted model.

1. K. He, X. Zhang, S. Ren, and J. Sun. Deep Residual Learning for Image Recognition. CVPR, 2016. [Source](https://arxiv.org/abs/1512.03385).
2. S. Zagoruyko and N. Komodakis. Wide Residual Networks. BMVC, 2016. [Source](https://arxiv.org/abs/1605.07146).
3. C. Rother, V. Kolmogorov, and A. Blake. GrabCut: Interactive Foreground Extraction Using Iterated Graph Cuts. ACM TOG, 23(3), 2004. [Source](https://www.microsoft.com/en-us/research/publication/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts/).
4. P. Khosla et al. Supervised Contrastive Learning. NeurIPS, 2020. [Source](https://arxiv.org/abs/2004.11362).
5. R. Zhang. Making Convolutional Networks Shift-Invariant Again. ICML, 2019. [Source](https://arxiv.org/abs/1904.11486).
6. T. DeVries and G. W. Taylor. Improved Regularization of Convolutional Neural Networks with Cutout. 2017. [Source](https://arxiv.org/abs/1708.04552).
7. N. Srivastava et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. JMLR, 15:1929-1958, 2014. [Source](https://jmlr.org/papers/v15/srivastava14a.html).
8. G. Huang et al. Deep Networks with Stochastic Depth. ECCV, 2016. [Source](https://arxiv.org/abs/1603.09382).
9. P. Foret et al. Sharpness-Aware Minimization for Efficiently Improving Generalization. ICLR, 2021. [Source](https://arxiv.org/abs/2010.01412).
10. P. Izmailov et al. Averaging Weights Leads to Wider Optima and Better Generalization. UAI, 2018. [Source](https://arxiv.org/abs/1803.05407).
11. G. Hinton, O. Vinyals, and J. Dean. Distilling the Knowledge in a Neural Network. 2015. [Source](https://arxiv.org/abs/1503.02531).
12. I. Loshchilov and F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR, 2017. The submitted recipe uses cosine decay without restarts. [Source](https://arxiv.org/abs/1608.03983).
13. NYU Tandon. Deep Learning Project 1, Fall 2026. Course assignment and competition rules. [Source](https://www.kaggle.com/competitions/cifar-10-res-net-architecture-challenge-dl/rules).

The submitted model and training code implement the selected methods locally. No pretrained model, pretrained teacher or external training images are used. Distillation and other rejected alternatives are documented as development experiments, not as parts of the final pipeline.

The cosine schedule has no warm restarts. Residual dropping masks each sample after residual computation, so it does not reduce training convolutional work.
