### SqueezeNet-DSD-Training

This repo demos Dense-Sparse-Dense(DSD) training methodology.

The repo contains the DSD-SqueezeNet caffemodel, which is obtained by applying Dense-Sparse-Dense training methodology to SqueezeNet v1.0. DSD training methodology improves the top-1 accuracy of SqueezeNet by 4.3% on ImageNet without changing the model architecture and model size. 

DSD training is an interesting byproduct of network pruning: re-densifying and retraining from a sparse model can improve the accuracy. That is, compared to a dense CNN baseline, dense→sparse→dense (DSD) training yielded higher accuracy. 

We now explain our DSD training strategy. On top of the sparse SqueezeNet (pruned 3x), we let the killed weights recover, initializing them from zero. We let the survived weights keeping their value. We retrained the whole network using learning rate of 1e − 4. After 20 epochs of training, we observed that the top-1 ImageNet accuracy improved by 4.3 percentage-points; 

Sparsity is a powerful form of regularization. Our intuition is that, once the network arrives at a local minimum given the sparsity constraint, relaxing the constraint gives the network more freedom to escape the saddle point and arrive at a higher-accuracy local minimum. So far, we trained in just three stages of density (dense→sparse→dense), but regularizing models by intermittently pruning parameters10 throughout training would be an interesting area of future work.


### Usage:

    $CAFFE_ROOT/build/tools/caffe test --model=trainval.prototxt --weights=DSD_SqueezeNet_top1_0.617579_top5_0.834742.caffemodel --iterations=1000 --gpu 0

### Result:
      
    I0421 13:58:46.246104  5184 caffe.cpp:293] accuracy_top1 = 0.617579
    I0421 13:58:46.246115  5184 caffe.cpp:293] accuracy_top5 = 0.834742
    I0421 13:58:46.246126  5184 caffe.cpp:293] loss = 1.7059 (* 1 = 1.7059 loss)    

    
# Related SqueezeNet repo and paper:
[SqueezeNet](https://github.com/DeepScale/SqueezeNet)

[SqueezeNet-Deep-Compression](https://github.com/songhan/SqueezeNet-Deep-Compression)

[SqueezeNet-Generator](https://github.com/songhan/SqueezeNet-Generator)

[SqueezeNet-DSD-Training](https://github.com/songhan/SqueezeNet-DSD-Training)

[SqueezeNet-Residual](https://github.com/songhan/SqueezeNet-Residual)


If you find SqueezeNet, DSD training, network pruning and Deep Compression useful in your research, please consider citing the paper:

    @article{SqueezeNet,
      title={SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5MB model size},
      author={Iandola, Forrest N and Han, Song and Moskewicz, Matthew W and Ashraf, Khalid and Dally, William J and Keutzer, Kurt},
      journal={arXiv preprint arXiv:1602.07360},
      year={2016}
    }

    @article{DeepCompression,
      title={Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding},
      author={Han, Song and Mao, Huizi and Dally, William J},
      journal={International Conference on Learning Representations (ICLR)},
      year={2016}
    }

    @inproceedings{han2015learning,
      title={Learning both Weights and Connections for Efficient Neural Network},
      author={Han, Song and Pool, Jeff and Tran, John and Dally, William},
      booktitle={Advances in Neural Information Processing Systems (NIPS)},
      pages={1135--1143},
      year={2015}
    }

    @article{han2016eie,
      title={EIE: Efficient Inference Engine on Compressed Deep Neural Network},
      author={Han, Song and Liu, Xingyu and Mao, Huizi and Pu, Jing and Pedram, Ardavan and Horowitz, Mark A and Dally, William J},
      journal={International Conference on Computer Architecture (ISCA)},
      year={2016}
    }

