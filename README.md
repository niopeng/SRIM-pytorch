# SRIM-pytorch

Setup:

1. go to "/code/dci" and then "make"

2. go to "/code" and then do:

    i. "ln -s ./dci/src/dci.py ./"

    ii. "ln -s ./dci/build/_dci.so ./"

Training Procedure:

1. Train a vanilla network without code input, the upsampling layer uses nearest neighbour interpolation, ouput layer uses recentering tanh activation.

2. Fine tune the network with random code input (when loading the weights, set "load_partial_cat" to "True"), the upsampling layer uses nearest neighbour interpolation, output layer uses sigmoid activation.

3. Further fine tune the network (set "load_partial_cat" to "False"), use bilinear interpolation for the upsampling layer and recentering tanh activation for the output layer.

