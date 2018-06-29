# Two-Step Quantization on AlexNet
This is a demo of [Two-Step Quantization](http://openaccess.thecvf.com/content_cvpr_2018/papers/Wang_Two-Step_Quantization_for_CVPR_2018_paper.pdf).

# Usage:

Copy source files into caffe's directories and then build caffe.

Download [model](https://drive.google.com/file/d/1-kf7mQX5nktUt3Qtg0dq147pdxlje1bt/view?usp=sharing).

./build/tools/caffe test -model test_2_ternary.prototxt -weights caffe_2_ternary.caffemodel -iterations 1000 -gpu 0
