#include <algorithm>
#include <vector>

#include "caffe/layers/relu2b_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLU2BLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ReLUParameter relu_param = this->layer_param().relu_param();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  const Dtype th = relu_param.thresh();
  const Dtype d = relu_param.delta();
  for (int i = 0; i < count; ++i) {
    if (bottom_data[i] <= th) {
      top_data[i] = Dtype(0);
    } else if (bottom_data[i] <= 1.5*d) {
      top_data[i] = Dtype(d);
    } else if (bottom_data[i] <= 2.5*d) {
      top_data[i] = Dtype(2*d);
    } else {
      top_data[i] = Dtype(3*d);
    }
    //top_data[i] = std::max(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
void ReLU2BLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  ReLUParameter relu_param = this->layer_param().relu_param();
  const Dtype d = relu_param.delta();
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    for (int i = 0; i < count; ++i) {
       bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0) * (bottom_data[i] < 3*d));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ReLU2BLayer);
#endif

INSTANTIATE_CLASS(ReLU2BLayer);

}  // namespace caffe
