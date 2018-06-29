#include <algorithm>
#include <vector>

#include "caffe/layers/relu2b_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void ReLU2BForward(const int n, const Dtype* in, Dtype* out, Dtype th, Dtype d) {
  CUDA_KERNEL_LOOP(i, n) {
    if (in[i] <= th) {
      out[i] = Dtype(0);
    } else if (in[i] <= 1.5*d) {
      out[i] = Dtype(d);
    } else if (in[i] <= 2.5*d) {
      out[i] = Dtype(2*d);
    } else {
      out[i] = Dtype(3*d);
    }
    //out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

template <typename Dtype>
void ReLU2BLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ReLUParameter relu_param = this->layer_param().relu_param();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  const Dtype th = relu_param.thresh();
  const Dtype d = relu_param.delta();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ReLU2BForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, th, d);
  CUDA_POST_KERNEL_CHECK;
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data
  //     << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ReLU2BBackward(const int n, const Dtype* in_data, const Dtype* in_diff, Dtype* out_diff, const Dtype d) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > 0) * (in_data[index] < 3*d));
  }
}

template <typename Dtype>
void ReLU2BLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  ReLUParameter relu_param = this->layer_param().relu_param();
  const Dtype d = relu_param.delta();
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    // NOLINT_NEXT_LINE(whitespace/operators)
    ReLU2BBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, top_diff, bottom_diff, d);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(ReLU2BLayer);


}  // namespace caffe
