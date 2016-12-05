#ifndef CAFFE_DETECTION_LAYER_HPP_
#define CAFFE_DETECTION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"
namespace caffe {
template <typename Dtype>
class DetectionLayer : public LossLayer<Dtype> {
 public:
  explicit DetectionLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "Detection"; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
 
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  /// The internal SoftmaxLayer used to map predictions to a distribution.
  shared_ptr<Layer<Dtype> > softmax_layer_;
  /// prob stores the output probability predictions from the SoftmaxLayer.
  Blob<Dtype> prob_;
  /// bottom vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  /// top vector holder used in call to the underlying SoftmaxLayer::Forward
  vector<Blob<Dtype>*> softmax_top_vec_;
  
  Blob<Dtype> cls_bottom_;
  Blob<Dtype> coord_bottom_;
  Blob<Dtype> coord_diff_;
  Blob<Dtype> bootstrap_map_;
  Blob<Dtype> weight_map_;
  /// Whether to ignore instances with a certain label.
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  /// Whether to normalize the loss by the total number of values present
  /// (otherwise just by the batch size).
  bool normalize_;
  
  int cls_num_;
  int coord_num_;
  float lambda_;
  int field_h_;
  int field_w_;
  float field_whr_;
  float field_xyr_;
  int downsample_rate_;
  bool bb_smooth_;
  float bg_threshold_;
  int bg_multiple_;
  string sample_mode_;
  bool objectness_;
  bool iou_weighted_;
  bool pos_neg_weighted_;
};

}  // namespace caffe

#endif  // CAFFE_DETECTION_LAYER_HPP_
