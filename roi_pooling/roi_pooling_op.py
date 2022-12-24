import os.path as osp
from typing import Union, List
import tensorflow as tf


filename = osp.join(osp.dirname(__file__), "roi_pooling.so")
_roi_pooling_module = tf.load_op_library(filename)


def roi_pool(
    bottom_data: tf.types.experimental.TensorLike,
    bottom_rois: tf.types.experimental.TensorLike,
    pooled_height: int,
    pooled_width: int,
    spatial_scale: float = 1.0,
    name=None,
):

    with tf.name_scope(name or "roi_pool"):
        bottom_data_tensor = tf.convert_to_tensor(bottom_data, name="bottom_data")
        bottom_rois_tensor = tf.convert_to_tensor(bottom_rois, name="bottom_rois")

        return _roi_pooling_module.roi_pool(
            bottom_data_tensor,
            bottom_rois_tensor,
            pooled_height,
            pooled_width,
            spatial_scale,
        )


@tf.RegisterGradient("RoiPool")
def _roi_pool_grad(
    op: tf.types.experimental.TensorLike,
    grad_output: tf.types.experimental.TensorLike,
    _,
) -> tf.Tensor:
    data = op.inputs[0]
    rois = op.inputs[1]
    argmax = op.outputs[1]
    pooled_height = op.get_attr("pooled_height")
    pooled_width = op.get_attr("pooled_width")
    spatial_scale = op.get_attr("spatial_scale")

    data_grad = _roi_pool_grad(
        data, rois, argmax, grad_output, pooled_height, pooled_width, spatial_scale
    )

    return [data_grad, None]


tf.no_gradient("RoiPoolGrad")
