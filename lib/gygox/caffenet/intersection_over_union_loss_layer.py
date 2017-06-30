import caffe
import numpy as np


class IntersectionOverUnionLossLayer(caffe.Layer):
    """
    Bottom contains two inputs.
    - bottom[0]: the output from sigmoid, same size as the image
    - bottom[1]: the label image, with the gt segmentation

    to add to prototxt:
    layer {
      type: 'Python'
      name: 'loss'
      bottom: 'sigmoid-fuse'
      bottom: 'label'
      top: 'fuse_loss'
      python_param {
        # the module name -- usually the filename -- that needs to be in $PYTHONPATH
        module: 'gygox.caffenet.intersection_over_union_loss_layer'
        # the layer name -- the class name in the module
        layer: 'IntersectionOverUnionLossLayer'
      }
      # set loss weight so Caffe knows this is a loss layer
      loss_weight: 1
    }
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        """
        Adjust the shapes of top blobs and internal buffers to accommodate the shapes of the bottom blobs.

        This method should reshape top blobs as needed according to the shapes
        of the bottom (input) blobs, as well as reshaping any internal buffers
        and making any other necessary adjustments so that the layer can
        accommodate the bottom blobs.

        :param bottom: the input blobs, with the requested input shapes
        :param top: the top blobs, which should be reshaped as needed
        :return:
        """
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")

        self.intersection = 0.0
        self.union = 0.0
        top[0].reshape(1)

    def forward(self, bottom, top):
        pos_probabilities = bottom[0].data * bottom[1].data
        self.intersection = np.sum(pos_probabilities)
        self.union = np.sum(
            bottom[0].data + bottom[1].data - pos_probabilities)
        top[0].data[...] = 1 - self.intersection / self.union

    def backward(self, top, propagate_down, bottom):
        """
        bottom is the loss and top are the fuse layer activations in this func
        """
        label_is_foreground = bottom[1].data[...]
        label_is_background = 1 - bottom[1].data[...]
        bottom[0].diff[...] = (
            -(1 / self.union) * label_is_foreground + (
                (self.intersection / self.union ** 2) * label_is_background)
        )

        # for i in range(2):
        #     if not propagate_down[i]:
        #         continue
        #     if i == 0:
        #         sign = 1
        #     else:
        #         sign = -1
        #     bottom[i].diff[...] = sign * self.diff / bottom[i].num


def intersection_over_union_loss_layer_unitest():
    """
    For manually debugging the iou_loss_layer code
    """
    import cv2
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    label = cv2.imread('/Users/eddie/Downloads/00024.png', 0) / 255.0
    sigmoid_fuse = cv2.imread('/Users/eddie/Downloads/00024_result.png', 0) / 255.0

    plt.figure()
    plt.imshow(label)
    plt.figure()
    plt.imshow(sigmoid_fuse)

    pos_probabilities = sigmoid_fuse * label
    plt.figure()
    plt.imshow(pos_probabilities)
    intersection = np.sum(pos_probabilities)
    union = np.sum(
        sigmoid_fuse + label - pos_probabilities)
    plt.figure()
    plt.imshow(sigmoid_fuse + label - pos_probabilities)
    iou = 1 - intersection / union
    plt.show()
    pass # breakpoint here and explore the results
