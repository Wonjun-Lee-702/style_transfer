#original code from
#https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py
#https://github.com/hunter-heidenreich/ML-Open-Source-Implementations

'''
# References
    - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
    - Image Style Transfer Using Convolutional Neural Networks, Leon A. Gatys, Alexander S. Ecker, Matthias Bethge
'''

import argparse

import numpy as np
from PIL import Image
import imageio



import tensorflow as tf
import tensorflow.contrib.eager as tfe

from tensorflow.python.keras.preprocessing import image as kp_image
from tensorflow.python.keras import models
from tensorflow.python.keras import losses
from tensorflow.python.keras import layers
from tensorflow.python.keras import backend as K

parser = argparse.ArgumentParser(description='Image neural style transfer implemented with Keras')
parser.add_argument('content_img', metavar='content', type=str, help='Path to target content image')
parser.add_argument('style_img', metavar='style', type=str, help='Path to target style image')
parser.add_argument('result_img_prefix', metavar='res_prefix', type=str, help='Name of generated image')
parser.add_argument('--iter', type=int, default=1000, required=False, help='Number of iterations to run')
parser.add_argument('--content_weight', type=float, default=0.025, required=False, help='Content weight')
parser.add_argument('--style_weight', type=float, default=1.0, required=False, help='Style weight')
parser.add_argument('--var_weight', type=float, default=1.0, required=False, help='Total Variation weight')
parser.add_argument('--max_size', type=int, default=512, required=False, help='max width or height of the image')
#parser.add_argument('--width', type=int, default=512, required=False, help='Width of the images')

args = parser.parse_args()

content_path = args.content_img
style_path = args.style_img
target_path = args.result_img_prefix
target_extension = '.png'
max_size = args.max_size
content_weight = args.content_weight
style_weight = args.style_weight
img_channels = 3
iterations = args.iter
img_height = 0
img_width = 0



#enable eager exectution
tf.compat.v1.enable_eager_execution()
def load_img(path_to_img):
  max_dim = max_size
  img = Image.open(path_to_img)
  long = max(img.size)
  scale = max_dim/long
  img_height = img.size[0]*scale
  img_weight = img.size[1]*scale
  img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

  img = kp_image.img_to_array(img)

  # We need to broadcast the image array such that it has a batch dimension
  img = np.expand_dims(img, axis=0)
  return img


#process img
def process_img(img_path):
    img = load_img(img_path)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

#deprocess img
def deprocess_img(img):
    x = img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x,0)
    assert len(x.shape) == 3, ("image is not a 3D array")

    if len(x.shape) != 3:
        raise ValueError("image is not a 3D array")

    #perform the inverse of the perprocessing step
    x[:,:,0] += 103.939
    x[:,:,1] += 116.779
    x[:,:,2] += 123.68

    #BGR to RGB
    x = x[:,:,::-1]
    x = np.clip(x, 0, 255).astype("uint8")
    return x

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']

content_layers = ['block5_conv2']

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)


def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top = False, weights = 'imagenet')
    vgg.trainable = False
    #get VGG19 layers
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs

    return models.Model(vgg.input, model_outputs)

def get_content_loss(content, target):
    return 0.5 * tf.reduce_mean(tf.square(content - target))

def gram_matrix(input_tensor):
    channels = int(input_tensor.shape[-1])
    a = tf.reshape(input_tensor, [-1, channels])
    n = tf.shape(a)[0]
    gram = tf.matmul(a,a,transpose_a = True)
    return gram / tf.cast(n,tf.float32)

def get_style_loss(base_style, gram_target):
    h, w, channels = base_style.get_shape().as_list()
    gram_style = gram_matrix(base_style)

    return tf.reduce_mean(tf.square(gram_style - gram_target)) / (4. * (channels**2) * (w + h)**2)

def compute_grads(cfg):
  with tf.GradientTape() as tape:
    all_loss = compute_loss(**cfg)
  # Compute gradients wrt input image
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
  """This function will compute the loss total loss.

  Arguments:
    model: The model that will give us access to the intermediate layers
    loss_weights: The weights of each contribution of each loss function.
      (style weight, content weight, and total variation weight)
    init_image: Our initial base image. This image is what we are updating with
      our optimization process. We apply the gradients wrt the loss we are
      calculating to this image.
    gram_style_features: Precomputed gram matrices corresponding to the
      defined style layers of interest.
    content_features: Precomputed outputs from defined content layers of
      interest.

  Returns:
    returns the total loss, style loss, content loss, and total variational loss
  """
  style_weight, content_weight = loss_weights

  # Feed our init image through our model. This will give us the content and
  # style representations at our desired layers. Since we're using eager
  # our model is callable just like any other function!
  model_outputs = model(init_image)

  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]

  style_score = 0
  content_score = 0

  # Accumulate style losses from all layers
  # Here, we equally weight each contribution of each loss layer
  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)

  # Accumulate content losses from all layers
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, content_output_features):
    content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)

  style_score *= style_weight
  content_score *= content_weight

  # Get total loss
  loss = style_score + content_score
  return loss, style_score, content_score



def get_feature_representations(model, content_path, style_path):
  """Helper function to compute our content and style feature representations.

  This function will simply load and preprocess both the content and style
  images from their path. Then it will feed them through the network to obtain
  the outputs of the intermediate layers.

  Arguments:
    model: The model that we are using.
    content_path: The path to the content image.
    style_path: The path to the style image

  Returns:
    returns the style features and the content features.
  """
  # Load our images in
  content_image = process_img(content_path)
  style_image = process_img(style_path)

  # batch compute content and style features
  style_outputs = model(style_image)
  content_outputs = model(content_image)


  # Get the style and content feature representations from our model
  style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
  content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
  return style_features, content_features





if __name__ == '__main__':


    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(model, content_path, style_path)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    # Set initial image
    init_image = process_img(content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)

    readme = open('Readme.txt', 'w')
    l = ['height: {}\n'.format(img_height), 'weight: {}\n'.format(img_width),
         'content: {}\n'.format(content_path), 'style: {}\n'.format(style_path),
         'content_weight: {}\n'.format(args.content_weight), 'style_weight: {}\n'.format(args.style_weight),
         'variation_weight: {}\n'.format(args.var_weight)]
    readme.writelines(l)
    readme.close()

    # Create our optimizer
    opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
    best_loss, best_img = float('inf'), None
    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features
    }



    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means


    for i in range(iterations + 1):
        grads, all_loss = compute_grads(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        print("iteration: ", i)
        print("content_score: ", content_score)
        print("style_score: ", style_score)
        print("loss: ", loss)

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = deprocess_img(init_image.numpy())

        if i % 100 == 0:
            name = '{}-{}{}'.format(target_path, i, target_extension)
            imageio.imwrite(name,np.asarray(best_img))
