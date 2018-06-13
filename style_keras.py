import numpy as np
import tensorflow as tf
import scipy.misc
from nst_utils import *
import keras.backend as K
from keras.models import Model
from keras.applications.resnet50 import ResNet50
import winsound

'''
The goal of this project is to show how to load an arbitrary Keras model
and use it to stylize an image. The advantage to using Keras is that it provides
a variety of pre trained neural nets and an framework to quickly build and
train custom models.
'''

'''
let's first define some necessary functions
'''

# function to compute the content cost
# ie how similar a layer is to a
def compute_content_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- the content cost
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G
    a_C_unrolled = K.reshape(a_C, [n_H*n_W, n_C])
    a_G_unrolled = K.reshape(a_G, [n_H*n_W, n_C])
    # compute the cost with tensorflow
    J_content = 1/(4*n_H*n_W*n_C)*K.sum(K.square(a_C_unrolled - a_G_unrolled))

    return J_content

# computes the gram matrix, in essence a style matrix
# as it contains non localized information about the image
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """

    GA = K.dot(A, K.transpose(A))

    return GA

# style cost for one layer
def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, the style cost
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape the images to have them of shape (n_C, n_H*n_W)
    a_S = K.transpose(K.reshape(a_S, [n_H*n_W,n_C]))
    a_G = K.transpose(K.reshape(a_G, [n_H*n_W,n_C]))

    # Computing gram_matrices for both images S and G
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss
    J_style_layer = 1/(4*n_C**2*(n_H*n_W)**2)*K.sum(K.square(GG-GS))

    return J_style_layer

# total style cost
def compute_style_cost(STYLE_LAYERS, STYLE_COMPARISON, STYLE_WEIGHTS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    STYLE_LAYERS -- A python list containing the models of the
                        layers we would like to match the style from
    STYLE_COMPARISON -- A python list containing the evaluated layers of the style
                            image we would like to match to
    STYLE_WEIGHTS -- Weights for the importance of each layer

    Returns:
    J_style -- tensor representing a scalar value, style cost
    """

    # initialize the overall style cost
    J_style = 0

    for i in range(len(STYLE_LAYERS)):

        # Select the output tensor of the currently selected layer
        layer = STYLE_LAYERS[i]
        comparison = STYLE_COMPARISON[i]
        weight = STYLE_WEIGHTS[i]

        # Set a_S to be the hidden layer activation from the style image
        # from the layer we have selected
        a_S = comparison

        # Set a_G to be the hidden layer activation output from same layer.
        # Here the output isn't evaluated but will be when we run the model
        # after assigning G, the generated image to be the input
        a_G = layer.output

        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += weight * J_style_layer

    return J_style

# total cost
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """

    J = alpha*J_content + beta*J_style

    return J

# Adam optimization function
def adam(sess, image, grads, costs, steps = 200, lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, print_steps = 20):
    '''
    Adam optimization to minimize the image cost while outputting the cost
    every so many steps (print_steps)
    based on: https://arxiv.org/abs/1412.6980 (arXiv:1412.6980)
    by Diederik P. Kingma and Jimmy Ba
    '''
    [J, J_content, J_style] = costs
    m = 0
    v = 0
    t = 0
    for t in range(steps):
        t += 1
        g = sess.run(grads, feed_dict={input_tens: image})[0]
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        mhat = m / (1 - beta1**t)
        vhat = v / (1 - beta2**t)
        image = image - lr * mhat / (np.sqrt(vhat) + eps)
        if(t % print_steps == 0):
            [loss, content_loss, style_loss] = sess.run([J, J_content, J_style], feed_dict={input_tens: image})
            print('Loss '+str(t)+': '+str(loss))
            print('Content Loss '+str(t)+': '+str(content_loss))
            print('Style Loss '+str(t)+': '+str(style_loss))
    return image

'''

start of script

'''

# image locations
image_folder = 'input'
style_name = 'style.jpg'
content_name = 'content.jpg'
output_folder = 'output'

# get images
content_image = scipy.misc.imread(image_folder + "/" + content_name)
content_image = reshape_and_normalize_image(content_image)
style_image = scipy.misc.imread(image_folder + "/" + style_name)
style_image = reshape_and_normalize_image(style_image)
generated_image = generate_noise_image(content_image)
imshow(generated_image[0])
save_image("gen0.jpg", generated_image)

# get base model
model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(224,224,3), pooling=None, classes=1000)

# print model to look  at what layers we could use for the style layers
# or possibly the content layer if you decide not to use the last layer
# of ResNet50 for content comparison
model.summary()
input('continue...')

# select the layers of the model we want to stylize the image
STYLE_LAYER_NAMES = [
    'activation_1',
    'activation_2',
    'activation_3',
    'activation_4',
    'activation_5',
    'activation_6',
    'activation_7',
    'activation_8',
    'activation_9',
    'activation_10',
    'activation_11',
    'activation_12',
    'activation_13',
    'activation_14',
    'activation_15',
    'activation_16',
    'activation_17',
    'activation_18',
    'activation_19',
    'activation_20',
    'activation_21',
    'activation_22',
    'activation_23',
    'activation_24',
    'activation_25',
    'activation_26',
    'activation_27',
    'activation_28',
    'activation_29',
    'activation_30',
    'activation_31',
    'activation_32',
    'activation_33',
    'activation_34',
    'activation_35'
]

# list the weights we want to use for each layer, these will be normalized later
STYLE_LAYER_WEIGHTS = [
    10,
    10,
    10,
    10,
    10,
    20,
    20,
    20,
    20,
    20,
    30,
    30,
    30,
    30,
    30,
    40,
    40,
    40,
    40,
    40,
    50,
    50,
    50,
    50,
    50,
    40,
    40,
    40,
    40,
    40,
    30,
    30,
    30,
    30,
    30
]

# normalize the style layer weights so the can be tweaked independently of the
# relative weights of all style layer costs and the content cost
STYLE_LAYER_WEIGHTS = np.array(STYLE_LAYER_WEIGHTS)
STYLE_LAYER_WEIGHTS = STYLE_LAYER_WEIGHTS / np.sum(STYLE_LAYER_WEIGHTS)

# get style layers we want to evaluate for the generated image
STYLE_LAYERS = []
for i in range(len(STYLE_LAYER_NAMES)):
    STYLE_LAYERS.append(Model(inputs=model.input, outputs=model.get_layer(STYLE_LAYER_NAMES[i]).output))

# get style layers evaluated for the style image
STYLE_COMPARISON = []
for i in range(len(STYLE_LAYERS)):
    STYLE_COMPARISON.append(K.constant(STYLE_LAYERS[i].predict(style_image)))

# get content layer evaluation from the content image for evaluation
CONTENT_COMPARISON = K.constant(model.predict(content_image))

# setup cost tensors
J_style = compute_style_cost(STYLE_LAYERS, STYLE_COMPARISON, STYLE_LAYER_WEIGHTS)
J_content = compute_content_cost(CONTENT_COMPARISON, model.output)
J = total_cost(J_content, J_style, alpha = 10, beta = 500000)
costs = [J, J_content, J_style]

# get image input tensor
input_tens = model.get_input_at(0)
# get the gradient tensor for Adam
grads = K.gradients(J, input_tens)
# get the Keras session which will be used to evaluate gradient and cost tensors
sess = K.get_session()

# run the actual optimization / style transfer
generated_image = adam(sess, generated_image, grads, costs, lr = 2)

# save the output image
save_image(output_folder + '/generated_image.jpg', generated_image)

# this just plays a not too intrusive sound when done, delete or comment
# out if you would not like this
winsound.PlaySound('SystemExclamation', winsound.SND_ALIAS)
