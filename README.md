# Parsing CAPTCHA Images with a Robust Probabilistic Generative Model

Despite significant progress, contemporary visual foundation models require
training on order-of-millions of data points in order to successfully parse
images into objects, clutter, and background. Dedicated small-data models, on
the other hand, can parse crowded scenes at inference time, after learning from
as few as 10 or 20 labelled object exemplars, but these tend to require either
simplifying assumptions that leave them unable to generate novel data exemplars
or expensive, unstable training of deep neural networks on large data sets.
We present a simple generative model of CAPTCHA images, based on convolutional
sparse coding in the visual cortex, capable of parsing objects from clutter and
background in images, without training corpora or neural networks. We
demonstrate the performance of our model on a body of images generated de novo
from an open-source Python captcha library, inverting our generative model by
maximum-a-posteriori (MAP) estimation. We exploit the MAP estimates to
initialize a variational inference procedure, enabling us to learn the free
parameters endowing the model with robustness to visual style by maximizing a
lower bound on the data log-likelihood and thus acquire visual style; after
training, our model can simulate new image instances in the acquired style, an
advance on the major previous generative probabilistic model for CAPTCHA-parsing.
We document the inductive biases and modeling assumptions enabling small-data
learning in our setting and draw an analogy between visual attention and our
model's mechanism for robustness to unmodelled visual clutter and backgrounds.
