from flax import nnx
from flax.nnx.nn.linear import canonicalize_padding
import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float
import numpyro
import numpyro.distributions as dist
from typing import Optional, Sequence

from src.data.dictionary import ShapeDictionary
from src import utils

def conv_transpose_local(lhs: Array, rhs: Array, strides: Sequence[int],
                         padding: str | Sequence[tuple[int, int]],
                         rhs_dilation: Sequence[int] | None = None,
                         dimension_numbers: jax.lax.ConvGeneralDilatedDimensionNumbers = None,
                         transpose_kernel: bool = False,
                         precision: jax.lax.PrecisionLike = None,
                         use_consistent_padding: bool = False) -> Array:
    """Convenience wrapper for calculating the unshared N-d convolution "transpose".

    This function directly calculates a fractionally strided conv rather than
    indirectly calculating the gradient (transpose) of a forward convolution.

    Notes:
    TensorFlow/Keras Compatibility: By default, JAX does NOT reverse the
    kernel's spatial dimensions. This differs from TensorFlow's "Conv2DTranspose"
    and similar frameworks, which flip spatial axes and swap input/output channels.

    To match TensorFlow/Keras behavior, set "transpose_kernel=True" .

    Args:
    lhs: a rank `n+2` dimensional input array.
    rhs: a rank `n+2` dimensional array of kernel weights.
    strides: sequence of `n` integers, sets fractional stride.
    padding: 'SAME', 'VALID', or a sequence of `n` integer 2-tuples describing before-and-after
      padding for each spatial dimension. If `use_consistent_padding=True`, this is interpreted
      as the padding of the corresponding forward conv, which effectively adds
      `dilation * (kernel_size - 1) - padding` zero padding to each side
      of the input so that `conv_transpose` becomes the gradient of `conv` when given the same padding
      and stride arguments. This is the behavior in PyTorch. If `use_consistent_padding=False`,
      the 'SAME' and 'VALID' strings are interpreted as the padding of the corresponding forward conv,
      but integer tuples are interpreted as padding for the transposed convolution.
    rhs_dilation: `None`, or a sequence of `n` integers, giving the
      dilation factor to apply in each spatial dimension of `rhs`. RHS dilation
      is also known as atrous convolution.
    dimension_numbers: tuple of dimension descriptors as in
      lax.conv_general_dilated. Defaults to tensorflow convention.
    transpose_kernel: if True flips spatial axes and swaps the input/output
      channel axes of the kernel. This makes the output of this function identical
      to the gradient-derived functions like keras.layers.Conv2DTranspose
      applied to the same kernel. For typical use in neural nets this is completely
      pointless and just makes input/output channel specification confusing.
    precision: Optional. Either ``None``, which means the default precision for
      the backend, a :class:`~jax.lax.Precision` enum value (``Precision.DEFAULT``,
      ``Precision.HIGH`` or ``Precision.HIGHEST``) or a tuple of two
      :class:`~jax.lax.Precision` enums indicating precision of ``lhs``` and ``rhs``.
    preferred_element_type: Optional. Either ``None``, which means the default
      accumulation type for the input types, or a datatype, indicating to
      accumulate results to and return a result with that datatype.
    use_consistent_padding : In older versions of jax, the `padding` argument was interpreted differently
      depending on whether it was a string or a sequence of integers. Strings were interpreted as padding
      for the forward convolution, while integers were interpreted as padding for the transposed convolution.
      If `use_consistent_padding` is False, this inconsistent behavior is preserved for backwards compatibility.
    Returns:
    Transposed N-d convolution, with output padding following the conventions of
    keras.layers.Conv2DTranspose.
    """
    from jax._src import core
    from jax._src.lax import convolution
    import numpy as np

    assert len(lhs.shape) == len(rhs.shape) and len(lhs.shape) >= 2
    ndims = len(lhs.shape)
    one = (1,) * (ndims - 2)
    # Set dimensional layout defaults if not specified.
    if dimension_numbers is None:
        if ndims == 2:
            dimension_numbers = ('NC', 'IO', 'NC')
        elif ndims == 3:
            dimension_numbers = ('NHC', 'HIO', 'NHC')
        elif ndims == 4:
            dimension_numbers = ('NHWC', 'HWIO', 'NHWC')
        elif ndims == 5:
            dimension_numbers = ('NHWDC', 'HWDIO', 'NHWDC')
        else:
            raise ValueError('No 4+ dimensional dimension_number defaults.')
    dn = jax.lax.conv_dimension_numbers(lhs.shape, rhs.shape, dimension_numbers)
    k_shape = np.take(rhs.shape, dn.rhs_spec)
    k_sdims = k_shape[2:]
    # Calculate correct output shape given padding and strides.
    if rhs_dilation is None:
        rhs_dilation = (1,) * (rhs.ndim - 2)
    pads: str | Sequence[tuple[int, int]]
    if use_consistent_padding or (isinstance(padding, str) and padding in {'SAME', 'VALID'}):
        effective_k_size = map(lambda k, r: core.dilate_dim(k, r), k_sdims, rhs_dilation)
        replicated_padding = [padding] * len(strides) if isinstance(padding, str) else padding
        pads = tuple(convolution._conv_transpose_padding(k, s, p)
                     for k,s,p in zip(effective_k_size, strides,
                                      replicated_padding))
    else:
        pads = padding
    if transpose_kernel:
        # flip spatial dims and swap input / output channel axes
        rhs = convolution._flip_axes(rhs, np.array(dn.rhs_spec)[2:])
        rhs = rhs.swapaxes(dn.rhs_spec[0], dn.rhs_spec[1])

    import pdb; pdb.set_trace()
    spatial_shape = tuple(lhs.shape[d] for d in dn.lhs_spec[2:])
    rhs = jax.lax.collapse(rhs, 0, dn.rhs_spec[1] + 1)
    # rhs = jnp.broadcast_to(rhs[jnp.newaxis, jnp.newaxis, ...],
    #                        spatial_shape + rhs.shape)
    return jax.lax.conv_general_dilated_local(
    # return conv_general_dilated(
        lhs, rhs, one, pads, strides, rhs_dilation, dn, precision=precision,
    )

class PVaePrior(nnx.Module):
    def __init__(self, shape, *, rngs: nnx.Rngs):
        self.u = nnx.Param(rngs.uniform(shape=shape, minval=-10., maxval=-9.))

    def __call__(self, rngs=None):
        return dist.Poisson(jnp.exp(self.u)).to_event(2)

class PlacementsPrior(nnx.Module):
    def __init__(self, kw: int=40, kh: int=40, img_w: int=160, img_h: int=60,
                 num_features: int=36, stride: int=1, *, rngs: nnx.Rngs):
        height, width = (img_h - kh) // stride + 1, (img_w - kw) // stride + 1
        self.topography = PVaePrior(shape=(num_features, height, width),
                                    rngs=rngs)

    def __call__(self, rngs=None):
        return self.topography(rngs=rngs)

class ExternalKernelConvTranspose(nnx.ConvTranspose):
    def __init__(self, *args, **kwargs):
        kwargs["use_bias"] = False
        super().__init__(*args, **kwargs)
        self.kernel = nnx.data(None)

    def __call__(
        self,
        inputs: Float[Array, "*batch height width in_features"],
        kernel: Float[Array, "kernel_height kernel_width in_features out_features"],
    ) -> Float[Array, "*batch out_height out_width out_features"]:
        def maybe_broadcast(x):
            if x is None:
                x = 1
            if isinstance(x, int):
                return (x,) * len(self.kernel_size)
            return tuple(x)

        num_batch_dimensions = inputs.ndim - (len(self.kernel_size) + 1)
        if num_batch_dimensions != 1:
            input_batch_shape = inputs.shape[:num_batch_dimensions]
            inputs = jnp.reshape(inputs, (-1,) +\
                     inputs.shape[num_batch_dimensions:])

        strides = maybe_broadcast(self.strides)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)

        if self.mask is not None:
            kernel = kernel * self.mask

        padding_lax = canonicalize_padding(self.padding, len(self.kernel_size))
        if padding_lax == 'CIRCULAR':
            padding_lax = 'VALID'

        inputs, kernel = self.promote_dtype((inputs, kernel), dtype=self.dtype)

        y = conv_transpose_local(
            inputs, kernel, strides, padding_lax,
            rhs_dilation=kernel_dilation,
            transpose_kernel=self.transpose_kernel,
            precision=self.precision,
        )

        if self.padding == 'CIRCULAR':
            scaled_x_dims = [x * s for x, s in zip(jnp.shape(inputs)[1:-1],
                                                   strides)]
            size_diffs = [-(y_dim - x_dim) % (2 * x_dim)
                          for y_dim, x_dim in zip(y.shape[1:-1], scaled_x_dims)]
            if self.transpose_kernel:
                pad_fn = lambda d: (d // 2, (d + 1) // 2)
            else:
                pad_fn = lambda d: ((d + 1) // 2, d // 2)
            y = jnp.pad(y, [(0, 0)] + [pad_fn(d) for d in size_diffs] +\
                           [(0, 0)])
            for i in range(1, y.ndim - 1):
                y = y.reshape(y.shape[:i] + (-1, scaled_x_dims[i - 1]) +\
                    y.shape[i + 1:])
                y = y.sum(axis=i)

        if num_batch_dimensions != 1:
            y = jnp.reshape(y, input_batch_shape + y.shape[1:])

        return y

class ShapeConvTranspose(nnx.Module):
    def __init__(self, shape_dict: ShapeDictionary, *, rngs: nnx.Rngs,
                 **kwargs):
        channels = shape_dict.shapes.shape[1]
        kernel_size = shape_dict.shapes.shape[-2:]

        self.deconv = ExternalKernelConvTranspose(in_features=1,
                                                  out_features=channels,
                                                  kernel_size=kernel_size,
                                                  **kwargs, rngs=rngs)
        self.shape_dict = shape_dict

    def __call__(self, activations: Array, rngs=None):
        deconv = jax.vmap(self.deconv, in_axes=0, out_axes=0)
        return deconv(activations[..., jnp.newaxis],
                      self.shape_dict.shapes[..., jnp.newaxis])

class ShapePlacements(nnx.Module):
    def __init__(self, prior: PlacementsPrior, shaper: ShapeConvTranspose,
                 num_hiddens=64, *, rngs: nnx.Rngs):
        num_shapes = len(prior.topography.u)
        self.prior = prior
        self.shaper = shaper
        assert len(self.shaper.shape_dict.shapes) == num_shapes

    def __call__(self, rngs=None):
        wheres = numpyro.sample("what_x_where", self.prior(rngs=rngs))
        layers = self.shaper(wheres)
        spikes = wheres.sum(axis=(-2, -1), keepdims=True)
        spikes = spikes / (spikes.sum(axis=0, keepdims=True) + 1)
        return (layers * spikes[..., jnp.newaxis]).sum(axis=0)

class BackgroundDecoder(nnx.Module):
    def __init__(self, embedding_dim: int=50, height=60, hiddens=400, width=160,
                 *, rngs: nnx.Rngs):
        self.bg_shape = (height, width)
        self.decoder = nnx.Sequential(
            nnx.Linear(embedding_dim, hiddens, rngs=rngs), nnx.silu,
            nnx.Linear(hiddens, height * width, rngs=rngs), nnx.sigmoid
        )
        self.embedding_dim = embedding_dim

    def __call__(self, rngs=None):
        loc = jnp.zeros((self.embedding_dim,))
        scale = jnp.ones_like(loc)
        z_bg = numpyro.sample("bg", dist.Normal(loc, scale).to_event(1))
        background = self.decoder(z_bg)
        background = jnp.where(background > 0., background,
                               jnp.ones_like(background))
        return jnp.reshape(background, self.bg_shape + (1,))

def over(bg, fg):
    alphas = jnp.where(fg.sum(axis=-1, keepdims=True),
                       jnp.ones(fg.shape[:-1] + (1,)),
                       jnp.zeros(fg.shape[:-1] + (1,)))
    return alphas * fg + (1. - alphas) * bg

def captcha_model(placements: ShapePlacements,
                  backgrounder: Optional[BackgroundDecoder]=None):
    rgb_prior = dist.Uniform(0., 1.).expand((3,))
    color = numpyro.sample("color", rgb_prior.to_event(1))
    color = color[jnp.newaxis, jnp.newaxis, :]

    foreground = placements() * color
    if backgrounder is not None:
        background = backgrounder() * color
    else:
        background = jnp.ones_like(foreground)

    return utils.soft_clamp(over(background, foreground), 0., 1.)
