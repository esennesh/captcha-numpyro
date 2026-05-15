from bidict import bidict
from flax import nnx
from flax.nnx.nn.linear import canonicalize_padding
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import numpy as np
import numpyro
import numpyro.distributions as dist
from typing import Dict, Tuple, Union

from src import utils


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

@nnx.dataclass
class ShapeDictionary(nnx.Pytree):
    shapes: Float[Array, "K H W C"] = nnx.data()
    targets: Dict[str, int] = nnx.data()

    @classmethod
    def load(cls, path: str):
        def transform(img):
            img = np.array(img, dtype=jnp.float32) / 255.
            return 1. - img

        shapes = utils.load_dictionary(path, mode="L", transform=transform)
        return cls(shapes=jnp.stack(tuple(shapes.values()), axis=0),
                   targets={i: k for i, k in enumerate(shapes)})

    @classmethod
    def randomize(cls, k: int, h: int, w: int, *, rngs: nnx.Rngs):
        return cls(shapes=rngs.normal(shape=(k, h, w, 1)),
                   targets={str(c): c for c in range(k)})

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

        y = jax.lax.conv_transpose(
            inputs, kernel, strides, padding_lax,
            rhs_dilation=kernel_dilation,
            transpose_kernel=self.transpose_kernel,
            precision=self.precision,
            preferred_element_type=self.preferred_element_type,
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
    def __init__(self, prior: PlacementsPrior, shaper: ShapeConvTranspose, *,
                 rngs: nnx.Rngs):
        self.prior = prior
        self.shaper = shaper
        assert len(self.shaper.shape_dict.shapes) == len(self.prior.topography.u)

    def __call__(self, rngs=None):
        wheres = numpyro.sample("what_x_where", self.prior(rngs=rngs))
        return self.shaper(wheres)

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
