import numpyro.distributions as dist
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from flax import nnx
from numpyro.contrib.module import nnx_module
import numpyro

from src.inference import slot_attn


class BackgroundEncoder(nnx.Module):
    def __init__(self, embedding_dim: int=50, height=60, hiddens=400, width=160,
                 *, rngs: nnx.Rngs):
        self.bg_shape = (height, width)
        self.embedding_dim = embedding_dim
        self.encoder = nnx.Sequential(
            nnx.Linear(height * width, hiddens, rngs=rngs), nnx.silu,
            nnx.Linear(hiddens, embedding_dim * 2, rngs=rngs)
        )

    def __call__(self, images, rngs: nnx.Rngs=None):
        images = jax.lax.collapse(images, -2)
        params = self.encoder(images).reshape(-1, self.embedding_dim, 2)
        loc, scale = params[..., 0], jnp.exp(params[..., 1])
        return numpyro.sample("bg", dist.Normal(loc, scale).to_event(1))

class ColorFinder(nnx.Module):
    def __init__(self, hidden_dim=128, *, rngs: nnx.Rngs):
        self.cnn = nnx.Sequential(
            nnx.Conv(3,  16, (3, 3), strides=(2, 2), padding='SAME', rngs=rngs),
            nnx.relu,
            nnx.Conv(16, 32, (3, 3), strides=(2, 2), padding='SAME', rngs=rngs),
            nnx.relu,
            nnx.Conv(32, 64, (3, 3), strides=(2, 2), padding='SAME', rngs=rngs),
            nnx.relu,
            nnx.Conv(64, 64, (3, 3), strides=(2, 2), padding='SAME', rngs=rngs),
            nnx.relu,
        )
        self.linear_head = nnx.Linear(64, 3 * 2, rngs=rngs)

    def __call__(
        self, images: Float[Array, "*batch H W channels"]
    ) -> Float[Array, "*batch channels"]:
        x = self.cnn(images)          # (B, 4, 10, 64)
        x = x.mean(axis=(1, 2))       # global average pool → (B, 64)
        params = self.linear_head(x).reshape(-1, 2, 3) # (B, 6)
        rgb_q = dist.Normal(jax.nn.sigmoid(params[:, 0]),
                            jax.nn.softplus(params[:, 1]))
        return numpyro.sample("color", rgb_q.to_event(1))

class ShapePlacer(nnx.Module):
    def __init__(self, kw: int=40, kh: int=40, hidden_dim=128, img_w: int=160,
                 img_h: int=60, num_features: int=36, num_iterations: int=10,
                 stride: int=1, *, rngs: nnx.Rngs):
        self.num_features = num_features
        self.slots_h = (img_h - kh) // stride + 1
        self.slots_w = (img_w - kw) // stride + 1

        self.slot_encoder = slot_attn.SlotAttentionEncoder(
            (img_h, img_w), self.slots_h * self.slots_w, num_iterations,
            rngs=rngs, slot_dim=self.num_features
        )

    def __call__(
        self, images: Float[Array, "*batch IH IW channels"]
    ) -> Float[Array, "*batch K OH OW"]:
        u_hw = self.slot_encoder(images).reshape(-1, self.num_features,
                                                 self.slots_h, self.slots_w)
        rate = jnp.exp(u_hw)
        return numpyro.sample("what_x_where", dist.Poisson(rate).to_event(3))

def captcha_guide(images, backgrounder: BackgroundEncoder,
                  color_finder: ColorFinder, placements: ShapePlacer):
    backgrounder = nnx_module("backgrounder_q", backgrounder)
    color_finder = nnx_module("color_finder", color_finder)
    placements = nnx_module("placements_q", placements)

    with numpyro.plate("batch", images.shape[0]):
        c = color_finder(images)
        backgrounder(images.mean(axis=-1))
        placements(images)
