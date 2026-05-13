import kornia as pt_kornia
kornia = pt_kornia.to_jax()
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
import flax.nnx as nnx
from typing import Optional


@nnx.dataclass
class Rigid(nnx.Pytree):
    angle: Float[Array, "batch angle"] = nnx.data()
    translation: Float[Array, "batch dx dy"] = nnx.data()

    def __call__(
        self,
        imgs: Float[Array, "batch channels height width"]
    ) -> Float[Array, "batch channels height width"]:
        affine = self.transform.matrix()[:, :2, :]
        def single_affine(img, m):
            img = img[jnp.newaxis, ...]
            return kornia.geometry.transform.affine(img, m)
        batch_affine = jax.vmap(single_affine, in_axes=0, out_axes=0)
        return batch_affine(imgs, affine)

    def __matmul__(a, b):
        algebra_element = (a.transform * b.transform).log()
        angle, translation = algebra_element[:, 2:], algebra_element[:, :2]
        return Rigid(angle=angle, translation=translation, learnable=False)

    @classmethod
    def random(rngs: nnx.Rngs):
        angle = rngs.uniform(minval=-jnp.pi, maxval=jnp.pi, shape=(1, 1))
        translation = rngs.normal(shape=(1, 2))
        return Rigid(angle, translation)

    @property
    def transform(self) -> kornia.geometry.liegroup.Se2:
        from kornia.geometry.liegroup import Se2

        transform = jnp.concatenate((self.translation, self.angle), axis=-1)
        return Se2.exp(transform)

@nnx.dataclass
class Rotation(nnx.Pytree):
    angle: Float[Array, "batch angle"] = nnx.data()

    def __call__(
        self,
        imgs: Float[Array, "batch channels height width"]
    ) -> Float[Array, "batch channels height width"]:
        return kornia.geometry.transform.rotate(imgs, self.angle)

    def __matmul__(a, b):
        return Rotation(angle=(a.transform * b.transform).log(),
                        learnable=False)

    @classmethod
    def random(rngs: nnx.Rngs):
        angle = rngs.uniform(minval=-jnp.pi, maxval=jnp.pi, shape=(1, 1))
        return Rotation(angle)

    @property
    def transform(self) -> kornia.geometry.liegroup.So2:
        from kornie.liegroup.geometry import So2

        return So2.exp(self.angle)
