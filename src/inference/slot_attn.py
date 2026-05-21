import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from flax import nnx

class SlotInitializer(nnx.Module):
    def __init__(self, num_slots: int, slot_dim: int, hidden_dim: int=128,
                 pool_size: int=100, *, rngs: nnx.Rngs):
        self.num_slots = num_slots
        self.pool_size = pool_size
        self.slot_dim = slot_dim

        self.combiner = nnx.OptimizedLSTMCell(in_features=slot_dim,
                                              hidden_features=hidden_dim,
                                              keep_rngs=False, rngs=rngs)
        self.projection = nnx.Linear(hidden_dim * 2, num_slots * slot_dim,
                                     rngs=rngs)

    def __call__(
        self, inputs: Float[Array, "*batch HW in_dim"], *, rngs=None
    ) -> Float[Array, "*batch num_slots out_dim"]:
        B, HW, _ = inputs.shape
        # (B, HW // self.pool_size, in_dim)
        inputs = nnx.avg_pool(inputs, window_shape=(self.pool_size,),
                              strides=(self.pool_size,), padding="VALID")

        h = self.combiner.hidden_features
        initial_carry = (jnp.zeros((B, h)), jnp.zeros((B, h)))
        carries, _ = jax.lax.scan(self.combiner, initial_carry,
                                  inputs.swapaxes(0, 1))
        xs = jnp.concatenate(carries, axis=-1)
        return self.projection(xs).reshape(B, self.num_slots, self.slot_dim)

class SlotAttention(nnx.Module):
    """Iterative slot attention (Locatello et al., 2020).  Mainly derived from
       the canonical Tensorflow implementation by Claude. Modified to use the
       above bidirectional recurrent slot initializer by Eli Sennesh."""

    def __init__(self, num_slots: int, num_iterations: int, slot_dim: int,
                 hidden_dim: int=64, mlp_hidden_dim: int = 128,
                 eps: float = 1e-8, *, rngs: nnx.Rngs):
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.slot_dim = slot_dim
        self.eps = eps
        self.scale = slot_dim ** -0.5

        # Slot initialization network
        self.slot_initializer = SlotInitializer(num_slots, slot_dim,
                                                hidden_dim, rngs=rngs)

        # Layer norms
        self.norm_inputs  = nnx.LayerNorm(slot_dim, rngs=rngs)
        self.norm_slots   = nnx.LayerNorm(slot_dim, rngs=rngs)
        self.norm_pre_ff  = nnx.LayerNorm(slot_dim, rngs=rngs)

        # Attention projections — no bias, as in the paper
        self.to_q = nnx.Linear(slot_dim, slot_dim, use_bias=False, rngs=rngs)
        self.to_k = nnx.Linear(slot_dim, slot_dim, use_bias=False, rngs=rngs)
        self.to_v = nnx.Linear(slot_dim, slot_dim, use_bias=False, rngs=rngs)

        # GRU for slot updates
        self.gru = nnx.GRUCell(slot_dim, slot_dim, rngs=rngs)

        # MLP residual
        self.mlp = nnx.Sequential(
            nnx.Linear(slot_dim, mlp_hidden_dim, rngs=rngs), nnx.relu,
            nnx.Linear(mlp_hidden_dim, slot_dim, rngs=rngs),
        )

    def __call__(self, inputs: jax.Array, rngs: nnx.Rngs | None = None) -> jax.Array:
        # slots: (B, N, slot_dim)
        B = inputs.shape[0]

        # Initialize slots deterministically via an LSTM sweep
        slots = self.slot_initializer(inputs, rngs=rngs)      # (B, S, slot_dim)

        # K and V are fixed across iterations
        inputs_normed = self.norm_inputs(inputs)
        k = self.to_k(inputs_normed)                          # (B, N, slot_dim)
        v = self.to_v(inputs_normed)                          # (B, N, slot_dim)

        for _ in range(self.num_iterations):
            slots_prev = slots
            q = self.to_q(self.norm_slots(slots))             # (B, S, slot_dim)

            # Attention logits over (inputs × slots): (B, N, S)
            dots = jnp.einsum('bnd,bsd->bns', k, q) * self.scale

            # Softmax over slots (competition), then normalise over inputs (weighted mean)
            attn = jax.nn.softmax(dots, axis=-1) + self.eps  # (B, N, S)
            attn = attn / attn.sum(axis=-2, keepdims=True)   # (B, N, S)

            # Aggregate values into slot updates: (B, S, slot_dim)
            updates = jnp.einsum('bns,bnd->bsd', attn, v)

            # GRU update — reshape so each slot is an independent sequence step
            slots, _ = self.gru(
                slots_prev.reshape(B * self.num_slots, self.slot_dim),
                updates.reshape(B * self.num_slots, self.slot_dim),
            )
            slots = slots.reshape(B, self.num_slots, self.slot_dim)

            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots                                          # (B, S, slot_dim)

class SlotAttentionEncoder(nnx.Module):
    """CNN backbone + positional embedding + slot attention."""

    def __init__(self, resolution: tuple[int, int], num_slots: int,
                 num_iterations: int, eps: float = 1e-8, hidden_dim: int = 64,
                 slot_dim: int = 64, mlp_hidden_dim: int = 128, *,
                 rngs: nnx.Rngs):
        H, W = resolution

        # Four-layer CNN — stride-1, so spatial resolution is preserved
        self.cnn = nnx.Sequential(
            nnx.Conv(3, hidden_dim, (5, 5), padding='SAME', rngs=rngs),
            nnx.relu,
            nnx.Conv(hidden_dim, hidden_dim, (5, 5), padding='SAME', rngs=rngs),
            nnx.relu,
            nnx.Conv(hidden_dim, hidden_dim, (5, 5), padding='SAME', rngs=rngs),
            nnx.relu,
            nnx.Conv(hidden_dim, hidden_dim, (5, 5), padding='SAME', rngs=rngs),
            nnx.relu
        )

        # Post-CNN layer norm + MLP (projects to slot_dim if it differs from hidden_dim)
        self.encoder_norm = nnx.LayerNorm(hidden_dim, rngs=rngs)
        self.encoder_mlp  = nnx.Sequential(
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs), nnx.relu,
            nnx.Linear(hidden_dim, slot_dim,   rngs=rngs),
        )

        # Learned soft positional embedding, same shape as a feature map
        self.pos_embedding = nnx.Param(jnp.zeros((1, H, W, hidden_dim)))

        self.slot_attention = SlotAttention(
            num_slots=num_slots, num_iterations=num_iterations,
            slot_dim=slot_dim, hidden_dim=hidden_dim,
            mlp_hidden_dim=mlp_hidden_dim, eps=eps, rngs=rngs,
        )

    def __call__(self, image: jax.Array, rngs: nnx.Rngs | None = None) -> jax.Array:
        # image: (B, H, W, 3) — channels-last, values in [0, 1]
        B, H, W, _ = image.shape
        x = self.cnn(image)                          # (B, H, W, hidden_dim)
        x = x + self.pos_embedding.value             # soft positional embedding
        x = x.reshape(x.shape[0], -1, x.shape[-1])   # (B, H*W, hidden_dim)
        x = self.encoder_mlp(self.encoder_norm(x))   # (B, H*W, slot_dim)
        return self.slot_attention(x)      # (B, num_slots, slot_dim)
