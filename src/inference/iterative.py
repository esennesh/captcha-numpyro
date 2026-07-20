import jax
import numpyro
from numpyro.handlers import block, substitute, trace
from numpyro.infer.autoguide import AutoGuide
from numpyro.infer.initialization import init_to_sample
import optimistix
from typing import Optional

from src.inference.tracer import ParticleTracer
from src.utils import is_autoguide

def exponential_decay(init_value: float, transition_steps: int,
                      decay_rate: float):
    """Drop-in for ``optax.exponential_decay`` whose closure equinox can
    convert: optax's version carries an empty closure cell that crashes
    ``OptaxMinimiser``'s closure-to-pytree pass."""
    def schedule(count):
        return init_value * decay_rate ** (count / transition_steps)
    return schedule

class IterativeGuide(AutoGuide):
    """Iterative (per-batch) variational inference wrapped as a guide.

    ``adapt`` re-initializes the inner guide's parameters for the batch at
    hand and minimises the tracer's variational objective over them with an
    ``optimistix`` solver. The base noise is drawn once per call and held
    fixed across every objective evaluation (sample-average approximation),
    so the inner problem the solver sees is deterministic. With the default
    ``max_steps=None`` the solver runs to convergence; a finite ``max_steps``
    imposes an iteration budget and accepts whatever point it reaches.

    The adapted parameters are constants to the outer learning loop: model
    parameters receive gradients only through the objective evaluated at the
    adapted guide, never through the inner solve.
    """

    def __init__(self, model, guide, tracer: ParticleTracer,
                 solver: Optional[optimistix.AbstractMinimiser]=None,
                 max_steps: Optional[int]=None, *, create_plates=None,
                 prefix="auto"):
        self.guide = guide
        self.model = model
        if solver is None:
            solver = optimistix.NonlinearCG(rtol=1e-6, atol=1e-6)
        self.solver = solver
        self.max_steps = max_steps
        self.tracer = tracer

        super().__init__(model, init_loc_fn=init_to_sample, prefix=prefix,
                         create_plates=create_plates)

    def adapt(self, *args, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        # Hide the inner guide's own params from outer handlers so each batch
        # starts from the guide's fresh init values; model params still come
        # from the surrounding substitute.
        def hide_guide_params(msg):
            return msg["type"] == "param" and\
                   msg["name"] not in self.model_params
        with block(hide_fn=hide_guide_params):
            guide_trace = trace(self.guide).get_trace(*args, **kwargs)

        buffers, params = {}, {}
        model_params = {k: numpyro.param(k) for k in self.model_params}
        for name, site in guide_trace.items():
            if name in self.model_params:
                continue
            if site["type"] == "param":
                params[name] = site["value"]
            elif site["type"] == "mutable":
                buffers[name] = site["value"]
        buffers.update(**{k: v for k, v in model_params.items()
                          if v is not None})

        # The objective must be a fresh closure on every call: optimistix
        # closure-converts it, and JAX caches the conversion (including any
        # closed-over tracers) keyed on the function's identity. A long-lived
        # function here leaks one jit trace's tracers into the next
        # (e.g. train_step's into valid_step's).
        buffers = jax.lax.stop_gradient(buffers)
        rng = numpyro.prng_key()
        def objective(params, _):
            return self.tracer.loss(rng, params | buffers, {}, self.model,
                                    self.guide, *args, **kwargs)[0]
        solution = optimistix.minimise(objective, self.solver, params,
                                       args=None, max_steps=self.max_steps,
                                       throw=self.max_steps is None)
        return jax.lax.stop_gradient(solution.value)

    def __call__(self, *args, adaptation=None, **kwargs):
        if self.prototype_trace is None:
            self._setup_prototype(*args, **kwargs)

        guide = substitute(self.guide, data=adaptation) if adaptation\
                else self.guide
        with block(expose_types=["sample"]):
            return guide(*args, **kwargs)

    @property
    def model_params(self):
        return {name for name, site in self.prototype_trace.items()
                if site["type"] == "param"}

    def sample_posterior(self, rng_key, params, *args, sample_shape=(),
                         **kwargs):
        raise NotImplementedError()

    def _setup_prototype(self, *args, **kwargs):
        self.guide = self.guide(self.model) if is_autoguide(self.guide)\
                     else self.guide
        self.guide._setup_prototype(*args, **kwargs)
        self.prototype_trace = self.guide.prototype_trace
