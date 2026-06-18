from abc import ABC
from collections import defaultdict
from functools import cached_property
import functools
import jax
import jax.numpy as jnp
import jax.random as random
import numpyro
from numpyro.infer.elbo import MultiFrameTensor
from numpyro.infer.util import compute_log_probs, get_importance_trace
from numpyro._typing import Message
from numpyro.util import _validate_model, check_model_guide_match

from typing import Callable, Dict, Optional

def broadcast_observed(observed, target_shape):
    while observed.ndim < len(target_shape):
        observed = jnp.expand_dims(observed, axis=-1)
    return jnp.broadcast_to(observed, target_shape)

def configure_sample(msg: Message, /, **kwargs) -> Dict:
    return kwargs

def expected_value(site: Dict):
    fn = site.get("fn", None)
    mean = getattr(fn, "mean", None)
    if mean is None:
        return site["value"]
    return mean() if callable(mean) else mean

def trace_entry(site: Dict, log_p, log_q, observed):
    return {
        "value": site["value"],
        "log_p": log_p,
        "log_q": log_q,
        "observed": observed,
        "ev": expected_value(site),
    }

class VariationalMixin(ABC):
    def log_weights(self, traces, mutables):
        raise NotImplementedError

    def loss_fn(self, log_ws, traces):
        raise NotImplementedError

class ELBOMixin(VariationalMixin):
    def log_weights(self, traces, mutables):
        log_ws = 0.
        beta = getattr(self, "beta", 1.)
        for name, site in traces.items():
            term = site["log_p"] - site["log_q"]
            log_ws = log_ws + jnp.where(site["observed"], term, beta * term)
        return log_ws

    def loss_fn(self, log_ws, traces):
        return -jnp.mean(log_ws, axis=0).sum()

class IwaeMixin(ELBOMixin):
    def loss_fn(self, log_ws, traces):
        return -jax.nn.logmeanexp(log_ws)

class ParticleTracer(ELBOMixin):
    def __init__(self, beta: float=1., num_particles: int=1):
        self.beta = beta
        self.num_particles = num_particles

    def __call__(self, rng_key, param_map, particle_params, model, guide,
                 *args, **kwargs):
        param_map = param_map.copy()
        particle_params = jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (self.num_particles,
                                                 *leaf.shape))
                         if leaf.shape[0] != self.num_particles else leaf,
            particle_params
        )

        def single_trace(rng_key, pwise_params, param_map=param_map,
                         particle=None):
            import functools

            param_map = param_map | pwise_params
            particle_guide, particle_model = guide, model

            model_seed, guide_seed = random.split(rng_key)
            if particle is not None:
                particle_guide = numpyro.handlers.infer_config(
                    particle_guide,
                    functools.partial(configure_sample, k=particle)
                )
                particle_model = numpyro.handlers.infer_config(
                    particle_model,
                    functools.partial(configure_sample, k=particle)
                )
            seeded_model = numpyro.handlers.seed(particle_model, model_seed)
            seeded_guide = numpyro.handlers.seed(particle_guide, guide_seed)
            model_trace, guide_trace = get_importance_trace(seeded_model,
                                                            seeded_guide, args,
                                                            kwargs, param_map)

            check_model_guide_match(model_trace, guide_trace)
            _validate_model(model_trace, plate_warning="loose")

            graph_state = {
                name: trace_entry(
                    site,
                    site["log_prob"],
                    guide_trace[name]["log_prob"] if name in guide_trace
                    else jnp.zeros_like(site["log_prob"]),
                    site["is_observed"],
                )
                for name, site in model_trace.items()
                if site["type"] == "sample"
            }
            graph_state.update({
                name: trace_entry(
                    site,
                    jnp.zeros_like(site["log_prob"]),
                    site["log_prob"],
                    False,
                )
                for name, site in guide_trace.items()
                if site["type"] == "sample" and name not in graph_state
            })
            graph_state.update({
                name: trace_entry(site, 0., 0., False)
                for name, site in model_trace.items()
                if site["type"] == "deterministic"
            })
            graph_state.update({
                name: trace_entry(site, 0., 0., False)
                for name, site in guide_trace.items()
                if site["type"] == "deterministic"
            })
            mutables = {name: site["value"] for name, site in
                        model_trace.items() if site["type"] == "mutable"}

            return graph_state, mutables

        rng_keys = random.split(rng_key, self.num_particles)
        particles = jnp.arange(self.num_particles)
        particle_traces = jax.vmap(single_trace)
        return particle_traces(rng_keys, particle_params, particle=particles)

    def loss(self, *args, **kwargs):
        traces, mutables = self(*args, **kwargs)
        for k, v in traces.items():
            is_observed = broadcast_observed(v["observed"],
                                             v["log_p"].shape)
            traces[k] = v | {"observed": is_observed}
        log_ws = self.log_weights(traces, mutables)
        return self.loss_fn(log_ws, traces), {"log_w": log_ws.sum(axis=-1),
                                              "mutables": mutables,
                                              "trace": traces}

    def setup(self, guide_deps, model_deps, guide_trace, model_trace):
        pass

class ELBOTracer(ParticleTracer):
    def __init__(self, num_particles: int=1):
        super().__init__(num_particles=num_particles)
        self._guide_deps, self._model_deps = None, None
        self._guide_properties, self._model_properties = {}, {}

    def log_weights(self, traces, mutables):
        if jax.tree.leaves(mutables):
            return super().log_weights(traces, mutables)
        log_ws = jnp.array(0.0)
        # mapping from non-reparameterizable sample sites to cost terms
        # influenced by each of them
        downstream_costs: Dict[str, MultiFrameTensor] =\
            defaultdict(lambda: MultiFrameTensor())
        for name, site in traces.items():
            log_ws = log_ws + site["log_p"]
            for key in self._model_deps.get(name, []):
                downstream_costs[key].add((
                    self._model_properties[name]["cond_indep_stack"],
                    site["log_p"]
                ))
            if name in self._guide_properties:
                log_q = site["log_q"]
                if not self._guide_properties[name]["reparameterized"]:
                    log_q = jax.lax.stop_gradient(log_q)
                log_ws = log_ws - log_q
                for key in self._guide_deps[name]:
                    downstream_costs[key].add((
                        self._guide_properties[name]["cond_indep_stack"],
                        -site["log_q"]
                    ))

        for node, cost in downstream_costs.items():
            downstream_cost = cost.sum_to(
                self._guide_properties[node]["cond_indep_stack"]
            )
            advantage = downstream_cost - downstream_cost.mean(axis=0)
            surrogate = traces[node]["log_q"] * jax.lax.stop_gradient(
                advantage
            )
            log_ws = log_ws + surrogate - jax.lax.stop_gradient(surrogate)
        return log_ws

    def loss_fn(self, log_ws, traces):
        reparameterized = all(site["reparameterized"] for site
                              in self._guide_properties.values())
        if reparameterized:
            return super().loss_fn(log_ws, traces)
        return -(jnp.sum(log_ws, axis=0) / (log_ws.shape[0] - 1)).sum()

    def setup(self, guide_deps, model_deps, guide_trace, model_trace):
        self._guide_deps, self._model_deps = guide_deps, model_deps
        for name, site in guide_trace.items():
            if site["type"] != "sample":
                continue

            self._guide_properties[name] = {
                "cond_indep_stack": site["cond_indep_stack"],
                "reparameterized": site["fn"].has_rsample
            }

        for name, site in model_trace.items():
            if site["type"] != "sample":
                continue

            self._model_properties[name] = {
                "cond_indep_stack": site["cond_indep_stack"],
            }

class OvisTracer(ParticleTracer):
    def __init__(self, beta=1., include_aux=True, num_particles: int=1,
                 num_auxiliary: Optional[int]=None):
        self._guide_deps, self._model_deps = None, None
        self._guide_properties, self._model_properties = {}, {}
        self._include_aux = include_aux
        if not num_auxiliary:
            num_auxiliary = num_particles
        self._num_aux = num_auxiliary
        super().__init__(beta=beta, num_particles=num_particles + num_auxiliary)

    @cached_property
    def control_variate(self):
        def fn(log_ws, log_aux):
            # log_ws: K x B
            # log_aux: S x B
            B, K, S = log_ws.shape[-1], log_ws.shape[0], log_aux.shape[0]

            log_ws = jnp.expand_dims(log_ws, (0, 1)) # -> 1 x 1 X K x B
            log_ws = jnp.broadcast_to(log_ws, (S, K, K, B))
            log_aux = jnp.expand_dims(log_aux, (1, 2)) # -> S x 1 x 1 x B
            log_aux = jnp.broadcast_to(log_aux, (S, K, K, B))

            mask = jnp.expand_dims(jnp.identity(K), (0, -1)) # -> 1 x K x K x 1
            log_w_hat = (1 - mask) * log_ws + mask * log_aux # S x K x K x B
            # S x K x K x B -> S x B x K x K
            objectives = jnp.moveaxis(self.objective(log_w_hat, axis=-2), -1, 1)
            # S x B x K x K -> S x B x K -> B x K
            results = jnp.diagonal(objectives, axis1=-2, axis2=-1).mean(axis=0)
            # B x K -> K x B
            return jnp.moveaxis(results, 0, -1)
        return fn

    def loss_fn(self, log_ws, traces):
        num_particles = self.num_particles - self._num_aux
        log_weights, log_aux = log_ws[:num_particles], log_ws[num_particles:]

        rewards = self.objective(log_weights, axis=0)
        values = self.control_variate(log_weights, log_aux)
        advantages = rewards - values

        if self._include_aux:
            log_evidence = jax.nn.logmeanexp(log_ws, axis=0)
        else:
            log_evidence = jax.nn.logmeanexp(log_weights, axis=0)

        surrogates = jnp.zeros_like(log_weights)
        for name, site in traces.items():
            if name in self._guide_properties and\
               not self._guide_properties[name]["reparameterized"]:
                log_q = site["log_q"][:num_particles]
                surrogate = log_q * jax.lax.stop_gradient(advantages)
                surrogates = surrogates + surrogate
        surrogates = surrogates.sum(axis=0)
        loss = -(log_evidence + surrogates - jax.lax.stop_gradient(surrogates))
        return loss.sum()

    @cached_property
    def objective(self):
        def fn(log_ws, axis=0):
            return jax.nn.logmeanexp(log_ws, axis=axis, keepdims=True) -\
                   jax.nn.softmax(log_ws, axis=axis)
        return fn

    def setup(self, guide_deps, model_deps, guide_trace, model_trace):
        self._guide_deps, self._model_deps = guide_deps, model_deps
        for name, site in guide_trace.items():
            if site["type"] != "sample":
                continue

            self._guide_properties[name] = {
                "cond_indep_stack": site["cond_indep_stack"],
                "reparameterized": site["fn"].has_rsample
            }

        for name, site in model_trace.items():
            if site["type"] != "sample":
                continue

            self._model_properties[name] = {
                "cond_indep_stack": site["cond_indep_stack"],
            }

class VarGradMixin(VariationalMixin):
    def log_weights(self, traces, mutables):
        return sum(jnp.sum(site["log_p"], axis=-1) -
                   jnp.sum(site["log_q"], axis=-1)
                   for name, site in traces.items())

    def loss_fn(self, log_ws, traces):
        return (jnp.var(-log_ws, axis=0, ddof=1.) / 2).sum()

class VarGradTracer(VarGradMixin, ParticleTracer):
    pass

class AdaptiveParticleTracer(ParticleTracer):
    def __call__(self, rng_key, param_map, particle_params, model, guide,
                 *args, **kwargs):
        param_map = param_map.copy()
        particle_params = jax.tree.map(
            lambda leaf: jnp.broadcast_to(leaf, (self.num_particles,
                                                 *leaf.shape))
                         if leaf.shape[0] != self.num_particles else leaf,
            particle_params
        )
        if hasattr(guide, "adapt") and isinstance(guide.adapt, Callable):
            adaptation_rng, rng_key = random.split(rng_key)

            adapt = numpyro.handlers.seed(guide.adapt, adaptation_rng)
            with numpyro.handlers.substitute(data=param_map):
                adapted_params = adapt(*args, **kwargs)
            guide = functools.partial(guide, adaptation=adapted_params)

        def single_trace(rng_key, pwise_params, param_map=param_map,
                         particle=None):
            import functools

            param_map = param_map | pwise_params
            particle_guide, particle_model = guide, model

            model_seed, guide_seed = random.split(rng_key)
            if particle is not None:
                particle_guide = numpyro.handlers.infer_config(
                    particle_guide,
                    functools.partial(configure_sample, k=particle)
                )
                particle_model = numpyro.handlers.infer_config(
                    particle_model,
                    functools.partial(configure_sample, k=particle)
                )
            seeded_model = numpyro.handlers.seed(particle_model, model_seed)
            seeded_guide = numpyro.handlers.seed(particle_guide, guide_seed)
            model_trace, guide_trace = get_importance_trace(seeded_model,
                                                            seeded_guide, args,
                                                            kwargs, param_map)

            check_model_guide_match(model_trace, guide_trace)
            _validate_model(model_trace, plate_warning="loose")

            graph_state = {
                name: trace_entry(
                    site,
                    site["log_prob"],
                    guide_trace[name]["log_prob"] if name in guide_trace
                    else jnp.zeros_like(site["log_prob"]),
                    site["is_observed"],
                )
                for name, site in model_trace.items()
                if site["type"] == "sample"
            }
            graph_state.update({
                name: trace_entry(
                    site,
                    jnp.zeros_like(site["log_prob"]),
                    site["log_prob"],
                    False,
                )
                for name, site in guide_trace.items()
                if site["type"] == "sample" and name not in graph_state
            })
            graph_state.update({
                name: trace_entry(site, 0., 0., False)
                for name, site in model_trace.items()
                if site["type"] == "deterministic"
            })
            graph_state.update({
                name: trace_entry(site, 0., 0., False)
                for name, site in guide_trace.items()
                if site["type"] == "deterministic"
            })
            mutables = {name: site["value"] for name, site in
                        model_trace.items() if site["type"] == "mutable"}

            return graph_state, mutables

        rng_keys = random.split(rng_key, self.num_particles)
        particles = jnp.arange(self.num_particles)
        particle_traces = jax.vmap(single_trace)
        return particle_traces(rng_keys, particle_params, particle=particles)

class AdaptiveElboTracer(AdaptiveParticleTracer, ELBOTracer):
    pass

class DoubleCVTracer(ParticleTracer):
    """Double control variate for discrete latents (Titsias & Shi, 2022),
    'more computation' variant: the control variate is the first-order Taylor
    expansion of the ELBO integrand about the variational mean E_q[z], using
    grad_z ELBO evaluated there (their Eq. 7).

    DRAFT — written against the tracer interface but NOT yet run.

      grad_phi ELBO = E[(f - f~) s_phi]            (CV2: REINFORCE the remainder)
                    + grad_phi E_q[f~],            (CV1: exact = g . grad_phi E_q[z])
      f~(z) = f(zbar) + g . (z - zbar),  zbar = E_q[z],  g = grad_z f(zbar).

    Only the *sampling* of z is non-differentiable; the integrand
    f(z) = log p(x, z) - log q(z) is smooth in z, so g = grad_z f at the mean is
    one extra backward pass through the decoder — the 'more computation' the
    paper trades for lower variance, and it needs no analytic E_q, so it
    survives a nonlinear decoder. E_q[z] for every guide latent is obtained
    generically with the ``posterior_mean`` handler.

    Assumes each latent's log_prob has a smooth continuous extension (Poisson
    via gammaln) and a defined ``.mean``, and that ``param_map`` carries the
    guide params (true for the graphical monad; batchvi keeps them elsewhere).
    """
    def __init__(self, *args, online=False, **kwargs):
        self._online = online
        super().__init__(*args, **kwargs)

    def _guide_means(self, rng_key, param_map, guide, args, kwargs):
        seeded = numpyro.handlers.substitute(
            numpyro.handlers.seed(guide, rng_key), data=param_map)
        gtr = numpyro.handlers.trace(
            posterior_mean(seeded)).get_trace(*args, **kwargs)
        return {n: s["value"] for n, s in gtr.items()
                if s["type"] == "sample" and not s.get("is_observed", False)}

    def loss(self, rng_key, param_map, particle_params, model, guide,
             *args, **kwargs):
        sg = jax.lax.stop_gradient
        # 1. K particles from the parent sampler
        traces, mutables = self(rng_key, param_map, particle_params, model,
                                guide, *args, **kwargs)
        logp_total = sum(site[1] * (site[3] | (not self._online)) for site in
                         traces.values())    # (K, B)
        logq_total = sum(site[2] * (site[3] | (not self._online)) for site in
                         traces.values())    # (K, B)
        f = logp_total - logq_total                              # (K, B)

        # 2. variational means E_q[z] for every guide latent (live in phi)
        means = self._guide_means(rng_key, param_map, guide, args, kwargs)
        zbar = {k: sg(v) for k, v in means.items()}              # detached refs
        frozen = sg(param_map)

        # 3. ELBO integrand and its z-gradient AT the means (Eq. 7)
        def elbo_at(values):
            m_lp, _ = compute_log_probs(model, args, kwargs,
                                        {**frozen, **values})
            q_lp, _ = compute_log_probs(guide, args, kwargs,
                                        {**frozen, **values})
            fbar = sum(v * (traces[k][3] | (not self._online)) for k, v in
                       m_lp.items()) -\
                   sum(q_lp[k] * (traces[k][3] | (not self._online)) for k in
                       values)  # (B,)
            return fbar.sum(), fbar
        g, fbar = jax.grad(elbo_at, has_aux=True)(zbar)          # g: {n: (B, D)}
        g = {k: sg(v) for k, v in g.items()}
        fbar = sg(fbar)                                          # (B,)

        # 4. linear surrogate f~(z) and its (small) remainder
        ftilde = fbar[None]                                      # (1, B) -> (K, B)
        for name, gz in g.items():
            zk = traces[name][0]                                 # (K, B, D)
            ftilde = ftilde + jnp.sum(gz[None] * (zk - zbar[name][None]),
                                      axis=-1)
        remainder = f - ftilde                                   # (K, B)
        K = remainder.shape[0]
        baseline = (remainder.sum(0, keepdims=True) - remainder) / (K - 1)
        advantage = sg(remainder - baseline)                     # (K, B)

        # 5. surrogate: theta + value pathwise (entropy detached in phi);
        #    CV2 REINFORCE on the remainder; CV1 exact grad_phi E_q[f~]
        #    = g . grad_phi E_q[z]  (value 0, gradient is the reparam-like term)
        cv2 = logq_total * advantage                             # (K, B)
        cv1 = sum(jnp.sum(g[k] * (means[k] - sg(means[k])), axis=-1)
                  for k in g)                                    # (B,)
        elbo = (logp_total - sg(logq_total)) + (cv2 - sg(cv2))   # (K, B)
        loss = -(elbo.mean(axis=0) + cv1).sum()

        return loss, {"log_w": f.sum(axis=-1), "mutables": mutables,
                      "trace": traces}
