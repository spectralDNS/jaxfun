import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax

pi = jnp.pi
key = jax.random.PRNGKey(0)


def f(x):
    # -u''(x) = pi^2 sin(pi x)  with solution u(x)=sin(pi x)
    return (pi**2) * jnp.sin(pi * x)


class NN(nnx.Module):
    def __init__(self, widths: tuple[int], key):
        super().__init__()
        keys = jax.random.split(key, len(widths) + 1)
        dims = [1] + list(widths) + [1]

        self.ws = nnx.List()
        self.bs = nnx.List()

        for k, din, dout in zip(keys, dims[:-1], dims[1:]):
            wk, bk = jax.random.split(k)
            W = jax.random.normal(wk, (din, dout)) * jnp.sqrt(2.0 / din)
            b = jnp.zeros((dout,), dtype=W.dtype)
            self.ws.append(nnx.Param(W))
            self.bs.append(nnx.Param(b))

    def __call__(self, x):
        z = x
        for W, b in zip(self.ws[:-1], self.bs[:-1]):
            z = jax.nn.tanh(z @ W + b)
        return z @ self.ws[-1] + self.bs[-1]


def pack_params(model):
    ws = tuple(jnp.asarray(W) for W in model.ws)
    bs = tuple(jnp.asarray(b) for b in model.bs)
    return (ws, bs)


def apply_params(model, params):
    ws, bs = params
    assert len(ws) == len(model.ws) == len(bs) == len(model.bs)
    for Wi, bi, Wp, bp in zip(model.ws, model.bs, ws, bs):
        Wi.value = Wp
        bi.value = bp


def forward_params(params, x2d):
    ws, bs = params
    z = x2d
    for W, b in zip(ws[:-1], bs[:-1]):
        z = jax.nn.tanh(z @ W + b)
    return z @ ws[-1] + bs[-1]


def u_hat_params(params, x):
    return x * (1.0 - x) * forward_params(params, x[:, None])[:, 0]


def u_scalar_params(params, t):
    return u_hat_params(params, jnp.array([t]))[0]


def u_x_params(params, x):
    return jax.jacfwd(lambda t: u_scalar_params(params, t))(x)


def u_xx_params(params, x):
    return jax.jacfwd(jax.jacrev(lambda t: u_scalar_params(params, t)))(x)


def residual_params(params, x):
    return -jax.vmap(lambda t: u_xx_params(params, t))(x) - f(x)


def loss_fn_params(params, x):
    r = residual_params(params, x)
    return jnp.mean(r**2)


# ---------- Train ----------
@nnx.jit(static_argnames=["opt"])
def train_step_adam(model, opt_state, x, opt):
    params = pack_params(model)
    loss, grads = jax.value_and_grad(loss_fn_params)(params, x)
    updates, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    apply_params(model, new_params)
    return model, opt_state, loss


#train_step_adam = jax.jit(_train_step_adam, static_argnames=["opt"])

@nnx.jit(static_argnames=["opt"])
def train_step_lbfgs(model, opt_state, x, opt):
    params = pack_params(model)
    value_and_grad_fn = optax.value_and_grad_from_state(loss_fn_params)
    value, grads = value_and_grad_fn(params, x, state=opt_state)
    updates, opt_state = opt.update(
        grads,
        opt_state,
        params,
        value=value,
        grad=grads,
        value_fn=lambda p: loss_fn_params(p, x),
    )
    new_params = optax.apply_updates(params, updates)
    apply_params(model, new_params)
    return model, opt_state, value


#train_step_lbfgs = jax.jit(_train_step_lbfgs, static_argnames=["opt"])


def train(model, opt, x, steps, lbfgs=False):
    opt_state = opt.init(pack_params(model) if lbfgs else pack_params(model))
    losses = []
    for i in range(steps):
        if lbfgs:
            model, opt_state, L = train_step_lbfgs(model, opt_state, x, opt)
        else:
            model, opt_state, L = train_step_adam(model, opt_state, x, opt)
        losses.append(float(L))
        if (i + 1) % 100 == 0:
            print(f"step {i + 1:5d} | loss = {L:.3e}")
    return model, jnp.array(losses)


# ---------- Data ----------
N = 10000
key, k1, k2, kx = jax.random.split(key, 4)
x_train = jax.random.uniform(kx, (N,), minval=0.0, maxval=1.0, dtype=jnp.float32)
x_eval = jnp.linspace(0.0, 1.0, 400, dtype=jnp.float32)
u_true = jnp.sin(pi * x_eval)

# ---------- Adam ----------
print("\nTraining with Adam")
model_adam = NN((32, 32), key=k1)
opt_adam = optax.adam(1e-3)
t0 = time.time()
model_adam, losses_adam = train(model_adam, opt_adam, x_train, steps=4000, lbfgs=False)
params_adam = pack_params(model_adam)
print(f"Time Adam: {time.time() - t0:.1f}s")
# ---------- L-BFGS ----------
print("\nTraining with L-BFGS (Optax)")
t1 = time.time()
model_lbfgs = NN((32, 32), key=k2)
opt_lbfgs = optax.lbfgs()
model_lbfgs, losses_lbfgs = train(
    model_lbfgs, opt_lbfgs, x_train, steps=1000, lbfgs=True
)
params_lbfgs = pack_params(model_lbfgs)
print(f"Time L-BFGS: {time.time() - t1:.1f}s")
# ---------- Eval ----------
u_pred_adam = u_hat_params(params_adam, x_eval)
u_pred_lbfgs = u_hat_params(params_lbfgs, x_eval)

relL2_adam = float(jnp.linalg.norm(u_pred_adam - u_true) / jnp.linalg.norm(u_true))
relL2_lbfgs = float(jnp.linalg.norm(u_pred_lbfgs - u_true) / jnp.linalg.norm(u_true))

print(f"\nRelative L2 error — Adam :  {relL2_adam:.3e}")
print(f"Relative L2 error — L-BFGS: {relL2_lbfgs:.3e}")

# ----- Plots -----
plt.figure()
plt.plot(x_eval, u_true, color="black", linewidth=2.5, linestyle="-", label="True")
plt.plot(
    x_eval, u_pred_adam, color="red", linewidth=1.8, linestyle="-.", label="PINN (Adam)"
)
plt.plot(
    x_eval,
    u_pred_lbfgs,
    color="blue",
    linewidth=1.8,
    linestyle="--",
    label="PINN (L-BFGS)",
)
plt.xlabel("x")
plt.ylabel("u(x)")
plt.title("1D Poisson — solution")
plt.legend()
plt.grid(True)
plt.savefig("figs/1D_poisson_solution.pdf")

plt.figure()
plt.semilogy(losses_adam, label="Adam")
plt.semilogy(losses_lbfgs, label="L-BFGS")
plt.xlabel("step")
plt.ylabel("physics loss")
plt.title("Training loss")
plt.legend()
plt.grid(True)
plt.savefig("figs/1D_poisson_loss.pdf")

plt.figure()
plt.plot(x_eval, jnp.abs(u_pred_adam - u_true), label="|err| Adam")
plt.plot(x_eval, jnp.abs(u_pred_lbfgs - u_true), label="|err| L-BFGS")
plt.xlabel("x")
plt.ylabel("|error|")
plt.title("Absolute error")
plt.legend()
plt.grid(True)
plt.savefig("figs/1D_poisson_abs_error.pdf")

plt.show()