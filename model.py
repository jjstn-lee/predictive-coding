from jax import numpy as jnp, random
from jax import jit
from ngclearn import Context
from ngclearn.utils import JaxProcess
from ngcsimlib.compilers.process import Process
from ngclearn.components import RateCell, HebbianSynapse
import ngclearn.utils.weight_distribution as dist

## create seeding keys
dkey = random.PRNGKey(1234)
dkey, *subkeys = random.split(dkey, 4)

## create simple dynamical system: a --> w_ab --> b
with Context("model") as model:
    a = RateCell(name="a", n_units=1, tau_m=0.0, act_fx="identity", key=subkeys[0])
    b = RateCell(name="b", n_units=1, tau_m=20.0, act_fx="identity", key=subkeys[1])
    Wab = HebbianSynapse(
        name="Wab", shape=(1, 1), weight_init=dist.constant(value=1.0), key=subkeys[2]
    )
    ## wire a to w_ab and wire w_ab to b
    Wab.inputs << a.zF
    b.j << Wab.outputs

    ## configure desired commands for simulation object
    reset_process = (JaxProcess() >> a.reset >> Wab.reset >> b.reset)
    model.wrap_and_add_command(jit(reset_process.pure), name="reset")
    advance_process = (
        JaxProcess() >> a.advance_state >> Wab.advance_state >> b.advance_state
    )
    model.wrap_and_add_command(jit(advance_process.pure), name="advance")

    ## set up clamp as a non-compiled utility commands
    @Context.dynamicCommand
    def clamp(x):
        a.j.set(x)

## run some data through our simple dynamical system
x_seq = jnp.asarray([[1., 2., 3., 4., 5.]], dtype=jnp.float32)

model.reset()
for ts in range(x_seq.shape[1]):
    x_t = jnp.expand_dims(x_seq[0, ts], axis=0)  ## get data at time ts
    model.clamp_data(x_t)
    model.advance(t=ts * 1., dt=1.)
    ## naively extract simple statistics at time ts and print them to I/O
    a_out = a.zF
    b_out = b.zF
    print(" {}: a.zF = {} ~> b.zF = {}".format(ts, a_out, b_out))