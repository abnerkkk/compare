import numpy as np
import line_solver
from line_solver import *

__all__ = ['opt_lqn_2', 'opt_lqn_4', 'opt_lqn_8', 'opt_lqn_20', 'stoch_opt_lqn_20']


def _get_vesta_classes():
    """Late import to avoid circular dependency."""
    import vesta
    return vesta.ServiceTopology, vesta.Host, vesta.Container, vesta.Service, vesta.SynchCaller

    # •	ServiceTopology：整个 LQN 模型容器（JLINE 的 LayeredNetwork / LQN model）
	# •	Host：更接近 LQN 的 processor（有 cores/multiplicity、调度策略 ps 等）
	# •	Container：更接近 LQN 的 task（部署在 processor 上）
	# •	Service：更接近 LQN 的 entry + activity（vesta 会帮你生成 entry，并生成一个 compute activity，比如 service1_compute）
	# •	SynchCaller：ref task/用户类（参考任务），代表外部请求源；它会产生一个“caller host + ref task”，并用 DAG 表达同步调用链

# 2-variable throughput maximization
def opt_lqn_2(x, display=False):
    ServiceTopology, Host, Container, Service, SynchCaller = _get_vesta_classes()
    topology = ServiceTopology("topology")
    host1 = Host(topology, "host1")
    host2 = Host(topology, "host2")
    container1 = Container(topology, "container1")
    container2 = Container(topology, "container2")
    service1 = Service(topology, container1, "service1", Exp(0.001 + x[0]))
    service2 = Service(topology, container2, "service2", Erlang(0.001 + x[1], 2))
    users = SynchCaller(topology, "serviceUsers", 1, Exp(1))
    graph = users.init_dag()
    graph.add_edges_from([(service1, service2)])
    users.set_dag(graph)
    container1.on(host1)
    container2.on(host2)
    topology.solve()
    # throughput
    t = users.tput()
    if display:
        print(t)
    return t


# 4-variable mixed-integer throughput maximization
# integrality=(0,0,1,1)
def opt_lqn_4(x, display=False):
    ServiceTopology, Host, Container, Service, SynchCaller = _get_vesta_classes()
    topology = ServiceTopology("topology")
    host1 = Host(topology, "host1", cores=x[3])
    host2 = Host(topology, "host2")
    container1 = Container(topology, "container1")
    container2 = Container(topology, "container2")
    service1 = Service(topology, container1, "service1", Exp(0.001 + x[0]))
    service2 = Service(topology, container2, "service2", Erlang(0.001 + x[1], 2))
    users = SynchCaller(topology, "serviceUsers", 1, Exp(1))
    graph = users.init_dag()
    graph.add_edges_from([(service1, service2)])
    users.set_dag(graph)
    if x[2] == 1:
        container1.on(host1)
        container2.on(host2)
    else:
        container2.on(host1)
        container1.on(host2)
    topology.solve()
    t = users.tput()
    if display:
        print(t)
    return t


# 8-variable mixed-integer weighted throughput maximization
# integrality=(0,0,0,0,1,1,1,1)
def opt_lqn_8(x, display=False):
    ServiceTopology, Host, Container, Service, SynchCaller = _get_vesta_classes()
    topology = ServiceTopology("topology")
    host1 = Host(topology, "host1", cores=x[4])
    host2 = Host(topology, "host2", cores=x[5])
    host3 = Host(topology, "host3", cores=x[6])
    host4 = Host(topology, "host4", cores=x[7])

    container1 = Container(topology, "container1")
    container2 = Container(topology, "container2")
    container3 = Container(topology, "container3")
    container4 = Container(topology, "container4")

    service1 = Service(topology, container1, "service1", Exp(0.001 + x[0]))
    service2 = Service(topology, container2, "service2", Exp(0.001 + x[1]))
    service3 = Service(topology, container3, "service3", Exp(0.001 + x[2]))
    service4 = Service(topology, container4, "service4", Exp(0.001 + x[3]))

    users1 = SynchCaller(topology, "serviceUsers1", 1, Exp(1))
    graph1 = users1.init_dag()
    graph1.add_edges_from([(service1, service2)])
    users1.set_dag(graph1)

    users2 = SynchCaller(topology, "serviceUsers2", 1, Exp(1))
    graph2 = users2.init_dag()
    graph2.add_edges_from([(service3, service4)])
    users2.set_dag(graph2)

    container1.on(host1)
    container2.on(host2)
    container3.on(host3)
    container4.on(host4)

    topology.solve()

    t = 0.3 * users1.tput() + 0.7 * users2.tput() - 4.0 * x[4] - 5.0 * x[5] - 6.0 * x[6] - 7.0 * x[7]
    if display:
        print(t)
    return t


# 20-variable mixed-integer weighted throughput maximization
# integrality=(0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1)
def opt_lqn_20(x, display=False):
    ServiceTopology, Host, Container, Service, SynchCaller = _get_vesta_classes()
    topology = ServiceTopology("topology")
    host1 = Host(topology, "host1", cores=x[10])
    host2 = Host(topology, "host2", cores=x[11])
    host3 = Host(topology, "host3", cores=x[12])
    host4 = Host(topology, "host4", cores=x[13])
    host5 = Host(topology, "host5", cores=x[14])
    host6 = Host(topology, "host6", cores=x[15])
    host7 = Host(topology, "host7", cores=x[16])
    host8 = Host(topology, "host8", cores=x[17])
    host9 = Host(topology, "host9", cores=x[18])
    host10 = Host(topology, "host10", cores=x[19])

    container1 = Container(topology, "container1")
    container2 = Container(topology, "container2")
    container3 = Container(topology, "container3")
    container4 = Container(topology, "container4")
    container5 = Container(topology, "container5")
    container6 = Container(topology, "container6")
    container7 = Container(topology, "container7")
    container8 = Container(topology, "container8")
    container9 = Container(topology, "container9")
    container10 = Container(topology, "container10")

    service1 = Service(topology, container1, "service1", Exp(0.001 + x[0]))
    service2 = Service(topology, container2, "service2", Exp(0.001 + x[1]))
    service3 = Service(topology, container3, "service3", Exp(0.001 + x[2]))
    service4 = Service(topology, container4, "service4", Exp(0.001 + x[3]))
    service5 = Service(topology, container5, "service5", Exp(0.001 + x[4]))
    service6 = Service(topology, container6, "service6", Exp(0.001 + x[5]))
    service7 = Service(topology, container7, "service7", Exp(0.001 + x[6]))
    service8 = Service(topology, container8, "service8", Exp(0.001 + x[7]))
    service9 = Service(topology, container9, "service9", Exp(0.001 + x[8]))
    service10 = Service(topology, container10, "service10", Exp(0.001 + x[9]))

    users1 = SynchCaller(topology, "serviceUsers1", 1, Exp(1))
    graph1 = users1.init_dag()
    graph1.add_edges_from([(service1, service2)])
    users1.set_dag(graph1)

    users2 = SynchCaller(topology, "serviceUsers2", 1, Exp(1))
    graph2 = users2.init_dag()
    graph2.add_edges_from([(service3, service4)])
    users2.set_dag(graph2)

    users3 = SynchCaller(topology, "serviceUsers3", 1, Exp(1))
    graph3 = users3.init_dag()
    graph3.add_edges_from([(service5, service6)])
    users3.set_dag(graph3)

    users4 = SynchCaller(topology, "serviceUsers4", 1, Exp(1))
    graph4 = users4.init_dag()
    graph4.add_edges_from([(service7, service8)])
    users4.set_dag(graph4)

    users5 = SynchCaller(topology, "serviceUsers5", 1, Exp(1))
    graph5 = users5.init_dag()
    graph5.add_edges_from([(service9, service10)])
    users5.set_dag(graph5)

    container1.on(host1)
    container2.on(host2)
    container3.on(host3)
    container4.on(host4)
    container5.on(host5)
    container6.on(host6)
    container7.on(host7)
    container8.on(host8)
    container9.on(host9)
    container10.on(host10)

    topology.solve()

    w = (0.1,0.1,0.2,0.3,0.2)
    c = np.linspace(1.0, 10.0, 10)
    t = 0.
    t = t + w[0] * users1.tput()
    t = t + w[1] * users2.tput()
    t = t + w[2] * users3.tput()
    t = t + w[3] * users4.tput()
    t = t + w[4] * users5.tput()
    for i in range(10):
        t = t - c[i] * x[i + 10]
    if display:
        print(t)
    return t

def stoch_opt_lqn_20(x, display=False):
    t = opt_lqn_20(x, display)
    t = t * (1+0.10 * np.random(len()))