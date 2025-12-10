# In __init__.py
import line_solver
from line_solver import *
import networkx as nx

__all__ = [
    'ServiceTopology', 'User', 'Host', 'Container', 'Service', 'ServiceStep', 'SynchCaller',
    'opt_lqn_2', 'opt_lqn_4', 'opt_lqn_8', 'opt_lqn_20', 'stoch_opt_lqn_20'
]

class ServiceTopology(line_solver.LayeredNetwork):
    def __init__(self, name):
        super().__init__(name)
        self.avgTable = None
        GlobalConstants.setVerbose(VerboseLevel.SILENT)

    def solve(self):
        self.avgTable = SolverLQNS(self).getAvgTable()

class User(line_solver.Processor):
    def __init__(self, model, name, mult):
        super().__init__(model, name, mult, SchedStrategy.REF)


class Host(line_solver.Processor):
    def __init__(self, model, name, cores=1, sched=SchedStrategy.PS):
        super().__init__(model, name, int(cores), sched)

class Container(line_solver.Task):
    def __init__(self, model, name, vcores=1, sched=SchedStrategy.FCFS):
        super().__init__(model, name, vcores, sched)
    def set_affinity(self, host):
        self.affinity(host)

class Service(line_solver.Entry):
    def __init__(self, model, container, name, svcTime):
        super().__init__(model, name)
        self.computeStep = ServiceStep(model, name + "_compute", svcTime).repliesTo(self)
        self.on(container)
        self.computeStep.boundTo(self)
        self.computeStep.on(container)

    def getName(self):
        return self.obj.getName()


class ServiceStep(line_solver.Activity):
    def __init__(self, model, name, distrib):
        super().__init__(model, name, distrib)


class SynchCaller(line_solver.Task):
    def __init__(self, model, name, mult=1, thinkTime=Immediate()):
        super().__init__(model, name, mult, SchedStrategy.REF)
        self.model = model
        self.name = name
        self.mult = mult
        self.userThink = thinkTime
        self.setThinkTime(thinkTime)
        self.refhost = Host(model, "callerhost_" + name, mult)
        self.on(self.refhost)

    def init_dag(self):
        graph = nx.DiGraph()
        return graph

    def set_dag(self, G):
        self.entry = Entry(self.model, self.name + "_dag")
        self.entry.on(self)
        self.act = {}
        for u, d in G.in_degree():
            self.act[u] = ServiceStep(self.model, u.getName() + "_call", Exp(1e8)).on(self).synchCall(u, 1.0)
            if d == 0:  # dag root is the only node with zero in-degree
                self.act[u].boundTo(self.entry)
        for u, v, a in G.edges(data=True):
            self.addPrecedence(line_solver.ActivityPrecedence.Serial(self.act[u], self.act[v]))

    def tput(self):
        return self.model.avgTable[self.model.avgTable['Node'] == self.name]['Tput'].item()

    def respt(self):
        return self.model.avgTable[self.model.avgTable['Node'] == self.name]['RespT'].item()

    def residt(self):
        return self.model.avgTable[self.model.avgTable['Node'] == self.name]['ResidT'].item()

    def qlen(self):
        return self.model.avgTable[self.model.avgTable['Node'] == self.name]['QLen'].item()

    def util(self):
        return self.model.avgTable[self.model.avgTable['Node'] == self.name]['Util'].item()


# class AsynchCaller():
#     def __init__(self, model, name, arvProc):
#         self.model = model
#         self.name = name
#         self.arvProc = arvProc
#         self.model.setCaller(self)
# stream = AsynchCaller(topology, "serviceUsers", Exp(0.1))

try:
    from .problems import opt_lqn_2, opt_lqn_4, opt_lqn_8, opt_lqn_20, stoch_opt_lqn_20
except ImportError as e:
    print(f"Warning: Could not import problems: {e}")