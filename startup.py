from gurobipy import*
m = Model('Protorype example_type1')

x_1 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x_1')
x_2 = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='x_2')

m.update()

m.setObjective(3*x_1+5*x_2, GRB.MAXIMIZE)
m.addConstr(x_1 <= 4, 'c0')
m.addConstr(2*x_2 <= 12, 'c1')
m.addConstr(3*x_1+2*x_2 <= 18, 'c2')

m.optimize()
