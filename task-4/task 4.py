from pulp import LpMaximize, LpProblem, LpVariable, LpStatus, value

# Define the optimization problem
model = LpProblem(name="profit-maximization", sense=LpMaximize)

# Decision variables
x = LpVariable(name="Product_A", lowBound=0)
y = LpVariable(name="Product_B", lowBound=0)

# Constraints
model += (2 * x + 1 * y <= 100, "Labor_Constraint")
model += (1 * x + 1 * y <= 80, "Material_Constraint")

# Objective function
model += 40 * x + 30 * y

# Solve
model.solve()

# Output
print("----- Optimal Production Plan -----")
print(f"Produce {x.value():.2f} units of Product A")
print(f"Produce {y.value():.2f} units of Product B")
print(f"Maximum Profit: ${value(model.objective):.2f}")

print("\n--- Solver Status ---")
print(f"Status: {LpStatus[model.status]}")

print("\n--- Constraints ---")
for name, constraint in model.constraints.items():
    rhs = -constraint.constant
    used = rhs - constraint.slack
    print(f"{name}: Used {used:.2f} / Limit {rhs}")
