# Author: David Castrejon
# ENG 605.745.81 Reasoning Under Uncertainty
# Credal Network with Extreme-Point Marginal Computation

from itertools import product

# ------------------------------------------------------
# Node Class
# ------------------------------------------------------
class CredalNode:
    def __init__(self, name, parents=None):
        self.name = name
        self.parents = parents if parents else []
        self.children = []
        self.credal_table = {}  # Maps parent assignment -> {state: (low, high)}

    def add_child(self, child):
        self.children.append(child)

    def set_credal_table(self, table):
        self.credal_table = table

# ------------------------------------------------------
# Credal Network Class (with detailed prints)
# ------------------------------------------------------
class CredalNetwork:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node):
        self.nodes[node.name] = node
        for parent in node.parents:
            parent.add_child(node)

    # --------------------------------------------------------------
    # Extreme-point marginal computation with debug prints
    # --------------------------------------------------------------
    def compute_marginal_extreme(self, node, indent=0):
        pad = "  " * indent
        print(f"{pad}Computing marginal for node '{node.name}'")

        # Base case
        if not node.parents:
            print(f"{pad}Root node detected, returning credal table directly:")
            for state, (low, high) in node.credal_table[()].items():
                print(f"{pad}  {state}: [{low}, {high}]")
            return {state: (low, high) for state, (low, high) in node.credal_table[()].items()}

        # Compute parent marginals
        parent_marginals = [self.compute_marginal_extreme(p, indent=indent+1) for p in node.parents]

        parent_states = [list(pm.keys()) for pm in parent_marginals]
        node_states = list(next(iter(node.credal_table.values())).keys())

        lower_bounds = {s: float('inf') for s in node_states}
        upper_bounds = {s: float('-inf') for s in node_states}

        # Enumerate all combinations of parent extreme points
        parent_extreme_options = []
        for idx, pm in enumerate(parent_marginals):
            extremes = []
            states = list(pm.keys())
            for choice in product(*[(l, u) for l, u in pm.values()]):
                extreme_dict = {s: val for s, val in zip(states, choice)}
                total = sum(extreme_dict.values())
                if total > 0:
                    extreme_dict = {s: v / total for s, v in extreme_dict.items()}
                extremes.append(extreme_dict)
            parent_extreme_options.append(extremes)
            print(f"{pad}Parent '{node.parents[idx].name}' extreme points:")
            for e in extremes:
                print(f"{pad}  {e}")

        # Iterate over all combinations of parent extremes
        for parent_extremes_combo in product(*parent_extreme_options):
            # Compute all joint parent assignments
            for parent_assignment in product(*[list(d.keys()) for d in parent_extremes_combo]):
                prob = 1.0
                for i, state in enumerate(parent_assignment):
                    prob *= parent_extremes_combo[i][state]

                credal_entry = node.credal_table[parent_assignment]

                for state in node_states:
                    low, high = credal_entry[state]
                    lower_bounds[state] = min(lower_bounds[state], prob * low) if lower_bounds[state] != float('inf') else prob * low
                    upper_bounds[state] = max(upper_bounds[state], prob * high) if upper_bounds[state] != float('-inf') else prob * high

                print(f"{pad}Parent assignment {parent_assignment}, prob={prob:.4f}, node bounds updated:")
                for state in node_states:
                    print(f"{pad}  {state}: [{lower_bounds[state]:.4f}, {upper_bounds[state]:.4f}]")

        for s in node_states:
            if lower_bounds[s] == float('inf'):
                lower_bounds[s] = 0.0
            if upper_bounds[s] == float('-inf'):
                upper_bounds[s] = 1.0

        print(f"{pad}Final marginal for '{node.name}':")
        for state in node_states:
            print(f"{pad}  {state}: [{lower_bounds[state]:.4f}, {upper_bounds[state]:.4f}]")
        print()

        return {s: (lower_bounds[s], upper_bounds[s]) for s in node_states}

    # ----------------------------------------------------------------------
    # Propagate to all nodes with pretty printing
    # ----------------------------------------------------------------------
    def propagate(self):
        print("Starting propagation through credal network...\n")
        marginals = {}
        for node in self.nodes.values():
            marginals[node.name] = self.compute_marginal_extreme(node)
        print("Propagation complete.\n")
        print("=== Final Marginal Bounds (Extreme-Point Method) ===")
        for node, bounds in marginals.items():
            print(f"{node}:")
            for state, (low, high) in bounds.items():
                print(f"  {state}: [{low:.4f}, {high:.4f}]")
            print()
        return marginals

# -----------------------------------------------------------------
# Section 4 — Medical Diagnosis Example
# -----------------------------------------------------------------
def medical_diagnosis_example():
    # Nodes
    flu = CredalNode("Flu")  # root node
    cold = CredalNode("Cold")  # root node
    fever = CredalNode("Fever", parents=[flu, cold])
    cough = CredalNode("Cough", parents=[flu, cold])

    # Credal tables 
    # Root nodes prior probabilities
    flu.set_credal_table({(): {'Present': (0.05, 0.1), 'Absent': (0.9, 0.95)}})
    cold.set_credal_table({(): {'Present': (0.1, 0.2), 'Absent': (0.8, 0.9)}})

    # Symptoms conditional on diseases
    fever.set_credal_table({
        ('Present', 'Present'): {'Present': (0.9, 1.0), 'Absent': (0.0, 0.1)},
        ('Present', 'Absent'): {'Present': (0.8, 0.95), 'Absent': (0.05, 0.2)},
        ('Absent', 'Present'): {'Present': (0.7, 0.85), 'Absent': (0.15, 0.3)},
        ('Absent', 'Absent'): {'Present': (0.0, 0.1), 'Absent': (0.9, 1.0)}
    })

    cough.set_credal_table({
        ('Present', 'Present'): {'Present': (0.8, 0.95), 'Absent': (0.05, 0.2)},
        ('Present', 'Absent'): {'Present': (0.6, 0.8), 'Absent': (0.2, 0.4)},
        ('Absent', 'Present'): {'Present': (0.5, 0.7), 'Absent': (0.3, 0.5)},
        ('Absent', 'Absent'): {'Present': (0.0, 0.1), 'Absent': (0.9, 1.0)}
    })

    # Build network
    cn = CredalNetwork()
    cn.add_node(flu)
    cn.add_node(cold)
    cn.add_node(fever)
    cn.add_node(cough)

    # Propagate
    marginals = cn.propagate()

    # Print results
    print("=== Credal Network Marginal Bounds for Medical Diagnosis ===")
    for node, bounds in marginals.items():
        print(f"{node}:")
        for state, (low, high) in bounds.items():
            print(f"  {state}: [{low:.4f}, {high:.4f}]")
        print()

    print("\n=== End of Example ===\n")


# --------------------------------
# Entry Point
# --------------------------------
if __name__ == "__main__":
    medical_diagnosis_example()
