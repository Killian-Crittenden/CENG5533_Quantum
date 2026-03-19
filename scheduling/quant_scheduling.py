import dimod
import neal
import networkx as nx
import time
from scheduling_tester import build_dag_test_suite

def text_pretty_print(G):
    print("--- Graph Adjacency List ---")
    for line in nx.generate_network_text(G):
        print(line)

def solve_optimized_m_machine_scheduling(dag=None):
    #Define the DAG
    G = nx.DiGraph()
    #G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
    if dag == None:
        G.add_edges_from([(0, 1), (0, 2), (0, 6),
                       (1, 3), (1, 4), 
                       (3, 7), 
                       (2, 5),(5, 8), (8, 10), 
                       (6, 9),
                       (7, 10), (4, 10), (5, 8), (8, 10), (9, 10)])
    else:
        G = dag
    
    text_pretty_print(G)
    
    tasks = list(G.nodes)
    N = len(tasks)
    num_machines = 10 
    
    bqm = dimod.BinaryQuadraticModel('BINARY')
    
    # Weights
    gamma = 10.0  # Hard constraint penalty (do not break rules)
    alpha = 0.25   # Soft objective penalty (try to finish early)

    # Each task must be scheduled exactly once
    for i in tasks:
        bqm.add_linear_equality_constraint(
            [(f"{i}_{t}", 1.0) for t in range(N)],
            constant=-1.0,
            lagrange_multiplier=gamma
        )

    #Up to 'm' tasks per time step
    for t in range(N):
        step_vars = [(f"{i}_{t}", 1.0) for i in tasks]
        slack_vars = [(f"slack_t{t}_{k}", 1.0) for k in range(num_machines)]
        
        bqm.add_linear_equality_constraint(
            step_vars + slack_vars,
            constant=-num_machines,
            lagrange_multiplier=gamma
        )

    #u must finish before v starts
    for u, v in G.edges:
        for t_u in range(N):
            for t_v in range(N):
                if t_u >= t_v:
                    bqm.add_interaction(f"{u}_{t_u}", f"{v}_{t_v}", gamma)

    #Minimize time
    #Add a small energy penalty for every time step a task is delayed.
    for i in tasks:
        for t in range(N):
            bqm.add_linear(f"{i}_{t}", t * t * alpha)

    #Solve the BQM
    print(f"Optimizing schedule for {len(tasks)} tasks on {num_machines} machines...")
    sampler = neal.SimulatedAnnealingSampler()
    
    sampleset = sampler.sample(bqm, num_reads=5000)
    
    best_sample = sampleset.first.sample
    energy = sampleset.first.energy

    # Parse Output
    print(f"\nLowest Energy Found: {energy:.2f}")
    
    # Calculate what the energy should be if zero rules are broken.
    # It will equal the sum of (scheduled time * alpha) for all tasks.
    
    results = []
    for i in tasks:
        for t in range(N):
            if best_sample[f"{i}_{t}"] == 1:
                results.append((t, i))
    
    schedule = {}
    objective_score = 0
    for t, i in results:
        schedule.setdefault(t, []).append(i)
        objective_score += (t * t * alpha)

    # Check if the lowest energy found is essentially just the objective score
    # (Allowing a tiny bit of floating point tolerance)
    # if energy > objective_score + 0.01:
    #     print("WARNING: Schedule violates constraints (Energy is too high).")
    # else:
    #     print("\nOptimal Compressed Schedule:")
    #     for t in sorted(schedule.keys()):
    #         print(f"Time {t}: Tasks {schedule[t]}")
    for t in sorted(schedule.keys()):
            print(f"Time {t}: Tasks {schedule[t]}")

if __name__ == "__main__":
    for name, dag in build_dag_test_suite().items():
        start_time = time.perf_counter()
        solve_optimized_m_machine_scheduling(dag)
        print(f'{name} took {(time.perf_counter()-start_time):.2f} seconds to find a soltion')