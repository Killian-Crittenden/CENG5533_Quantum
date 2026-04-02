import dimod
import neal
import networkx as nx
import time
import math
import shutil
from scheduling_tester import build_dag_test_suite

import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout

def visualize_machine_dag(G, machine_assignments, title="Task Dependency Graph"):
    """
    Hierarchical DAG visualization. 
    Uses G.graph attributes to force top-to-bottom flow and spacing.
    """
    # 1. Canvas Adjustment: Tall for vertical flow, high DPI for clarity
    plt.figure(figsize=(10, 9)) 

    # 2. Attribute Injection: Set spacing and direction in the graph itself
    # This avoids the "unexpected keyword argument 'args'" error
    G.graph['graph'] = {
        'rankdir': 'TB',   # Top-to-Bottom
        'ranksep': '2.0',  # Vertical space between layers
        'nodesep': '1.0'   # Horizontal space between nodes
    }

    try:
        # Use dot engine; attributes are read from G.graph automatically
        pos = graphviz_layout(G, prog='dot')
    except Exception as e:
        print(f"Graphviz Error: {e}. Falling back to spring layout.")
        pos = nx.spring_layout(G, k=0.5)

    # 3. Machine Color Palette
    machine_colors = [
        '#FFADAD', '#A0C4FF', '#CAFFBF', '#FFD6A5', '#BDB2FF', 
        '#FDFFB6', '#9BF6FF', '#FFC6FF', '#D4A373', '#E0E0E0'
    ]
    
    node_colors = [machine_colors[machine_assignments.get(node, 9) % 10] for node in G.nodes()]

    # 4. Drawing: Scaled down for high-density 40+ node graphs
    nx.draw_networkx_nodes(G, pos, 
                           node_size=600, 
                           node_color=node_colors, 
                           edgecolors='#424242', 
                           linewidths=1.0)
    
    nx.draw_networkx_edges(G, pos, 
                           edge_color='#BDBDBD', 
                           width=1.0, 
                           arrows=True, 
                           arrowsize=12)

    nx.draw_networkx_labels(G, pos, 
                            font_size=8, 
                            font_weight='bold', 
                            font_family='sans-serif')
    
    # 5. Resource Legend
    unique_machines = sorted(set(machine_assignments.values()))
    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=machine_colors[m % 10], 
                                 markersize=10, label=f'Machine {m}') 
                      for m in unique_machines]
    
    plt.legend(handles=legend_handles, loc='upper right', title="Resource Allocation", fontsize=10)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    
    # 6. Save as PNG for easy scrolling/zooming on Ubuntu
    plt.savefig("optimized_schedule.png", dpi=300, bbox_inches='tight')
    plt.show()

def text_pretty_print(G):
    print("--- Graph Adjacency List ---")
    for line in nx.generate_network_text(G):
        print(line)

def calculate_metrics(dag, optimized_makespan, num_machines):
    # 1. Calculate Critical Path (Theoretical Limit)
    critical_path_len = nx.dag_longest_path_length(dag) + 1 # +1 for node count
    
    # 2. Estimate "Unoptimized" (Serial)
    serial_time = dag.number_of_nodes() 
    
    # 3. Calculate Speedup & Efficiency
    speedup = serial_time / optimized_makespan
    efficiency = speedup / num_machines
    
    # 4. Optimality Gap
    lower_bound = max(critical_path_len, serial_time / num_machines)
    gap = (optimized_makespan - lower_bound) / lower_bound
    
    print(f'"Speedup": {round(speedup, 2)}')
    print(f'"Efficiency": {round(efficiency * 100, 1)}%')
    print(f'"Gap to Bound": {round(gap * 100, 1)}%')

def get_time_windows(G, N):
    est = {node: len(nx.ancestors(G, node)) for node in G.nodes}
    lst = {node: (N - 1) - len(nx.descendants(G, node)) for node in G.nodes}
    return est, lst

def assign_machines(schedule_by_time, G, num_machines):
    """
    Takes a valid time schedule and greedily assigns tasks to machines,
    prioritizing keeping child tasks on the same machine as their parents.
    
    schedule_by_time: dict format {time_slot: [task1, task2, ...]}
    """
    machine_assignments = {} # {task_id: machine_id}
    
    for t in sorted(schedule_by_time.keys()):
        tasks_at_t = schedule_by_time[t]
        busy_machines_this_step = set()
        
        for task in tasks_at_t:
            # Using NetworkX's predecessors method instead of dict iteration
            parents = list(G.predecessors(task))
            placed = False
            
            # Try to stay on a parent's machine
            for p in parents:
                if p in machine_assignments:
                    parent_m = machine_assignments[p]
                    if parent_m not in busy_machines_this_step:
                        machine_assignments[task] = parent_m
                        busy_machines_this_step.add(parent_m)
                        placed = True
                        break
            
            # Fallback to any free machine
            if not placed:
                for m in range(num_machines):
                    if m not in busy_machines_this_step:
                        machine_assignments[task] = m
                        busy_machines_this_step.add(m)
                        break
                        
    return machine_assignments

def solve_optimized_m_machine_scheduling(dag=None, graph_name='Test'):
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

    #text_pretty_print(G)
    
    tasks = list(G.nodes)
    N = len(tasks)
    num_machines = 4 

    effective_machines = min(num_machines, N)

    est, lst = get_time_windows(G, N)
    
    bqm = dimod.BinaryQuadraticModel('BINARY')
    
    # Weights
    N = len(G.nodes)
    alpha = 5
    
    # The penalty for missing a task must exceed the worst-case time tax
    max_time_penalty = sum((lst[i]**2) * alpha for i in tasks)
    # Hierarchy: Precedence > Assignment > Capacity
    gamma_assignment = max_time_penalty * 3
    gamma_capacity = gamma_assignment # Slightly "softer" to allow movement
    #gamma_prec = gamma_assignment * 7     # "Hard" constraint
    gamma_prec = gamma_assignment * 2.5     # "Hard" constraint

    active_vars = {}
    for i in tasks:
        active_vars[i] = [t for t in range(est[i], lst[i] + 1)]

    # Each task must be scheduled exactly once
    for i in tasks:
        # Task i can ONLY exist between its EST and LST
        bqm.add_linear_equality_constraint(
            [(f"{i}_{t}", 1.0) for t in active_vars[i]],
            constant=-1.0,
            lagrange_multiplier=gamma_assignment
        )

    #Up to 'm' tasks per time step
    for t in range(N):
        # Only collect variables that are VALID for this specific time step
        step_vars = [(f"{i}_{t}", 1.0) for i in tasks if t in active_vars[i]]
        
        # If there are fewer tasks than machines, no capacity constraint is needed
        if len(step_vars) <= effective_machines:
            continue
            
        # Logarithmic Slack Encoding
        slack_terms = []
        current_sum = 0
        num_slack_bits = math.floor(math.log2(effective_machines)) + 1
        
        for k in range(num_slack_bits):
            weight = 2**k
            if current_sum + weight > effective_machines:
                weight = effective_machines - current_sum
            if weight > 0:
                slack_terms.append((f"slack_t{t}_bit{k}", weight))
                current_sum += weight
        
        bqm.add_linear_equality_constraint(
            step_vars + slack_terms,
            constant=-effective_machines,
            lagrange_multiplier=gamma_capacity
        )

    #u must finish before v starts
    #We generate a fully defined map of every dependency so it is not profitable for the program to break an order constraint
    TC = nx.transitive_closure(G)
    for u, v in TC.edges:
        # u must finish before v starts. 
        # If there is ANY overlap in their windows where t_u >= t_v, we penalize it.
        for t_u in active_vars[u]:
            for t_v in active_vars[v]:
                if t_u >= t_v:
                    bqm.add_interaction(f"{u}_{t_u}", f"{v}_{t_v}", gamma_prec)

    #Minimize time
    #Add a small energy penalty for every time step a task is delayed.
    for i in tasks:
        for t in active_vars[i]:
            bqm.add_linear(f"{i}_{t}", t * t * alpha)
            #bqm.add_linear(f"{i}_{t}", t * alpha)

    #Solve the BQM
    print(f"Optimizing schedule for {len(tasks)} tasks on {num_machines} machines...")
    sampler = neal.SimulatedAnnealingSampler()
    
    #sampleset = sampler.sample(bqm, num_reads=5000, num_sweeps=2000)
    sampleset = sampler.sample(bqm, num_reads=5000, num_sweeps=25000, beta_range=[0.01, 100])
    
    best_sample = sampleset.first.sample
    energy = sampleset.first.energy

    # Parse Output
    print(f"\nLowest Energy Found: {energy:.2f}")
    
    # Calculate what the energy should be if zero rules are broken.
    # It will equal the sum of (scheduled time * alpha) for all tasks.
    
    results = []
    for i in tasks:
        for t in active_vars[i]:
            if best_sample[f"{i}_{t}"] == 1:
                results.append((t, i))
    
    schedule = {}
    objective_score = 0
    for t, i in results:
        schedule.setdefault(t, []).append(i)
        objective_score += ((t**2) * alpha)

    # Check if the lowest energy found is essentially just the objective score
    # (Allowing a tiny bit of floating point tolerance)
    if energy > objective_score + 0.01:
        print("WARNING: Schedule violates constraints (Energy is too high).")
    # else:
    #     print("\nOptimal Compressed Schedule:")
    #     for t in sorted(schedule.keys()):
    #         print(f"Time {t}: Tasks {schedule[t]}")
    for t in sorted(schedule.keys()):
            print(f"Time {t}: Tasks {schedule[t]}")

    print(f'Unoptimized Serial Runtime: {dag.number_of_nodes()} Units')
    print(f'Optimized Runtime: {len(schedule.keys())} Units')
    calculate_metrics(dag, len(schedule.keys()), num_machines)
    machine_assignment = assign_machines(schedule, dag, num_machines)
    visualize_machine_dag(dag, machine_assignment, graph_name)

if __name__ == "__main__":
    #solve_optimized_m_machine_scheduling()
    for name, dag in build_dag_test_suite().items():
        start_time = time.perf_counter()
        solve_optimized_m_machine_scheduling(dag, name)
        print(f'{name} took {(time.perf_counter()-start_time):.2f} seconds to find a soltion')