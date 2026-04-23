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

def get_machine_network():
    """
    Defines the 4-machine diamond topology.
    Mapping: Old 0->0, Old 2->1, Old 3->2, Old 4->3
    """
    M_net = nx.Graph()
    # Machine 0 connected to 1 (old 2) and 2 (old 3)
    # 1 and 2 connected to 3 (old 4)
    M_net.add_edges_from([
        (0, 1), (0, 2), 
        (1, 3), (2, 3)
    ])
    
    # All-pairs shortest path for hop distances
    dists = dict(nx.all_pairs_shortest_path_length(M_net))
    return dists

def visualize_machine_dag(G, machine_assignments, title="Task Dependency Graph"):
    """
    Hierarchical DAG visualization with fixed machine color mapping.
    """
    plt.figure(figsize=(10, 9)) 

    # 1. Attribute Injection for Layout
    G.graph['graph'] = {
        'rankdir': 'TB',
        'ranksep': '2.0',
        'nodesep': '1.0'
    }

    try:
        pos = graphviz_layout(G, prog='dot')
    except Exception as e:
        print(f"Graphviz Error: {e}. Falling back to spring layout.")
        pos = nx.spring_layout(G, k=0.5)

    # 2. Hard-coded Machine Color Palette
    # 0: Blue, 1: Green, 2: Red, 3: Yellow
    machine_colors = {
        0: '#A0C4FF', # Blue
        1: '#CAFFBF', # Green
        2: '#FFADAD', # Red
        3: '#FDFFB6'  # Yellow
    }
    
    # Map nodes to colors; fallback to gray if machine ID is missing
    node_colors = [machine_colors.get(machine_assignments.get(node), '#E0E0E0') for node in G.nodes()]

    # 3. Drawing
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
    
    # 4. Resource Legend (Fixed to Machines 0-3)
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', 
                   markerfacecolor=machine_colors[m], 
                   markersize=10, label=f'Machine {m}') 
        for m in sorted(machine_colors.keys())
    ]
    
    plt.legend(handles=legend_handles, loc='upper right', title="Resource Allocation", fontsize=10)
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.axis('off')
    
    plt.savefig(f"{title}_graph.png", dpi=300, bbox_inches='tight')
    #plt.show()

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

def apply_fixed_schedule_mapping(raw_assignments, dag, m_dists, num_machines):
    """
    Strictly respects BQM timesteps while ensuring machine uniqueness.
    """
    machine_mapping = {}

    # 1. Sort tasks primarily by time, secondarily by node ID for stability
    sorted_tasks = sorted(dag.nodes(), key=lambda x: (raw_assignments[x], x))

    for task in sorted_tasks:
        target_time = raw_assignments[task]
        
        # 2. Find machines already assigned at THIS EXACT time in this loop
        # We check the machine_mapping we are currently building
        occupied_now = {
            machine_mapping[other] 
            for other in machine_mapping 
            if raw_assignments[other] == target_time
        }

        # 3. Filter for only available machines
        available = [m for m in range(num_machines) if m not in occupied_now]

        if not available:
            # Fallback if BQM over-schedules a single timestep
            machine_mapping[task] = 0
            continue

        # 4. Of the available machines, which one has the minimum hop cost from parents?
        # This uses the diamond network distances (m_dists)
        best_m = min(
            available,
            key=lambda m: sum(
                m_dists[machine_mapping[p]][m] 
                for p in dag.predecessors(task) 
                if p in machine_mapping
            ),
            default=available[0]
        )
        
        machine_mapping[task] = best_m

    return raw_assignments, machine_mapping

def apply_est_compression(raw_assignments, dag, num_machines):
    """
    Compresses the schedule and dynamically assigns machines to avoid bottlenecks.
    """
    compressed_times = {} 
    machine_mapping = {}
    
    # Track when each machine is free {machine_id: free_time}
    machine_ready_time = {m: 0 for m in range(num_machines)}
    
    # Process tasks in topological order to respect precedence
    for task in nx.topological_sort(dag):
        
        # 1. Precedence: Find when all parent tasks are finished
        parent_finish = [compressed_times[p] + 1 for p in dag.predecessors(task)]
        dep_ready = max(parent_finish, default=0)
        
        # 2. Find the earliest possible time a machine is available
        best_start_time = float('inf')
        
        # Check all machines to find the earliest slot >= dep_ready
        for m in range(num_machines):
            start_time = max(dep_ready, machine_ready_time[m])
            if start_time < best_start_time:
                best_start_time = start_time
                
        # 3. Find ALL machines that can start the task at this 'best_start_time'
        tied_machines = [m for m in range(num_machines) if max(dep_ready, machine_ready_time[m]) == best_start_time]
        
        # 4. Affinity: Try to keep the task on the same machine as one of its parents
        parent_machines = [machine_mapping[p] for p in dag.predecessors(task) if p in machine_mapping]
        chosen_machine = next((m for m in parent_machines if m in tied_machines), None)
        
        # If no parent machine is free at the exact best time, pick the first free one
        if chosen_machine is None:
            chosen_machine = tied_machines[0]
            
        # 5. Lock in the assignment
        compressed_times[task] = best_start_time
        machine_mapping[task] = chosen_machine
        machine_ready_time[chosen_machine] = best_start_time + 1

    return compressed_times, machine_mapping

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
    alpha = 4
    
    # The penalty for missing a task must exceed the worst-case time tax
    # max_time_penalty = max((lst[i]**2) * alpha for i in tasks)
    # max_time_penalty = max((lst[i]) * alpha for i in tasks)*1.5
    # # Hierarchy: Precedence > Assignment > Capacity
    # # gamma_assignment = max_time_penalty * 3
    # gamma_assignment = max_time_penalty
    # # gamma_capacity = gamma_assignment # Slightly "softer" to allow movement
    # gamma_capacity = gamma_assignment * .8 # Slightly "softer" to allow movement
    # gamma_prec = gamma_assignment * 2    # "Hard" constraint

    max_obj = sum(lst[i] * alpha for i in tasks)

    gamma_assignment = 5 * max_obj
    gamma_capacity   = 7.5 * max_obj
    gamma_prec       = 10 * max_obj

    # max_single_move_savings = (N**2) * alpha 
    
    # # Base penalty just needs to exceed the maximum possible temptation to cheat
    # gamma_base = max_single_move_savings + (alpha * 2) 
    
    # # Keep multipliers tight and close together
    # gamma_assignment = gamma_base
    # gamma_capacity = gamma_base 
    # gamma_prec = gamma_base * 1.5  # Give precedence the highest priority

    active_vars = {}
    for i in tasks:
        active_vars[i] = [t for t in range(est[i], lst[i] + 1)]

    # --- MAKESPAN VARIABLES ---
    C_vars = [f"C_{t}" for t in range(N)]

    # Exactly one makespan time must be selected
    bqm.add_linear_equality_constraint(
        [(c, 1.0) for c in C_vars],
        constant=-1.0,
        lagrange_multiplier=gamma_assignment
    )

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
        active_at_t = [f"{i}_{t}" for i in tasks if t in active_vars[i]]

        for i in range(len(active_at_t)):
            for j in range(i + effective_machines, len(active_at_t)):
                bqm.add_interaction(active_at_t[i], active_at_t[j], gamma_capacity)
    # for t in range(N):
    #     # Only collect variables that are VALID for this specific time step
    #     step_vars = [(f"{i}_{t}", 1.0) for i in tasks if t in active_vars[i]]
        
    #     # If there are fewer tasks than machines, no capacity constraint is needed
    #     if len(step_vars) <= effective_machines:
    #         continue
            
    #     # Logarithmic Slack Encoding
    #     slack_terms = []
    #     current_sum = 0
    #     num_slack_bits = math.floor(math.log2(effective_machines)) + 1
        
    #     for k in range(num_slack_bits):
    #         weight = 2**k
    #         if current_sum + weight > effective_machines:
    #             weight = effective_machines - current_sum
    #         if weight > 0:
    #             slack_terms.append((f"slack_t{t}_bit{k}", weight))
    #             current_sum += weight
        
    #     bqm.add_linear_equality_constraint(
    #         step_vars + slack_terms,
    #         constant=-effective_machines,
    #         lagrange_multiplier=gamma_capacity
    #     )
        

    #u must finish before v starts
    #We generate a fully defined map of every dependency so it is not profitable for the program to break an order constraint
    #TC = nx.transitive_closure(G)
    for u, v in G.edges:
        # u must finish before v starts. 
        # If there is ANY overlap in their windows where t_u >= t_v, we penalize it.
        for t_u in active_vars[u]:
            for t_v in active_vars[v]:
                if t_v <= t_u:
                    bqm.add_interaction(f"{u}_{t_u}", f"{v}_{t_v}", gamma_prec)

    # --- LINK TASK TIMES TO MAKESPAN ---
    for i in tasks:
        for t in active_vars[i]:
            for t_c in range(t):
                # If task is at time t, makespan cannot be earlier than t
                bqm.add_interaction(f"{i}_{t}", f"C_{t_c}", gamma_prec)

    #Minimize time
    #Add a small energy penalty for every time step a task is delayed.
    for i in tasks:
        for t in active_vars[i]:
            # bqm.add_linear(f"{i}_{t}", (t**2) * alpha)
            bqm.add_linear(f"{i}_{t}", t * alpha)
            bqm.add_linear(f"C_{t}", t * alpha)

    #Solve the BQM
    print(f"Optimizing schedule for {len(tasks)} tasks on {num_machines} machines...")
    sampler = neal.SimulatedAnnealingSampler()
    
    #sampleset = sampler.sample(bqm, num_reads=5000, num_sweeps=2000)
    sampleset = sampler.sample(bqm, num_reads=5000, num_sweeps=25000, beta_range=[0.01, 100])
    # sampleset = sampler.sample(bqm, num_reads=5000, num_sweeps=25000)
    
    best_sample = sampleset.first.sample
    energy = sampleset.first.energy

    # Parse Output
    print(f"\nLowest Energy Found: {energy:.2f}")
    
    # Calculate what the energy should be if zero rules are broken.
    # It will equal the sum of (scheduled time * alpha) for all tasks.
    
    # results = []
    # for i in tasks:
    #     for t in active_vars[i]:
    #         if best_sample[f"{i}_{t}"] == 1:
    #             results.append((t, i))

    # 1. Extract raw results
    raw_assignments = {i: t for i in tasks for t in active_vars[i] if best_sample[f"{i}_{t}"] == 1}

    raw_times = {i: next(t for t in range(est[i], lst[i] + 1) 
              if best_sample.get(f"{i}_{t}", 0) > 0.5) for i in tasks}

    # 2. Apply the Compressor
    compressed_times, machine_mapping = apply_est_compression(raw_assignments, dag, num_machines)

    m_dists = get_machine_network()
    final_times, final_machines = apply_fixed_schedule_mapping(
    compressed_times, G, m_dists, num_machines=4
    )

    # 3. Build final schedule for printing/metrics
    final_schedule = {}
    for task, t in compressed_times.items():
        final_schedule.setdefault(t, []).append(task)

    # 4. Final Outputs
    for t in sorted(final_schedule.keys()):
        print(f"Time {t}: Tasks {final_schedule[t]}")
    
    # schedule = {}
    # objective_score = 0
    # for t, i in results:
    #     schedule.setdefault(t, []).append(i)
    #     objective_score += (t * alpha)
    #     #objective_score += ((t**2) * alpha)

    # # Check if the lowest energy found is essentially just the objective score
    # # (Allowing a tiny bit of floating point tolerance)
    # if energy > objective_score + 0.01:
    #     print("WARNING: Schedule violates constraints (Energy is too high).")
    # # else:
    # #     print("\nOptimal Compressed Schedule:")
    # #     for t in sorted(schedule.keys()):
    # #         print(f"Time {t}: Tasks {schedule[t]}")
    # for t in sorted(final_schedule.keys()):
    #         print(f"Time {t}: Tasks {final_schedule[t]}")

    print(f'Unoptimized Serial Runtime: {dag.number_of_nodes()} Units')
    print(f'Optimized Runtime: {len(final_schedule.keys())} Units')
    calculate_metrics(dag, max(compressed_times.values()) + 1, num_machines)
    #machine_assignment = assign_machines(final_schedule, dag, num_machines)
    visualize_machine_dag(dag, final_machines, graph_name)

if __name__ == "__main__":
    #solve_optimized_m_machine_scheduling()
    for name, dag in build_dag_test_suite().items():
        start_time = time.perf_counter()
        solve_optimized_m_machine_scheduling(dag, name)
        print(f'{name} took {(time.perf_counter()-start_time):.2f} seconds to find a soltion')