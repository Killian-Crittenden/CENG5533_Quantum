import networkx as nx
import random

def generate_random_dag(num_nodes, edge_probability):
    """Creates a random DAG by only allowing edges from lower to higher nodes."""
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_probability:
                G.add_edge(i, j)
    return G

def build_dag_test_suite():
    suite = {}

    # Category 1: Pure Chains (Worst-case for parallelization)
    # ---------------------------------------------------------
    # Tasks must be executed exactly one after the other.
    suite['User Graph 1'] = nx.DiGraph()
    suite['User Graph 1'].add_edges_from([(0,1), (0,2), (0,3), (1,4), (2,5), (3,6), (6,8), (3, 7), (4, 5), (5, 8), (4, 9), (9, 10), (8, 10), (7, 10)])
    
    suite['User Graph 2'] = nx.DiGraph()
    suite['User Graph 2'].add_edges_from([(0, 8), (1, 8), (2, 9), (3, 9), (4, 10), (5,10), (6,11), (7,11),
                                        (8, 12), (9, 12), (10, 13), (11, 13), (12, 15), (13, 15), (14, 17),
                                        (15, 16), (16, 17)])
    
    suite['User Graph 3'] = nx.DiGraph()
    suite['User Graph 3'].add_edges_from([(0,1),(1,2),(2,3),(3,10),(1,4),(4,10),(0, 5),(5,6),(6,7),(7,10),(0,8),(8,9),(9,10)])

    suite['01_chain_small'] = nx.path_graph(5, create_using=nx.DiGraph)

    diamond = nx.DiGraph()
    diamond.add_edges_from([(0,1), (0,2), (0,3), (1,4), (2,4), (3,4)])
    suite['08_diamond_simple'] = diamond

    double_diamond = nx.DiGraph()
    double_diamond.add_edges_from([(0,1), (0,2), (1,3), (2,3), (3,4), (3,5), (4,6), (5,6)])
    suite['09_diamond_double'] = double_diamond

    suite['19_complete_dag_small'] = generate_random_dag(8, 1.0)

    suite['04_independent_small'] = nx.empty_graph(10, create_using=nx.DiGraph)

    suite['10_sparse_10_nodes'] = generate_random_dag(10, 0.15)

    suite['14_dense_10_nodes'] = generate_random_dag(10, 0.60)

    suite['06_fork_out'] = nx.star_graph(10)
    suite['06_fork_out'] = nx.DiGraph([(0, i) for i in range(1, 11)]) # 0 unlocks 1-10

    suite['07_fork_in'] = nx.DiGraph([(i, 11) for i in range(1, 11)]) # 1-10 must finish before 11

    suite['02_chain_med'] = nx.path_graph(15, create_using=nx.DiGraph)

    suite['20_complete_dag_med'] = generate_random_dag(15, 1.0)

    suite['15_dense_15_nodes'] = generate_random_dag(15, 0.50)

    suite['16_dense_25_nodes'] = generate_random_dag(25, 0.40)

    suite['11_sparse_25_nodes'] = generate_random_dag(25, 0.10)

    suite['03_chain_large'] = nx.path_graph(30, create_using=nx.DiGraph)

    suite['17_dense_40_nodes'] = generate_random_dag(40, 0.30)


    # Category 2: Pure Parallel (Best-case for parallelization)
    # ---------------------------------------------------------
    # Zero dependencies. Tests basic machine bin-packing.
    #suite['05_independent_large'] = nx.empty_graph(50, create_using=nx.DiGraph)

    # Category 3: Star / Fork Graphs (Map-Reduce style)
    # ---------------------------------------------------------
    # One task unlocks many, or many tasks feed into one.
    
    

    # Category 4: Bottleneck / Diamond Graphs
    # ---------------------------------------------------------
    # Starts small, expands out, then squeezes back into a single task.

    # Category 5: Sparse Random DAGs
    # ---------------------------------------------------------
    # Typical real-world projects. Lots of nodes, relatively few strict dependencies.
    # suite['12_sparse_50_nodes'] = generate_random_dag(50, 0.05)
    # suite['13_sparse_100_nodes'] = generate_random_dag(100, 0.03)

    # Category 6: Dense Random DAGs
    # ---------------------------------------------------------
    # Extremely constrained networks. Very few valid schedules exist.

    # Category 7: Layered / Bipartite-like DAGs
    # ---------------------------------------------------------
    # Tasks grouped in distinct "stages" (like an assembly line).
    layered = nx.DiGraph()
    # Stage 1 (0-2) -> Stage 2 (3-6) -> Stage 3 (7-8)
    for i in range(3):
        for j in range(3, 7):
            layered.add_edge(i, j)
    for j in range(3, 7):
        for k in range(7, 9):
            layered.add_edge(j, k)
    #suite['18_layered_assembly'] = layered

    # Category 8: The "Complete" DAG
    # ---------------------------------------------------------
    # Every node points to every node created after it. Only ONE valid schedule exists.

    return suite

if __name__ == "__main__":
    # Generate the suite
    test_suite = build_dag_test_suite()
    
    # Print a summary of the generated graphs
    print(f"{'Test Name':<25} | {'Nodes':<5} | {'Edges':<5}")
    print("-" * 42)
    for name, graph in test_suite.items():
        # Quick validation to ensure it is actually acyclic
        is_dag = nx.is_directed_acyclic_graph(graph)
        assert is_dag, f"Graph {name} is not a valid DAG!"
        
        print(f"{name:<25} | {graph.number_of_nodes():<5} | {graph.number_of_edges():<5}")