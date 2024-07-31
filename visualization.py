import io
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from benchmark import all_benchmarks
from ACOMultiAgentPathfinder import ACOMultiAgentPathfinder
from PIL import Image

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv('benchmark_results_detailed.csv')
    # Convert boolean success to integer (0 or 1)
    df['success_int'] = df['success'].astype(int)
    return df

df = load_data()

st.title('ACO Multi-Agent Pathfinder Benchmark Results')

# Overall statistics
st.header('Overall Statistics')
col1, col2, col3 = st.columns(3)
col1.metric('Overall Success Rate', f"{df['success_int'].mean():.2%}")
col2.metric('Avg Path Length (successful)', f"{df[df['success']]['mean_path_length'].mean():.2f}")
col3.metric('Avg Max Path Length (successful)', f"{df[df['success']]['longest_path'].mean():.2f}")

# Success rate by benchmark
st.header('Success Rate by Benchmark')
success_by_benchmark = df.groupby('benchmark')['success_int'].mean().sort_values(ascending=False)
fig = px.bar(success_by_benchmark, x=success_by_benchmark.index, y='success_int', 
             labels={'success_int': 'Success Rate', 'benchmark': 'Benchmark'},
             title='Success Rate by Benchmark')
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig)

# Path length distribution
st.header('Path Length Distribution (Successful Runs)')
fig = px.box(df[df['success']], x='benchmark', y='mean_path_length', 
             title='Distribution of Mean Path Lengths by Benchmark')
st.plotly_chart(fig)

# Benchmark-specific parameter analysis
st.header('Benchmark-Specific Parameter Analysis')

benchmark_type = st.selectbox('Select Benchmark Type', df['benchmark'].unique())

if benchmark_type == 'Linear':
    st.subheader('Linear Benchmark Analysis')
    
    linear_df = df[df['benchmark'] == 'Linear'].groupby(['length', 'n_agents'])['success_int'].mean().reset_index()
    fig = px.scatter(linear_df, x='length', y='success_int', 
                     color='n_agents', size='success_int',
                     labels={'length': 'Graph Length', 'success_int': 'Success Rate', 'n_agents': 'Number of Agents'},
                     title='Linear Benchmark: Graph Length vs Success Rate')
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig)

elif benchmark_type == 'Star':
    st.subheader('Star Benchmark Analysis')
    
    star_df = df[df['benchmark'] == 'Star'].groupby(['n_branches', 'branch_length', 'n_agents'])['success_int'].mean().reset_index()
    fig = px.scatter(star_df, x='n_branches', y='success_int', 
                     color='branch_length', size='n_agents',
                     labels={'n_branches': 'Number of Branches', 'success_int': 'Success Rate', 
                             'branch_length': 'Branch Length', 'n_agents': 'Number of Agents'},
                     title='Star Benchmark: Number of Branches vs Success Rate')
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig)
    
    # Additional plot for empty_branch parameter
    empty_branch_df = df[df['benchmark'] == 'Star'].groupby('empty_branch')['success_int'].mean().reset_index()
    fig = px.bar(empty_branch_df, x='empty_branch', y='success_int',
                 labels={'empty_branch': 'Empty Branch', 'success_int': 'Success Rate'},
                 title='Star Benchmark: Impact of Empty Branch on Success Rate')
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig)

elif benchmark_type == 'Grid':
    st.subheader('Grid Benchmark Analysis')
    
    grid_df = df[df['benchmark'] == 'Grid'].groupby(['width', 'height', 'n_agents'])['success_int'].mean().reset_index()
    fig = px.scatter_3d(grid_df, x='width', y='height', z='success_int', 
                        color='n_agents', size='success_int',
                        labels={'width': 'Grid Width', 'height': 'Grid Height', 'success_int': 'Success Rate', 
                                'n_agents': 'Number of Agents'},
                        title='Grid Benchmark: Grid Dimensions vs Success Rate')
    fig.update_scenes(zaxis_tickformat=".0%")
    st.plotly_chart(fig)
    
elif benchmark_type == 'Random Grid':
    st.subheader('Random Grid Benchmark Analysis')
    
    grid_df = df[df['benchmark'] == 'Random Grid'].groupby(['width', 'height', 'n_agents'])['success_int'].mean().reset_index()
    fig = px.scatter_3d(grid_df, x='width', y='height', z='success_int', 
                        color='n_agents', size='success_int',
                        labels={'width': 'Grid Width', 'height': 'Grid Height', 'success_int': 'Success Rate', 
                                'n_agents': 'Number of Agents'},
                        title='Random Grid Benchmark: Grid Dimensions vs Success Rate')
    fig.update_scenes(zaxis_tickformat=".0%")
    st.plotly_chart(fig)

elif benchmark_type == 'Passage':
    st.subheader('Passage Benchmark Analysis')
    
    passage_df = df[df['benchmark'] == 'Passage'].groupby(['width', 'height', 'passage_length'])['success_int'].mean().reset_index()
    fig = px.scatter_3d(passage_df, x='width', y='height', z='success_int', 
                        color='passage_length', size='success_int',
                        labels={'width': 'Width', 'height': 'Height', 'success_int': 'Success Rate', 
                                'passage_length': 'Passage Length'},
                        title='Passage Benchmark: Dimensions vs Success Rate')
    fig.update_scenes(zaxis_tickformat=".0%")
    st.plotly_chart(fig)

# Scatter plot: Number of agents vs Success Rate
st.header('Number of Agents vs Success Rate')
agent_success = df.groupby(['benchmark', 'n_agents'])['success_int'].mean().reset_index()
fig = px.scatter(agent_success, x='n_agents', y='success_int', color='benchmark', 
                 labels={'success_int': 'Success Rate', 'n_agents': 'Number of Agents'},
                 title='Number of Agents vs Success Rate')
fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig)

# Heatmap: Parameters vs Success Rate
st.header('Parameter Impact on Success Rate')
params = ['alpha', 'beta', 'gamma', 'evaporation_rate', 'initial_epsilon']
param_success = df.groupby(params)['success_int'].mean().reset_index()
fig = go.Figure(data=go.Heatmap(
    z=param_success['success_int'],
    x=param_success['beta'],
    y=param_success['alpha'],
    colorscale='Viridis'))
fig.update_layout(title='Success Rate Heatmap (Alpha vs Beta)', 
                  xaxis_title='Beta', yaxis_title='Alpha')
st.plotly_chart(fig)

# Allow user to select parameters for custom scatter plot
st.header('Custom Parameter Comparison')
x_param = st.selectbox('Select X-axis parameter', params)
y_param = st.selectbox('Select Y-axis parameter', [p for p in params if p != x_param])
custom_df = df.groupby([x_param, y_param])['success_int'].mean().reset_index()
fig = px.scatter(custom_df, x=x_param, y=y_param, color='success_int', size='success_int',
                 labels={x_param: x_param.capitalize(), y_param: y_param.capitalize(), 'success_int': 'Success Rate'},
                 title=f'{x_param.capitalize()} vs {y_param.capitalize()}')
fig.update_layout(coloraxis_colorbar=dict(tickformat=".0%"))
st.plotly_chart(fig)

def run_benchmark(benchmark, method):
    solution = benchmark.run(
        n_episodes=20, n_iterations=100,
        alpha=1, beta=2, gamma=1,
        evaporation_rate=0.1, dispersion_rate=0.1,
        communication_interval=5, initial_epsilon=0.8,
        collision_weight=0.3, method=method
    )
    
    
    return benchmark.G, benchmark.start_positions, benchmark.goal_positions, solution

def visualize_solution(G, solution, start_positions, goal_positions):
    pos = nx.spring_layout(G)
    
    frames = []
    max_path_length = max(len(path) for path in solution)
    
    for frame in range(max_path_length):
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(G, pos, node_color='lightgray', node_size=500, ax=ax)
        
        # Draw start and goal positions
        nx.draw_networkx_nodes(G, pos, nodelist=start_positions, node_color='green', node_size=300, ax=ax)
        nx.draw_networkx_nodes(G, pos, nodelist=goal_positions, node_color='red', node_size=200, ax=ax)
        
        # Draw agent positions at current frame
        for i, path in enumerate(solution):
            if frame < len(path):
                node = path[frame][0]
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_color=f'C{i}', node_size=600, ax=ax)
        
        ax.set_title(f'Time step: {frame}')
        plt.tight_layout()
        
        # Save the figure to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        frames.append(Image.open(buf))
        plt.close(fig)
    
    # Create a GIF from the frames
    gif_buf = io.BytesIO()
    frames[0].save(gif_buf, format='GIF', append_images=frames[1:], save_all=True, duration=1000, loop=0)
    gif_buf.seek(0)
    
    return gif_buf
# Add a new section for running benchmarks
st.header('Run Benchmark')

# Create a dropdown to select the benchmark
benchmark_names = [str(b) for b in all_benchmarks]
selected_benchmark_name = st.selectbox('Select Benchmark', benchmark_names)

# Find the selected benchmark and its parameters
selected_benchmark = next(b for b in all_benchmarks if str(b) == selected_benchmark_name)

# Display and allow editing of benchmark parameters
st.subheader('Benchmark Parameters')

def format_path(path):
    return ' -> '.join(str(node[0]) for node in path)

# Add a button to run the benchmark
if st.button('Run Benchmark'):
    # Run the benchmark with both methods
    aco_results = run_benchmark(selected_benchmark, 'aco')
    q_learning_results = run_benchmark(selected_benchmark, 'q-learning')

    # Display results
    st.subheader('Results')
    col1, col2 = st.columns(2)

    for column, method, results in [(col1, 'ACO', aco_results), (col2, 'Q-Learning', q_learning_results)]:
        with column:
            st.write(f'{method} Results')
            G, start_positions, goal_positions, solution = results
            if solution:
                st.write(f'Success: Yes')
                path_lengths = [len(path) - 1 for path in solution]
                st.write(f'Average Path Length: {sum(path_lengths) / len(path_lengths):.2f}')
                st.write(f'Max Path Length: {max(path_lengths)}')
                
                # Visualize solution
                gif_buf = visualize_solution(G, solution, start_positions, goal_positions)
                st.image(gif_buf.getvalue(), caption=f'{method} Solution Animation')
                st.write("Paths:")
                for i, (start, goal, path) in enumerate(zip(start_positions, goal_positions, solution)):
                    st.write(f"Agent {i}: {start} -> {goal}")
                    st.write(format_path(path))
                    st.write(f"Path length: {len(path) - 1}")
                    if path[-1][0] != goal:
                        st.write(f"Warning: Path does not end at the goal location!")
                    st.write("---")
            else:
                st.write('Success: No')