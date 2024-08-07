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
    df['benchmark'] = df['benchmark'].astype('category')
    df['benchmark_type'] = df['benchmark_type'].astype('category')
    return df

df = load_data()

st.title('ACO Multi-Agent Pathfinder Benchmark Results')

# Overall statistics
method = st.radio('Select Method', ('all', 'aco', 'q-learning', 'simplified-q-learning'))
df = df if method == 'all' else df[df['method'] == method]
st.header('Overall Statistics')
col1, col2, col3 = st.columns(3)
col1.metric('Overall Success Rate', f"{df['success_int'].mean():.2%}")
col2.metric('Avg Path Length (successful)', f"{df[df['success']]['mean_path_length'].mean():.2f}")
col3.metric('Avg Max Path Length (successful)', f"{df[df['success']]['longest_path'].mean():.2f}")

# Success rate by benchmark
st.header('Success Rate by Benchmark')
fig = px.histogram(df, y="benchmark", x='success_int', color="benchmark_type",
             labels={'success_int': 'Success Rate', 'benchmark': 'Benchmark'},
             title='Success Rate by Benchmark')
#fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig)

# Path length distribution
st.header('Path Length Distribution (Successful Runs)')
fig = px.box(df[df['success']], x='benchmark', y='mean_path_length', color="benchmark_type", 
             title='Distribution of Mean Path Lengths by Benchmark')
st.plotly_chart(fig)

# Benchmark-specific parameter analysis

st.subheader(f'Prallel Coordinates')

filter_types = st.multiselect('Benchmark Type', list(df['benchmark_type'].unique()))
if not len(filter_types):
    selected_df = df
else:
    selected_df = df.loc[df['benchmark_type'].isin(filter_types)]
selectable_columns = selected_df.columns.tolist()
columns = st.multiselect('Select Parameters', selectable_columns)
    
grouped_df = selected_df.groupby(columns if "benchmark" in columns else ["benchmark"] + columns, dropna=True, observed=True)['success_int'].mean().reset_index().dropna(subset="success_int")
#st.write(grouped_df.describe())


#selected_df = df.loc[df['benchmark_type'].eq(benchmark_type)].groupby(['benchmark', 'n_agents', 'alpha', 'beta', 'gamma', 'evaporation_rate', 'initial_epsilon']).mean().reset_index()
#fig = px.parallel_categories(grouped_df.loc[:,columns+['success_int']], color='success_int', color_continuous_scale=px.colors.diverging.Tealrose, 
#                 title='Parallel Coordinates')

def mkdim(df, col):
    # make ticks for numerical columns:
    #if np.issubdtype(df[col].dtypes, np.number): 
    if df[col].dtypes == 'bool':
        print(f"{col}: {df[col].dtypes} (bool)")
        return dict(label=col, values=df[col].astype(int), tickvals=[0, 1], ticktext=['False', 'True'])
    if df[col].dtypes == 'category':
        print(f"{col}: {df[col].dtypes} (category)")
        print(f"{df[col].cat.categories}")
        print(f"{df[col].head().cat.codes}")
        return dict(label=col, values=df[col].cat.codes, tickvals=[i for i,_ in enumerate(df[col].cat.categories)], ticktext=df[col].cat.categories)
    if pd.api.types.is_numeric_dtype(df[col].dtypes):
        print(f"{col}: {df[col].dtypes} (numeric)")
        return dict(label=col, values=df[col], range=[df[col].min(), df[col].max()])
    return {}

dim_dict = [mkdim(grouped_df, col) for col in ['success_int'] + columns]
fig = go.Figure(
    data=go.Parcoords(
        line = dict(color = grouped_df['success_int'], colorscale = 'Portland_r', showscale=True),
        dimensions=dim_dict)
)
#fig.update_yaxes(tickformat=".0%")
st.plotly_chart(fig)

# Scatter plot: Number of agents vs Success Rate
st.header('Number of Agents vs Success Rate')
agent_success = df.groupby(['benchmark_type', 'n_agents'])['success_int'].mean().reset_index()
fig = px.scatter(agent_success, x='n_agents', y='success_int', color='benchmark_type', 
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

def run_benchmark(benchmark, method, **params):
    solution = benchmark.run(
            method=method, **params
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

params_description = dict(
        n_episodes=(1, 20, 20, 1),
        n_iterations=(10, 200, 100, 10),
        alpha=(0.0, 5.0, 1.0, 0.1),
        beta=(0.0, 5.0, 1.0, 0.1),
        gamma=(0.0, 5.0, 1.0, 0.1),
        evaporation_rate=(0.0, .5, 0.1, 0.01),
        dispersion_rate=(0.0, .1, 0.01, 0.01),
        communication_interval=(1, 20, 5, 1),
        initial_epsilon=(0.0, 1.0, 0.8, 0.1),
        collision_weight=(0.0, 1.0, 0.5, 0.1),
)
method = st.radio('Select Method', ('aco', 'q-learning', 'simplified-q-learning'))

params = {}
for param, description in params_description.items():
    params[param] = st.slider(param, *description)
params |= {'method': method}

# Add a button to run the benchmark
if st.button('Run Benchmark'):
    # Run the benchmark with both methods
    results = run_benchmark(selected_benchmark, **params)

    # Display results
    st.subheader(f'{method} results')
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