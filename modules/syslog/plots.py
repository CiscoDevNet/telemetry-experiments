"""
Copyright (c) 2024 Cisco and/or its affiliates.
This software is licensed to you under the terms of the Cisco Sample
Code License, Version 1.1 (the "License"). You may obtain a copy of the
License at
               https://developer.cisco.com/docs/licenses
All use of the material herein must be in accordance with the terms of
the License. All rights not expressly granted by the License are
reserved. Unless required by applicable law or agreed to separately in
writing, software distributed under the License is distributed on an "AS
IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
or implied.
"""

import plotly.graph_objs as go
import networkx as nx
import plotly.graph_objects as go
from networkx.algorithms.dag import topological_generations
import pandas as pd

def plot_file_type(dataset, type_column='type', path_column='path'):
    # Create a dictionary to count the occurrences of each file type
    file_type_count = {}
    file_type_paths = {}

    custom_colors = ['#00BCEB', '#FF6D2D', '#74BF4B', '#E2E2E2', '#0D274D']

    for data in dataset:
        file_type = data[type_column]
        file_path = data[path_column]
        if file_type in file_type_count:
            file_type_count[file_type] += 1
            file_type_paths[file_type].append(file_path)
        else:
            file_type_count[file_type] = 1
            file_type_paths[file_type] = [file_path]

    # Prepare data for the pie chart
    labels = list(file_type_count.keys())
    values = list(file_type_count.values())
    total_files = sum(values)

    # Generate hovertext with the list of paths for each file type
    hovertexts = []
    for ftype in labels:
        percentage = (file_type_count[ftype] / total_files) * 100
        file_list = '<br>'.join(file_type_paths[ftype])
        hovertexts.append(f"{ftype}: {file_type_count[ftype]} Files ({percentage:.2f}%)<br>{file_list}")

    fig = go.Figure(data=[go.Pie(labels=labels,
                                values=values,
                                textinfo='label+percent',
                                hovertext=hovertexts,
                                hoverinfo='text',
                                marker=dict(colors=custom_colors))])

    fig.update_layout(showlegend=True)
    fig.show()


def plot_tfidf_relevant_lines(dataset, type_column='file_type', path_column='file_path', content_column='content', sorted_content_column='snippet', syslog_only = False):
    file_indices = []  # To store the index of the file in the x-axis
    relative_positions = []  # To store the relative positions in the y-axis
    colors = []  # To store the color of each point
    hover_texts = []  # To store the hover text for each point

    # Create a mapping from file index to file path for the x-axis labels
    if syslog_only:
        file_index_to_path = {idx: data[path_column] for idx, data in enumerate(dataset) if data[type_column] == 'syslog'}
    else:
        file_index_to_path = {idx: data[path_column] for idx, data in enumerate(dataset)}

    for idx, data in enumerate(dataset):
        if syslog_only and data[type_column] != 'syslog':
            continue  # Skip files that are not of type 'syslog'
        
        file_path = data[path_column]
        relevant_lines = set(data[sorted_content_column].split('\n'))  # Extract the set of relevant lines
        
        # Get the total number of lines for this file
        total_lines = len(data[content_column].splitlines())
        
        # Iterate over all lines and mark relevant ones
        for line_number, line in enumerate(data[content_column].splitlines(), start=1):
            file_indices.append(idx)  # Use the file's index for the x-axis
            relative_position = (total_lines - line_number) / total_lines  # Calculate the relative position
            relative_positions.append(relative_position)
            
            if line in relevant_lines:
                colors.append('#00BCEB')
            else:
                colors.append('gray')
            
            # Create the hover text
            hover_text = f"{file_index_to_path[idx]}: Line {(1 - relative_position) * total_lines:.0f}<br>{line}"
            hover_texts.append(hover_text)

    # Create the scatter plot
    fig = go.Figure(data=go.Scatter(
        x=file_indices,
        y=relative_positions,
        mode='markers',
        marker=dict(color=colors),
        text=hover_texts  # Use the prepared hover text
    ))

    # Update layout to set the y-axis range from 0 to 1 and set custom tick labels
    fig.update_layout(
        yaxis=dict(autorange=False, range=[1, 0]),  # Set the y-axis range from 1 to 0 for inversion
        xaxis=dict(
            tickmode='array',
            tickvals=list(file_index_to_path.keys()),
            ticktext=list(file_index_to_path.values())
        ),
        title='Relevant Log Lines',
        xaxis_title='File',
        yaxis_title='Line Position (Normalized)',
        hovermode='closest',  # Update hover mode to show info for the closest point
        showlegend=False
    )

    # Show the figure
    fig.show()


def plot_DAG(dataset, dag, type_column='file_type', path_column='file_path', issue_id_column='_issue',issue_column='issue'):
    def wrap_description(description, n=100):
        words = description.split()
        lines = []
        current_line = ""

        for word in words:
            if len(current_line) + len(word) + 1 <= n:
                current_line += (word if not current_line else ' ' + word)
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)  # Append the last line

        return "<br>                                       ".join(lines)


    # Create a directed graph
    G = nx.DiGraph()

    hover_contents = {}
    dataset_issues = {row[issue_id_column]['id']: row for _, row in dataset.iterrows() if row[issue_column] == True }

    for node_id in dag['nodes']:
        if node_id in dataset_issues:
            row = dataset_issues[node_id]
            issue = row[issue_id_column]
            G.add_node(issue['id'], description=issue['description'])
            hover_contents[issue['id']] = f"""
            file: {row[path_column]}<br>
            type: {row[type_column]}<br>
            issue:<br>
                        id: {issue['id']}<br>
                        description: {wrap_description(issue['description'])}<br>
            """
        else:
            # Handle the case where the node_id is not found in the dataset_issues
            print(f"Warning: node_id {node_id} not found in the dataset")

    G.add_edges_from(dag['edges'])

    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError("The graph must be a DAG to use topological sort.")

    pos = {}
    y_pos = 0
    x_offset = 2  # Horizontal spacing between nodes
    for generation in topological_generations(G):
        x_pos = 0
        for node in generation:
            pos[node] = (x_pos, y_pos)
            x_pos += x_offset
        y_pos -= 1  # Move to the next level down

    # Create edge traces for the plotly plot
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color='black'),
        hoverinfo='none',
        mode='lines'
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='markers+text',
        hoverinfo='text',
        hovertext=[],
        marker=dict(size=50, color='#009edc', line=dict(width=2)),
        textfont=dict(size=16, color='white'),
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])
        node_trace['text'] += tuple([str(node)])
        node_trace['hovertext'] += tuple([hover_contents[node]])

    # Create text traces for the short descriptions
    text_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode='text',
        hoverinfo='none',
        textposition='bottom center'
    )

    label_len = 20
    for node in G.nodes():
        x, y = pos[node]
        # Shift the y position down to place text below the node
        text_trace['x'] += tuple([x])
        text_trace['y'] += tuple([y - 0.3])
        descr = G.nodes[node]['description']
        short_descr = descr if len(descr) < label_len else descr[:label_len] + ' ...'
        text_trace['text'] += tuple([short_descr])

    fig = go.Figure(data=[edge_trace, node_trace, text_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=5, l=5, r=5, t=5),
                        annotations=[],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    # Add arrow annotations for each edge
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        fig.add_annotation(
            x=x1, y=y1,
            ax=x0, ay=y0,
            xref='x', yref='y',
            axref='x', ayref='y',
            text='',
            showarrow=True,
            arrowhead=2,
            arrowsize=2,
            arrowwidth=2,
            arrowcolor='black'
        )

    # Display the figure
    fig.show()