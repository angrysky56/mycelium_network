#!/usr/bin/env python3
"""
Network Visualization Utilities for Mycelium Network

This module provides functions for visualizing mycelium networks
through data export and HTML generation.
"""

import json
from typing import Dict, Optional, List


def save_network_visualization_data(
    network,
    filename: str,
    include_node_details: bool = True,
    include_connection_details: bool = True
) -> None:
    """
    Save network visualization data to a JSON file.
    
    Parameters:
    -----------
    network : AdvancedMyceliumNetwork
        The mycelium network to visualize
    filename : str
        Path to save the JSON file
    include_node_details : bool
        Whether to include detailed node attributes
    include_connection_details : bool
        Whether to include detailed connection attributes
    """
    # Generate visualization data directly
    vis_data = {
        'nodes': [],
        'connections': [],
        'metrics': {
            'total_nodes': len(network.nodes),
            'input_nodes': len(network.input_nodes),
            'output_nodes': len(network.output_nodes),
            'regular_nodes': len(network.regular_nodes),
            'total_connections': sum(len(node.connections) for node in network.nodes.values()),
            'iteration': network.iteration,
        }
    }
    
    # Add node data
    for node_id, node in network.nodes.items():
        node_type = 'input' if node_id in network.input_nodes else 'output' if node_id in network.output_nodes else 'regular'
        node_data = {
            'id': node_id,
            'position': node.position,
            'type': node_type,
            'activation': node.activation,
        }
        
        # Add detailed node attributes if requested
        if include_node_details:
            node_data.update({
                'resource_level': node.resource_level,
                'energy': node.energy,
                'age': node.age,
                'sensitivity': node.sensitivity,
                'adaptability': node.adaptability,
                'specializations': node.specializations,
            })
        
        vis_data['nodes'].append(node_data)
    
    # Add connection data
    for source_id, node in network.nodes.items():
        for target_id, strength in node.connections.items():
            if target_id in network.nodes:  # Ensure target exists
                conn_data = {
                    'source': source_id,
                    'target': target_id,
                    'strength': strength,
                }
                
                # Add detailed connection attributes if requested
                if include_connection_details:
                    # Calculate actual distance
                    source_node = network.nodes[source_id]
                    target_node = network.nodes[target_id]
                    distance = network.environment.calculate_distance(
                        source_node.position, target_node.position
                    )
                    
                    conn_data.update({
                        'distance': distance,
                        'is_anastomosis': hasattr(source_node, 'anastomosis_targets') and 
                                         target_id in getattr(source_node, 'anastomosis_targets', set()),
                    })
                
                vis_data['connections'].append(conn_data)
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(vis_data, f, indent=2)


def generate_html_visualization(
    network_json_path: str,
    output_html_path: str,
    title: str = "Mycelium Network Visualization"
) -> None:
    """
    Generate an HTML file for visualizing the network.
    
    Parameters:
    -----------
    network_json_path : str
        Path to the network JSON data file
    output_html_path : str
        Path to save the HTML visualization
    title : str
        Title for the visualization
    """
    # Read the network data
    with open(network_json_path, 'r') as f:
        network_data = json.load(f)
    
    # Create HTML template with D3.js visualization
    html_template = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }}
        #visualization {{
            border: 1px solid #ccc;
            border-radius: 5px;
        }}
        .node {{
            stroke: #fff;
            stroke-width: 1.5px;
        }}
        .node.input {{
            fill: #6baed6;
        }}
        .node.output {{
            fill: #fd8d3c;
        }}
        .node.regular {{
            fill: #74c476;
        }}
        .link {{
            stroke: #999;
            stroke-opacity: 0.6;
        }}
        .tooltip {{
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            padding: 8px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        .controls {{
            margin-bottom: 10px;
        }}
        .metrics {{
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-top: 10px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <div class="controls">
        <button id="btn-toggle-physics">Toggle Physics</button>
        <button id="btn-toggle-labels">Toggle Labels</button>
        <label for="link-strength">Connection Visibility:</label>
        <input type="range" id="link-strength" min="0" max="1" step="0.1" value="0.2">
    </div>
    
    <div id="visualization"></div>
    
    <div class="tooltip" id="tooltip"></div>
    
    <div class="metrics">
        <h2>Network Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Nodes</td>
                <td id="total-nodes"></td>
            </tr>
            <tr>
                <td>Input Nodes</td>
                <td id="input-nodes"></td>
            </tr>
            <tr>
                <td>Output Nodes</td>
                <td id="output-nodes"></td>
            </tr>
            <tr>
                <td>Regular Nodes</td>
                <td id="regular-nodes"></td>
            </tr>
            <tr>
                <td>Total Connections</td>
                <td id="total-connections"></td>
            </tr>
        </table>
    </div>
    
    <script>
        // Network data
        const networkData = {JSON_DATA};
        
        // Set up the visualization
        const width = 800;
        const height = 600;
        let showLabels = false;
        let physicsEnabled = true;
        
        // Set up SVG
        const svg = d3.select("#visualization")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
        
        // Create the simulation
        const simulation = d3.forceSimulation(networkData.nodes)
            .force("link", d3.forceLink(networkData.connections)
                .id(d => d.id)
                .distance(50)
                .strength(0.1))
            .force("charge", d3.forceManyBody().strength(-100))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .on("tick", ticked);
        
        // Create links
        const link = svg.append("g")
            .selectAll("line")
            .data(networkData.connections)
            .enter()
            .append("line")
            .attr("class", "link")
            .style("stroke-width", d => Math.max(1, d.strength * 5));
        
        // Create nodes
        const node = svg.append("g")
            .selectAll("circle")
            .data(networkData.nodes)
            .enter()
            .append("circle")
            .attr("class", d => "node " + d.type)
            .attr("r", d => {
                if (d.type === "input") return 8;
                if (d.type === "output") return 8;
                return 5 + (d.resource_level || 0.5) * 3;
            })
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
        
        // Node labels
        const nodeLabels = svg.append("g")
            .selectAll("text")
            .data(networkData.nodes)
            .enter()
            .append("text")
            .text(d => d.id)
            .attr("font-size", 10)
            .attr("dx", 12)
            .attr("dy", 4)
            .style("opacity", 0);
        
        // Tooltip functionality
        const tooltip = d3.select("#tooltip");
        
        node.on("mouseover", function(event, d) {
            tooltip.style("opacity", 1)
                .html(generateNodeTooltip(d))
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        })
        .on("mouseout", function() {
            tooltip.style("opacity", 0);
        });
        
        link.on("mouseover", function(event, d) {
            tooltip.style("opacity", 1)
                .html(`Source: ${d.source.id}<br>Target: ${d.target.id}<br>Strength: ${d.strength.toFixed(3)}`)
                .style("left", (event.pageX + 10) + "px")
                .style("top", (event.pageY - 10) + "px");
        })
        .on("mouseout", function() {
            tooltip.style("opacity", 0);
        });
        
        // Control handlers
        d3.select("#btn-toggle-physics").on("click", togglePhysics);
        d3.select("#btn-toggle-labels").on("click", toggleLabels);
        d3.select("#link-strength").on("input", updateLinkVisibility);
        
        // Update network metrics
        updateMetrics();
        
        // Functions
        function ticked() {
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            node
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
                
            nodeLabels
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        }
        
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            if (!physicsEnabled) {
                d.fx = event.x;
                d.fy = event.y;
            } else {
                d.fx = null;
                d.fy = null;
            }
        }
        
        function togglePhysics() {
            physicsEnabled = !physicsEnabled;
            
            if (physicsEnabled) {
                // Re-enable physics
                node.each(d => {
                    d.fx = null;
                    d.fy = null;
                });
                simulation.alphaTarget(0.3).restart();
            } else {
                // Fix nodes in place
                node.each(d => {
                    d.fx = d.x;
                    d.fy = d.y;
                });
            }
        }
        
        function toggleLabels() {
            showLabels = !showLabels;
            nodeLabels.style("opacity", showLabels ? 1 : 0);
        }
        
        function updateLinkVisibility() {
            const threshold = parseFloat(this.value);
            link.style("opacity", d => d.strength >= threshold ? 1 : 0.1);
        }
        
        function updateMetrics() {
            d3.select("#total-nodes").text(networkData.metrics.total_nodes);
            d3.select("#input-nodes").text(networkData.metrics.input_nodes);
            d3.select("#output-nodes").text(networkData.metrics.output_nodes);
            d3.select("#regular-nodes").text(networkData.metrics.regular_nodes);
            d3.select("#total-connections").text(networkData.metrics.total_connections);
        }
        
        function generateNodeTooltip(node) {
            let html = `<strong>Node ${node.id} (${node.type})</strong><br>`;
            html += `Activation: ${node.activation.toFixed(3)}<br>`;
            
            if (node.resource_level !== undefined) {
                html += `Resource Level: ${node.resource_level.toFixed(3)}<br>`;
            }
            if (node.energy !== undefined) {
                html += `Energy: ${node.energy.toFixed(3)}<br>`;
            }
            if (node.age !== undefined) {
                html += `Age: ${node.age}<br>`;
            }
            
            return html;
        }
    </script>
</body>
</html>
'''.replace('{JSON_DATA}', json.dumps(network_data))

    # Write the HTML file
    with open(output_html_path, 'w') as f:
        f.write(html_template)
