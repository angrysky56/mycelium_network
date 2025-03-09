#!/usr/bin/env python3
"""
Animation Utilities for Mycelium Network

This module provides functions for creating animated visualizations
of network growth and evolution over time.
"""

import json
from typing import List, Dict


def visualize_network_growth(
    network_states: List[Dict],
    output_path: str,
    interval_ms: int = 500
) -> None:
    """
    Create an animated visualization of network growth over time.
    
    Parameters:
    -----------
    network_states : List[Dict]
        List of network visualization data at different time points
    output_path : str
        Path to save the animation HTML file
    interval_ms : int
        Interval between frames in milliseconds
    """
    # HTML template with D3.js animation
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mycelium Network Growth Visualization</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
        }
        #visualization {
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        .node {
            stroke: #fff;
            stroke-width: 1.5px;
        }
        .node.input {
            fill: #6baed6;
        }
        .node.output {
            fill: #fd8d3c;
        }
        .node.regular {
            fill: #74c476;
        }
        .node.new {
            fill: #ff9896;
        }
        .link {
            stroke: #999;
            stroke-opacity: 0.6;
        }
        .link.new {
            stroke: #d62728;
            stroke-opacity: 0.8;
        }
        .controls {
            margin: 20px 0;
        }
        .tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            border-radius: 4px;
            padding: 8px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
        }
        .time-info {
            margin-top: 10px;
            font-size: 16px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <h1>Mycelium Network Growth Visualization</h1>
    
    <div class="controls">
        <button id="btn-play-pause">Play</button>
        <button id="btn-prev">Previous</button>
        <button id="btn-next">Next</button>
        <label for="animation-speed">Speed:</label>
        <input type="range" id="animation-speed" min="100" max="2000" step="100" value="INTERVAL_MS">
        <span id="speed-value">INTERVAL_MS ms</span>
    </div>
    
    <div id="visualization"></div>
    
    <div class="tooltip" id="tooltip"></div>
    
    <div class="time-info">
        Timestep: <span id="current-timestep">0</span> / <span id="total-timesteps">MAX_TIMESTEP</span>
    </div>
    
    <script>
        // Network states data
        const networkStates = NETWORK_STATES;
        
        // Visualization parameters
        const width = 800;
        const height = 600;
        let currentStep = 0;
        let animationInterval = INTERVAL_MS;
        let animationTimer = null;
        let isPlaying = false;
        
        // Set up SVG
        const svg = d3.select("#visualization")
            .append("svg")
            .attr("width", width)
            .attr("height", height);
            
        // Create simulation
        const simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(d => d.id).distance(50).strength(0.1))
            .force("charge", d3.forceManyBody().strength(-100))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .on("tick", ticked);
        
        // Create link and node groups
        const linkGroup = svg.append("g");
        const nodeGroup = svg.append("g");
        
        // Initialize with first state
        updateVisualization(0);
        
        // Button event handlers
        d3.select("#btn-play-pause").on("click", togglePlayPause);
        d3.select("#btn-prev").on("click", showPreviousStep);
        d3.select("#btn-next").on("click", showNextStep);
        d3.select("#animation-speed").on("input", updateAnimationSpeed);
        
        // Functions
        function updateVisualization(stepIndex) {
            const state = networkStates[stepIndex];
            
            // Reset classes (for highlighting new elements)
            svg.selectAll(".node").classed("new", false);
            svg.selectAll(".link").classed("new", false);
            
            // Update simulation nodes and links
            simulation.nodes(state.nodes);
            simulation.force("link").links(state.connections);
            
            // Update links
            const link = linkGroup.selectAll("line")
                .data(state.connections, d => `${d.source.id || d.source}-${d.target.id || d.target}`);
                
            link.exit().remove();
            
            const linkEnter = link.enter()
                .append("line")
                .attr("class", "link new")
                .style("stroke-width", d => Math.max(1, d.strength * 5));
                
            link.merge(linkEnter)
                .transition()
                .duration(300)
                .style("stroke-width", d => Math.max(1, d.strength * 5));
            
            // Update nodes
            const node = nodeGroup.selectAll("circle")
                .data(state.nodes, d => d.id);
                
            node.exit().remove();
            
            const nodeEnter = node.enter()
                .append("circle")
                .attr("class", d => `node ${d.type} new`)
                .attr("r", d => {
                    if (d.type === "input") return 8;
                    if (d.type === "output") return 8;
                    return 5 + (d.resource_level || 0.5) * 3;
                })
                .call(d3.drag()
                    .on("start", dragstarted)
                    .on("drag", dragged)
                    .on("end", dragended));
                    
            node.merge(nodeEnter)
                .transition()
                .duration(300)
                .attr("r", d => {
                    if (d.type === "input") return 8;
                    if (d.type === "output") return 8;
                    return 5 + (d.resource_level || 0.5) * 3;
                });
                
            // Add tooltip interaction
            const tooltip = d3.select("#tooltip");
            
            nodeEnter.on("mouseover", function(event, d) {
                tooltip.style("opacity", 1)
                    .html(`Node ${d.id} (${d.type})<br>Activation: ${d.activation.toFixed(3)}`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 10) + "px");
            })
            .on("mouseout", function() {
                tooltip.style("opacity", 0);
            });
            
            // Update current timestep display
            d3.select("#current-timestep").text(stepIndex);
            
            // Restart simulation
            simulation.alpha(0.3).restart();
        }
        
        function ticked() {
            linkGroup.selectAll("line")
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
                
            nodeGroup.selectAll("circle")
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
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
            d.fx = null;
            d.fy = null;
        }
        
        function togglePlayPause() {
            isPlaying = !isPlaying;
            
            if (isPlaying) {
                d3.select("#btn-play-pause").text("Pause");
                animationTimer = setInterval(() => {
                    currentStep = (currentStep + 1) % networkStates.length;
                    updateVisualization(currentStep);
                }, animationInterval);
            } else {
                d3.select("#btn-play-pause").text("Play");
                clearInterval(animationTimer);
            }
        }
        
        function showPreviousStep() {
            if (isPlaying) {
                togglePlayPause();
            }
            currentStep = (currentStep - 1 + networkStates.length) % networkStates.length;
            updateVisualization(currentStep);
        }
        
        function showNextStep() {
            if (isPlaying) {
                togglePlayPause();
            }
            currentStep = (currentStep + 1) % networkStates.length;
            updateVisualization(currentStep);
        }
        
        function updateAnimationSpeed() {
            animationInterval = parseInt(this.value);
            d3.select("#speed-value").text(`${animationInterval}ms`);
            
            if (isPlaying) {
                clearInterval(animationTimer);
                animationTimer = setInterval(() => {
                    currentStep = (currentStep + 1) % networkStates.length;
                    updateVisualization(currentStep);
                }, animationInterval);
            }
        }
    </script>
</body>
</html>
"""

    # Prepare the HTML with the appropriate values
    html_content = html_content.replace('INTERVAL_MS', str(interval_ms))
    html_content = html_content.replace('MAX_TIMESTEP', str(len(network_states) - 1))
    html_content = html_content.replace('NETWORK_STATES', json.dumps(network_states))
    
    # Write the HTML file
    with open(output_path, 'w') as f:
        f.write(html_content)
