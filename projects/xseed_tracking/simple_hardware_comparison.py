def calculate_hardware_comparison(data_processor, comparison_processor, time_tolerance=0):
    """
    Simple function to calculate overall metrics comparing old vs new hardware
    using already computed metrics for each player/protocol combination.
    
    Args:
        data_processor: DataProcessor instance
        comparison_processor: ComparisonProcessor instance
        time_tolerance: Time tolerance in seconds
        
    Returns:
        Dictionary with summary metrics for both hardware types
    """
    import numpy as np
    import pandas as pd
    import logging
    
    # Initialize aggregation containers
    hw_metrics = {
        'old': {
            'mean_errors': [],
            'rmse_values': [],
            'max_errors': [],
            'matching_rates': [],
            'player_protocols': set(),  # Keep track of player-protocol combinations
            'data_points': 0,  # Track total data points
        },
        'new': {
            'mean_errors': [],
            'rmse_values': [],
            'max_errors': [],
            'matching_rates': [],
            'player_protocols': set(),
            'data_points': 0,
        }
    }
    
    # Process each player
    for player in data_processor.get_players_list():
        # Check which hardware types are available for this player
        available_hw_types = data_processor.get_available_hardware_types(player)
        
        for hw_type in available_hw_types:
            # Get protocols for this player and hardware
            protocol_df = data_processor.get_protocol_info(player, hw_type)
            
            if protocol_df.empty:
                continue
                
            # Process each protocol
            for _, protocol_row in protocol_df.iterrows():
                protocol_id = protocol_row['Protocol ID']
                protocol_name = protocol_row.get('Protocol name', f"Protocol {protocol_id}")
                
                # Create a unique identifier for this player-protocol combination
                player_protocol_key = f"{player}_{protocol_id}"
                
                # Process data for this protocol
                gt_data, _, protocol_info = data_processor.process_player_data(
                    player, protocol_id, time_tolerance, hw_type
                )
                
                if gt_data.empty:
                    continue
                    
                # Get Xseed data
                xseed_data = data_processor.load_xseed_data(
                    player,
                    protocol_info['Start_Time'],
                    protocol_info['End_Time'],
                    time_tolerance,
                    hw_type
                )
                
                if xseed_data.empty:
                    continue
                
                # Calculate metrics for this player/protocol
                metrics = comparison_processor.calculate_tracking_accuracy(
                    gt_data, xseed_data, time_tolerance
                )
                
                # Extract key metrics if they're valid
                if metrics.get('mean_error') != float('inf'):
                    hw_metrics[hw_type]['mean_errors'].append(metrics.get('mean_error', 0))
                    hw_metrics[hw_type]['rmse_values'].append(metrics.get('rmse', 0))
                    hw_metrics[hw_type]['max_errors'].append(metrics.get('max_error', 0))
                    hw_metrics[hw_type]['matching_rates'].append(metrics.get('matching_rate', 0))
                    hw_metrics[hw_type]['player_protocols'].add(player_protocol_key)
                    hw_metrics[hw_type]['data_points'] += len(xseed_data)
                    
                    # Log success
                    logging.info(f"Added metrics for {player}, protocol {protocol_id}, hw={hw_type}: " 
                                f"mean_error={metrics.get('mean_error', 0):.2f}m")
                else:
                    logging.warning(f"Invalid metrics for {player}, protocol {protocol_id}, hw={hw_type}")
    
    # Create summary for each hardware type
    summary = {}
    
    for hw_type in ['old', 'new']:
        mean_errors = hw_metrics[hw_type]['mean_errors']
        rmse_values = hw_metrics[hw_type]['rmse_values']
        max_errors = hw_metrics[hw_type]['max_errors']
        matching_rates = hw_metrics[hw_type]['matching_rates']
        
        if not mean_errors:
            summary[hw_type] = {
                'status': 'No data available',
                'count': 0
            }
            continue
            
        # Calculate aggregated metrics
        summary[hw_type] = {
            'status': 'Data available',
            'count': len(mean_errors),
            'player_protocol_count': len(hw_metrics[hw_type]['player_protocols']),
            'data_points': hw_metrics[hw_type]['data_points'],
            'mean_error': np.mean(mean_errors),
            'rmse': np.mean(rmse_values),
            'max_error': np.mean(max_errors),
            'matching_rate': np.mean(matching_rates)
        }
    
    # Calculate comparison if we have data for both hardware types
    if (summary.get('old', {}).get('status') == 'Data available' and 
        summary.get('new', {}).get('status') == 'Data available'):
        
        # Calculate improvement percentages
        mean_error_improvement = summary['old']['mean_error'] - summary['new']['mean_error']
        mean_error_pct = (mean_error_improvement / summary['old']['mean_error']) * 100 if summary['old']['mean_error'] > 0 else 0
        
        rmse_improvement = summary['old']['rmse'] - summary['new']['rmse']
        rmse_pct = (rmse_improvement / summary['old']['rmse']) * 100 if summary['old']['rmse'] > 0 else 0
        
        max_error_improvement = summary['old']['max_error'] - summary['new']['max_error']
        max_error_pct = (max_error_improvement / summary['old']['max_error']) * 100 if summary['old']['max_error'] > 0 else 0
        
        summary['comparison'] = {
            'mean_error_diff': mean_error_improvement,
            'mean_error_pct': mean_error_pct,
            'rmse_diff': rmse_improvement,
            'rmse_pct': rmse_pct,
            'max_error_diff': max_error_improvement,
            'max_error_pct': max_error_pct
        }
    
    return summary

def display_hardware_comparison(summary):
    """
    Display a simple hardware comparison in Streamlit.
    
    Args:
        summary: Result from calculate_hardware_comparison function
    """
    import streamlit as st
    
    st.title("Hardware Comparison Summary")
    
    # Check if we have data for both hardware types
    if (summary.get('old', {}).get('status') != 'Data available' or 
        summary.get('new', {}).get('status') != 'Data available'):
        
        st.warning("Cannot compare hardware types - data missing for one or both types")
        
        # Show what we have
        if summary.get('old', {}).get('status') == 'Data available':
            st.info(f"Data available for XSEED v1 (old hardware): {summary['old']['player_protocol_count']} player-protocol combinations")
        else:
            st.info("No data available for XSEED v1 (old hardware)")
            
        if summary.get('new', {}).get('status') == 'Data available':
            st.info(f"Data available for XSEED v1.5 (new hardware): {summary['new']['player_protocol_count']} player-protocol combinations")
        else:
            st.info("No data available for XSEED v1.5 (new hardware)")
            
        return
    
    # Create columns for the two hardware types
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("XSEED v1 (Old Hardware)")
        st.metric("Mean Error", f"{summary['old']['mean_error']:.2f} m")
        st.metric("RMSE", f"{summary['old']['rmse']:.2f} m")
        st.metric("Max Error", f"{summary['old']['max_error']:.2f} m")
        st.metric("Matching Rate", f"{summary['old']['matching_rate']*100:.1f}%")
        st.metric("Data Points", f"{summary['old']['data_points']}")
        st.metric("Player-Protocol Combinations", f"{summary['old']['player_protocol_count']}")
        
    with col2:
        st.subheader("XSEED v1.5 (New Hardware)")
        st.metric("Mean Error", f"{summary['new']['mean_error']:.2f} m")
        st.metric("RMSE", f"{summary['new']['rmse']:.2f} m")
        st.metric("Max Error", f"{summary['new']['max_error']:.2f} m")
        st.metric("Matching Rate", f"{summary['new']['matching_rate']*100:.1f}%")
        st.metric("Data Points", f"{summary['new']['data_points']}")
        st.metric("Player-Protocol Combinations", f"{summary['new']['player_protocol_count']}")
    
    # Display comparison metrics 
    if 'comparison' in summary:
        st.subheader("Improvement Analysis")
        
        # Format improvement values
        mean_diff = summary['comparison']['mean_error_diff']
        mean_pct = summary['comparison']['mean_error_pct']
        
        rmse_diff = summary['comparison']['rmse_diff'] 
        rmse_pct = summary['comparison']['rmse_pct']
        
        max_diff = summary['comparison']['max_error_diff']
        max_pct = summary['comparison']['max_error_pct']
        
        # Create metrics with color-coded deltas
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if mean_diff > 0:
                st.metric("Mean Error Improvement", 
                         f"{abs(mean_diff):.2f} m", 
                         f"{mean_pct:.1f}% better",
                         delta_color="normal")
            else:
                st.metric("Mean Error Change", 
                         f"{abs(mean_diff):.2f} m", 
                         f"{abs(mean_pct):.1f}% worse",
                         delta_color="inverse")
                
        with col2:
            if rmse_diff > 0:
                st.metric("RMSE Improvement", 
                         f"{abs(rmse_diff):.2f} m", 
                         f"{rmse_pct:.1f}% better",
                         delta_color="normal")
            else:
                st.metric("RMSE Change", 
                         f"{abs(rmse_diff):.2f} m", 
                         f"{abs(rmse_pct):.1f}% worse",
                         delta_color="inverse")
                
        with col3:
            if max_diff > 0:
                st.metric("Max Error Improvement", 
                         f"{abs(max_diff):.2f} m", 
                         f"{max_pct:.1f}% better",
                         delta_color="normal")
            else:
                st.metric("Max Error Change", 
                         f"{abs(max_diff):.2f} m", 
                         f"{abs(max_pct):.1f}% worse",
                         delta_color="inverse")
        
        # Add a summary statement
        st.subheader("Summary")
        
        if mean_diff > 0 and rmse_diff > 0:
            conclusion = (f"XSEED v1.5 (new hardware) shows an overall improvement in tracking accuracy " 
                         f"with {mean_pct:.1f}% reduction in mean error and {rmse_pct:.1f}% reduction in RMSE.")
            st.success(conclusion)
        elif mean_diff < 0 and rmse_diff < 0:
            conclusion = (f"XSEED v1.5 (new hardware) shows decreased tracking accuracy " 
                         f"with {abs(mean_pct):.1f}% increase in mean error and {abs(rmse_pct):.1f}% increase in RMSE.")
            st.error(conclusion)
        else:
            conclusion = "XSEED v1.5 (new hardware) shows mixed results compared to v1 (old hardware)."
            st.warning(conclusion)

def run_simple_hardware_comparison(data_processor, comparison_processor):
    """
    
    Args:
        data_processor: DataProcessor instance
        comparison_processor: ComparisonProcessor instance
    """
    import streamlit as st
    
    st.write("## XSEED Hardware Comparison Analysis")
    st.write("This analysis compares XSEED v1 (old hardware) and XSEED v1.5 (new hardware) across all available data.")
    
    # Add time tolerance control
    time_tolerance = st.sidebar.slider(
        "Time Tolerance (seconds)",
        min_value=0,
        max_value=5,
        value=1,
        help="Time tolerance in seconds for matching points"
    )
    
    # Calculate comparison metrics
    with st.spinner("Calculating hardware comparison metrics..."):
        summary = calculate_hardware_comparison(
            data_processor,
            comparison_processor,
            time_tolerance
        )
    
    # Display the comparison
    display_hardware_comparison(summary)