import plotly.express as px

def plot_transformed_coordinates(df: pd.DataFrame) -> None:
    """Plot transformed coordinates using Plotly."""
    
    fig = px.scatter(
        df, 
        x='x_real_m', 
        y='y_real_m', 
        title='Transformed Coordinates',
        labels={'x_real_m': 'X Real', 'y_real_m': 'Y Real'}
    )
    
    fig.update_layout(
        xaxis=dict(range=[-100, 100]),
        yaxis=dict(range=[-100, 100])
    )
    
    fig.show()