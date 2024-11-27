
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import numpy as np

# Sample data creation (replace with actual data loading)
def load_data():
    np.random.seed(42)
    data = {
        'n_jets_sel': np.random.randint(0, 6, 1000),
        'max_b_tag_score_sel': np.random.uniform(0, 1, 1000),
        'second_max_b_tag_score_sel': np.random.uniform(0, 1, 1000),
        'HT_sel': np.random.uniform(100, 500, 1000),
        'observed': np.random.poisson(5, 1000),
        'expected': np.random.uniform(4, 6, 1000)
    }
    return pd.DataFrame(data)

df = load_data()

# Example chi-squared function (replace with the actual one from your file)
def chi_squared_func(filtered_df):
    observed = filtered_df['observed']
    expected = filtered_df['expected']
    chi_squared = np.sum(((observed - expected) ** 2) / expected)
    return chi_squared

# Dash app setup
app = Dash(__name__)

# Layout with sliders
app.layout = html.Div([
    html.H1("Interactive Chi-Squared Dashboard", style={'textAlign': 'center'}),
    
    html.Div([
        html.Label("Min Number of Jets (n_jets_sel):"),
        dcc.Slider(0, 5, 1, value=2, id='n_jets_slider'),
        
        html.Label("Min Max B-Tag Score (max_b_tag_score_sel):"),
        dcc.Slider(0.0, 1.0, 0.01, value=0.7, id='max_b_tag_slider'),
        
        html.Label("Min Second Max B-Tag Score (second_max_b_tag_score_sel):"),
        dcc.Slider(0.0, 1.0, 0.01, value=0.4, id='second_max_b_tag_slider'),
        
        html.Label("Min HT Selection (HT_sel):"),
        dcc.Slider(100, 500, 10, value=200, id='HT_slider'),
    ], style={'width': '50%', 'margin': 'auto'}),
    
    html.Div(id='chi2-output', style={'textAlign': 'center', 'marginTop': '20px', 'fontSize': '20px'})
])

# Callback to update chi-squared value based on mask parameters
@app.callback(
    Output('chi2-output', 'children'),
    [
        Input('n_jets_slider', 'value'),
        Input('max_b_tag_slider', 'value'),
        Input('second_max_b_tag_slider', 'value'),
        Input('HT_slider', 'value')
    ]
)
def update_chi2(n_jets, max_b_tag, second_max_b_tag, HT):
    # Apply masking
    mask = (df['n_jets_sel'] >= n_jets) &            (df['max_b_tag_score_sel'] > max_b_tag) &            (df['second_max_b_tag_score_sel'] > second_max_b_tag) &            (df['HT_sel'] > HT)
    
    filtered_df = df[mask]
    
    # Calculate chi-squared
    if not filtered_df.empty:
        chi2 = chi_squared_func(filtered_df)
        return f"Chi-Squared Value: {chi2:.2f}"
    else:
        return "No data matches the selected criteria."

if __name__ == '__main__':
    app.run_server(debug=True)
