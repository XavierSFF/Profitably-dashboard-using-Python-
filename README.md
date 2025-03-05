# Profitably-dashboard-using-Python-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc

# Set styles
plt.style.use('ggplot')
sns.set_theme(style="whitegrid")

# Generate sample data
np.random.seed(42)

# Date range for 2 years of monthly data
dates = pd.date_range(start='2019-01-01', end='2020-12-31', freq='M')

# Create regions and products
regions = ['North', 'South', 'East', 'West']
products = ['Product A', 'Product B', 'Product C', 'Product D']

# Generate sample data
data = []
for date in dates:
    for region in regions:
        for product in products:
            revenue = np.random.normal(100000, 20000)
            cogs = revenue * np.random.uniform(0.4, 0.7)
            marketing = revenue * np.random.uniform(0.05, 0.15)
            overhead = revenue * np.random.uniform(0.1, 0.2)
            
            # Add seasonality to revenue
            if date.month in [11, 12]:  # Holiday season boost
                revenue *= 1.2
            elif date.month in [1, 2]:  # Post-holiday slump
                revenue *= 0.8
            
            # Add quarterly patterns for certain products
            if product == 'Product A' and date.month in [3, 6, 9, 12]:
                revenue *= 1.15
            
            # Add regional variations
            if region == 'North':
                revenue *= 1.1
            elif region == 'South' and date.month in [6, 7, 8]:  # Summer boost in South
                revenue *= 1.2
            
            # Calculate profit
            profit = revenue - cogs - marketing - overhead
            margin = profit / revenue * 100
            
            # Create data record
            data.append({
                'Date': date,
                'Year': date.year,
                'Month': date.month,
                'Quarter': (date.month-1)//3 + 1,
                'Region': region,
                'Product': product,
                'Revenue': revenue,
                'COGS': cogs,
                'Marketing': marketing,
                'Overhead': overhead,
                'Profit': profit,
                'Margin': margin
            })

# Create DataFrame
df = pd.DataFrame(data)

# Create the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Profitability Dashboard", className="text-center my-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    html.P("Year:"),
                    dcc.Dropdown(
                        id='year-filter',
                        options=[{'label': year, 'value': year} for year in df['Year'].unique()],
                        value=df['Year'].max(),
                        clearable=False
                    ),
                    html.P("Region:", className="mt-3"),
                    dcc.Dropdown(
                        id='region-filter',
                        options=[{'label': region, 'value': region} for region in regions],
                        value=regions,
                        multi=True
                    ),
                    html.P("Product:", className="mt-3"),
                    dcc.Dropdown(
                        id='product-filter',
                        options=[{'label': product, 'value': product} for product in products],
                        value=products,
                        multi=True
                    )
                ])
            ])
        ], width=3),
        
        dbc.Col([
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Total Revenue"),
                        dbc.CardBody(id="revenue-card")
                    ])
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Total Profit"),
                        dbc.CardBody(id="profit-card")
                    ])
                ]),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Average Margin"),
                        dbc.CardBody(id="margin-card")
                    ])
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Monthly Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="monthly-performance")
                        ])
                    ])
                ], width=12, className="mt-4")
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Regional Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="regional-performance")
                        ])
                    ])
                ], width=6, className="mt-4"),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Product Performance"),
                        dbc.CardBody([
                            dcc.Graph(id="product-performance")
                        ])
                    ])
                ], width=6, className="mt-4")
            ])
        ], width=9)
    ])
], fluid=True)

# Define callbacks
@app.callback(
    [Output("revenue-card", "children"),
     Output("profit-card", "children"),
     Output("margin-card", "children"),
     Output("monthly-performance", "figure"),
     Output("regional-performance", "figure"),
     Output("product-performance", "figure")],
    [Input("year-filter", "value"),
     Input("region-filter", "value"),
     Input("product-filter", "value")]
)
def update_dashboard(year, regions, products):
    # Filter data based on selections
    filtered_df = df[(df['Year'] == year) & 
                     (df['Region'].isin(regions)) & 
                     (df['Product'].isin(products))]
    
    # Calculate KPIs
    total_revenue = filtered_df['Revenue'].sum()
    total_profit = filtered_df['Profit'].sum()
    avg_margin = filtered_df['Margin'].mean()
    
    # Format KPIs for display
    revenue_display = html.H3(f"${total_revenue/1000000:.2f}M")
    profit_display = html.H3(f"${total_profit/1000000:.2f}M")
    margin_display = html.H3(f"{avg_margin:.2f}%")
    
    # Monthly performance chart
    monthly_data = filtered_df.groupby('Month').agg({
        'Revenue': 'sum',
        'Profit': 'sum',
        'Margin': 'mean'
    }).reset_index()
    
    # Sort by month for proper timeline
    monthly_data = monthly_data.sort_values('Month')
    
    # Create monthly performance figure
    monthly_fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    monthly_fig.add_trace(
        go.Bar(x=monthly_data['Month'], y=monthly_data['Revenue'], name="Revenue"),
        secondary_y=False
    )
    
    monthly_fig.add_trace(
        go.Scatter(x=monthly_data['Month'], y=monthly_data['Profit'], name="Profit", line=dict(color='green')),
        secondary_y=False
    )
    
    monthly_fig.add_trace(
        go.Scatter(x=monthly_data['Month'], y=monthly_data['Margin'], name="Margin %", line=dict(color='red')),
        secondary_y=True
    )
    
    monthly_fig.update_layout(
        title_text="Monthly Revenue, Profit, and Margin",
        xaxis_title="Month",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    monthly_fig.update_yaxes(title_text="Amount ($)", secondary_y=False)
    monthly_fig.update_yaxes(title_text="Margin (%)", secondary_y=True)
    
    # Regional performance
    regional_data = filtered_df.groupby('Region').agg({
        'Revenue': 'sum',
        'Profit': 'sum',
        'Margin': 'mean'
    }).reset_index()
    
    regional_fig = px.bar(
        regional_data, 
        x='Region', 
        y=['Revenue', 'Profit'],
        barmode='group',
        title="Revenue and Profit by Region"
    )
    
    # Product performance
    product_data = filtered_df.groupby('Product').agg({
        'Revenue': 'sum',
        'Profit': 'sum',
        'Margin': 'mean'
    }).reset_index().sort_values('Profit', ascending=False)
    
    product_fig = px.scatter(
        product_data,
        x='Revenue',
        y='Profit',
        size='Margin',
        color='Product',
        hover_name='Product',
        title="Product Performance (Size of bubble represents margin %)"
    )
    
    return revenue_display, profit_display, margin_display, monthly_fig, regional_fig, product_fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
    
