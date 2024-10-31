import argparse
import os
import pandas as pd
from bokeh.io import curdoc, output_notebook, output_file, show, save
from bokeh.plotting import figure

def get_title(portfolio: str, approach: str, predict: bool) -> str:
    """Generate dynamic title based on portfolio and approach."""
    portfolio_num = portfolio.replace('portfolio', '')
    approach_desc = 'with Prediction Model' if predict else 'without Prediction Model'
    approach_name = 'Gradual' if approach == 'Gradual Rebalancing' else 'Full Rebalancing'
    
    return f"Portfolio {portfolio_num} Net Asset Comparison - {approach_name} {approach_desc}"

def plot_daily_nav(df_list: list, stocks: list, output_path: str, title: str, x_col='Date'):
    """Plot daily net asset value comparison for the given dataframes and stocks."""
    p = figure(title=title, x_axis_type='datetime',
               background_fill_color="#fafafa", width=1000, height=600)

    # Plotting the RL rebalanced portfolio
    p.line(df_list[0][x_col], df_list[0]['Net'].values.tolist(), legend_label="RL rebalanced",
           line_color="black", line_width=2)

    # Colors for different stocks
    colors = ["red", "orange", "olivedrab", "blue", "purple"]
    for i, stock in enumerate(stocks):
        p.line(df_list[1][x_col], df_list[1][stock].values.tolist(), legend_label=stock,
               line_color=colors[i % len(colors)], line_width=2)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.grid.grid_line_alpha = 0.3
    output_file(output_path)
    show(p)

def main():
    parser = argparse.ArgumentParser(description="Visualize portfolio performance")
    parser.add_argument("--portfolio", required=True, help="Portfolio number (e.g., portfolio1, portfolio2)")
    parser.add_argument("--stocks", required=True, help="Comma-separated list of stock tickers")
    parser.add_argument("--approach", choices=['gradual', 'full_swing'], required=True, help="Choose between gradual or full_swing approach")
    parser.add_argument("--predict", action="store_true", help="Use LSTM prediction data")
    args = parser.parse_args()

    portfolio = args.portfolio
    stocks = args.stocks.split(',')
    approach = args.approach
    predict = args.predict

    # Base path for data files
    base_path = f'data/rl/{portfolio}'
    
    # Determine the correct subfolder based on approach and predict arguments
    if approach == 'gradual':
        subfolder = 'non_lagged' if predict else 'lagged'
    else:  # full_swing
        subfolder = 'fs_non_lagged' if predict else 'fs_lagged'
    
    try:
        folder_path = f'{base_path}/{subfolder}'
        
        # Load both files from the same subfolder
        df = pd.read_csv(f'{folder_path}/daily_nav.csv', parse_dates=['Date'])
        passive_df = pd.read_csv(f'{folder_path}/passive_daily_nav.csv', parse_dates=['Date'])
        
        df_list = [df, passive_df]

        # Generate dynamic title
        title = get_title(portfolio, approach, predict)

        # Define output file path for the visualization
        output_path = f'{folder_path}/daily_nav_comp_{approach}_{"predicted" if predict else "non_predicted"}.html'
        
        # Generate the plot
        plot_daily_nav(df_list, stocks, output_path, title)
        
        print(f"Visualization has been saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find required data files in {folder_path}/")
        print("Please ensure both daily_nav.csv and passive_daily_nav.csv exist in the specified directory.")
        print(f"Detailed error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()