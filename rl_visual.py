import argparse
import os
import pandas as pd
from bokeh.io import curdoc, output_notebook, output_file, show, save
from bokeh.plotting import figure

def get_title(portfolio: str, approach: str, predict: bool) -> str:
    portfolio_num = portfolio.replace('portfolio', '')
    approach_desc = 'with Prediction Model' if predict else 'without Prediction Model'
    approach_name = 'Gradual' if approach == 'gradual' else 'Full Rebalancing'
    return f"Portfolio {portfolio_num} Net Asset Value Comparison - {approach_name} {approach_desc}"


def plot_daily_nav(df_list: list, stocks: list, output_path: str, title: str, x_col='Date'):
    p = figure(title=title, x_axis_type='datetime',
               background_fill_color="#fafafa", width=1000, height=600)

    # Plot total portfolio values
    p.line(df_list[0][x_col], df_list[0]['Net'], 
           legend_label="RL Portfolio Total", 
           line_color="black", line_width=3)
    
    p.line(df_list[1][x_col], df_list[1]['Net'], 
           legend_label="Passive Portfolio Total", 
           line_color="gray", line_width=3, line_dash="dashed")

    # Colors for individual assets
    colors = ["red", "blue", "green", "orange", "purple", "brown", 
              "pink", "cyan", "magenta", "yellow", "lime", "teal", 
              "navy", "gold"]

    # Plot individual assets if they exist in the DataFrames
    for i, stock in enumerate(stocks):
        if stock in df_list[0].columns:
            p.line(df_list[0][x_col], df_list[0][stock], 
                   legend_label=f"{stock} (Active)",
                   line_color=colors[i % len(colors)], 
                   line_width=1.5)
            
        if stock in df_list[1].columns:
            p.line(df_list[1][x_col], df_list[1][stock], 
                   legend_label=f"{stock} (Passive)",
                   line_color=colors[i % len(colors)], 
                   line_width=1.5, 
                   line_dash="dotted")

    # Customize plot
    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.grid.grid_line_alpha = 0.3
    p.xaxis.axis_label = 'Date'
    p.yaxis.axis_label = 'Value'
    
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