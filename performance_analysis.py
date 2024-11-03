import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from util.algo_dataset import setup_portfolio_paths

def calculate_nav_metrics(daily_nav_path):
    # [Fungsi calculate_nav_metrics seperti yang diberikan sebelumnya]
    """
    Menghitung NAV Return dan Max Drawdown dari data NAV harian
    
    Args:
        daily_nav_path: Path ke file CSV yang berisi data NAV harian
        
    Returns:
        dict: Dictionary berisi metrics NAV (return dan max drawdown dalam persen)
    """
    # Baca data NAV
    df_nav = pd.read_csv(daily_nav_path, parse_dates=['Date'])
    
    # Hitung NAV Return
    initial_nav = df_nav['Net'].iloc[0]
    final_nav = df_nav['Net'].iloc[-1]
    nav_return = ((final_nav - initial_nav) / initial_nav) * 100
    
    # Hitung Max Drawdown
    rolling_max = df_nav['Net'].expanding().max()
    drawdowns = (df_nav['Net'] - rolling_max) / rolling_max * 100
    max_drawdown = drawdowns.min()
    
    # Hitung statistik tambahan
    daily_returns = df_nav['Net'].pct_change()
    annualized_return = ((1 + nav_return/100) ** (252/len(df_nav)) - 1) * 100
    volatility = daily_returns.std() * np.sqrt(252) * 100
    sharpe_ratio = annualized_return / volatility if volatility != 0 else 0
    
    metrics = {
        'Total Return (%)': round(nav_return, 2),
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Annualized Return (%)': round(annualized_return, 2),
        'Annualized Volatility (%)': round(volatility, 2),
        'Sharpe Ratio': round(sharpe_ratio, 2)
    }
    
    return metrics

def plot_nav_performance(daily_nav_path, passive_nav_path):
    df_active = pd.read_csv(daily_nav_path, parse_dates=['Date'])
    df_passive = pd.read_csv(passive_nav_path, parse_dates=['Date'])
    
    # Normalisasi NAV (konversi ke basis 100)
    df_active['Nav_Normalized'] = df_active['Net'] * 100 / df_active['Net'].iloc[0]
    df_passive['Nav_Normalized'] = df_passive['Net'] * 100 / df_passive['Net'].iloc[0]
    
    # Hitung drawdown
    def calculate_drawdown(nav_series):
        rolling_max = nav_series.expanding().max()
        drawdown = (nav_series - rolling_max) / rolling_max * 100
        return drawdown
    
    df_active['Drawdown'] = calculate_drawdown(df_active['Nav_Normalized'])
    df_passive['Drawdown'] = calculate_drawdown(df_passive['Nav_Normalized'])
    
    # Buat plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
    
    # Plot NAV
    ax1.plot(df_active['Date'], df_active['Nav_Normalized'], 
             label='Active Strategy', color='blue')
    ax1.plot(df_passive['Date'], df_passive['Nav_Normalized'], 
             label='Passive Strategy', color='gray', linestyle='--')
    ax1.set_title('NAV Performance Comparison')
    ax1.set_ylabel('NAV (Base 100)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot Drawdown
    ax2.fill_between(df_active['Date'], df_active['Drawdown'], 0, 
                     color='red', alpha=0.3, label='Active Drawdown')
    ax2.fill_between(df_passive['Date'], df_passive['Drawdown'], 0, 
                     color='gray', alpha=0.3, label='Passive Drawdown')
    ax2.set_title('Drawdown Analysis')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig

def display_performance_analysis(portfolio, approach, predict):
    # Get paths
    """
    Menampilkan analisis performa lengkap untuk strategi tertentu
    """
    # Get paths
    paths = setup_portfolio_paths(portfolio, approach, predict)
    
    # Calculate metrics
    active_metrics = calculate_nav_metrics(paths['subfolder'] + '/daily_nav.csv')
    passive_metrics = calculate_nav_metrics(paths['subfolder'] + '/passive_daily_nav.csv')
    
    # Display results
    print("\nPerformance Metrics:")
    print("-" * 50)
    print(f"{'Metric':<25} {'Active':<15} {'Passive':<15}")
    print("-" * 50)
    
    for metric in active_metrics.keys():
        print(f"{metric:<25} {active_metrics[metric]:>14.2f}% {passive_metrics[metric]:>14.2f}%")
    
    # Create and show plots
    fig = plot_nav_performance(paths['subfolder'] + '/daily_nav.csv', 
                               paths['subfolder'] + '/passive_daily_nav.csv')
    plt.show()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze portfolio performance")
    parser.add_argument("--portfolio", required=True, help="Portfolio name")
    parser.add_argument("--approach", required=True, choices=['gradual', 'full_swing'], help="Approach used")
    parser.add_argument("--predict", action="store_true", help="Whether prediction was used")
    
    args = parser.parse_args()
    
    display_performance_analysis(args.portfolio, args.approach, args.predict)

