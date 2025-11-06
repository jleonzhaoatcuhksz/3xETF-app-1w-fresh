"""
Comprehensive ML Result Viewer - Fixed Version
Training, Validation, Testing Results with Trading Analysis
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sqlite3
import os
from pathlib import Path

class MLResultViewer:
    def __init__(self):
        """Initialize ML Result Viewer"""
        print("📊 ML RESULT VIEWER - FIXED VERSION")
        print("Comprehensive analysis of training, validation & testing")
        print("=" * 60)
        
        self.results = {}
        self.load_all_results()
        
    def load_all_results(self):
        """Load all available ML results"""
        print("📂 Loading ML results...")
        
        # Result files to check
        result_files = [
            'best_fast_ml_results.json',
            'random_forest_results.json', 
            'xgboost_results.json',
            'lightgbm_results.json',
            'positive_strategy_results.json',
            'improved_strategy_results.json',
            'etf_switching_results.json'
        ]
        
        for file in result_files:
            if os.path.exists(file):
                try:
                    with open(file, 'r') as f:
                        data = json.load(f)
                    
                    model_name = file.replace('_results.json', '').replace('.json', '')
                    self.results[model_name] = data
                    print(f"✅ Loaded: {file}")
                    
                except Exception as e:
                    print(f"❌ Error loading {file}: {e}")
            else:
                print(f"⚠️ Not found: {file}")
        
        print(f"📊 Total results loaded: {len(self.results)}")
        
    def display_training_summary(self):
        """Display comprehensive training summary"""
        print("\n" + "="*60)
        print("🤖 TRAINING PHASE SUMMARY")
        print("="*60)
        
        # Training performance comparison
        training_data = []
        
        for model_name, results in self.results.items():
            if 'model_name' in results:
                training_data.append({
                    'Model': results['model_name'].upper(),
                    'Strategy Return (%)': f"{results.get('strategy_return', 0):.2f}",
                    'SPY Return (%)': f"{results.get('spy_return', 0):.2f}",
                    'Outperformance (%)': f"{results.get('outperformance', 0):.2f}",
                    'Total Trades': results.get('total_trades', 0),
                    'Status': '🏆 SUCCESS' if results.get('outperformance', 0) > 0 else '⚠️ UNDERPERFORMED'
                })
        
        if training_data:
            df = pd.DataFrame(training_data)
            print(df.to_string(index=False))
        else:
            print("No detailed training results available")
            
        # Best performing model
        if training_data:
            best_model = max(training_data, key=lambda x: float(x['Outperformance (%)']))
            print(f"\n🥇 BEST PERFORMING MODEL: {best_model['Model']}")
            print(f"   Outperformance: {best_model['Outperformance (%)']}%")
            print(f"   Total Trades: {best_model['Total Trades']}")
    
    def display_validation_analysis(self):
        """Display validation phase analysis"""
        print("\n" + "="*60)
        print("✅ VALIDATION PHASE ANALYSIS")
        print("="*60)
        
        # Load database for validation analysis
        try:
            conn = sqlite3.connect('./etf_data.db')
            
            # Get data statistics
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prices")
            total_records = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(DISTINCT symbol) FROM prices")
            total_etfs = cursor.fetchone()[0]
            
            cursor.execute("SELECT MIN(date), MAX(date) FROM prices")
            date_range = cursor.fetchone()
            
            cursor.execute("SELECT COUNT(*) FROM prices WHERE monthly_trend IS NOT NULL")
            valid_records = cursor.fetchone()[0]
            
            print(f"📊 DATA VALIDATION:")
            print(f"   Total Records: {total_records:,}")
            print(f"   Valid Records: {valid_records:,} ({valid_records/total_records*100:.1f}%)")
            print(f"   ETFs Available: {total_etfs}")
            print(f"   Date Range: {date_range[0]} to {date_range[1]}")
            
            # Feature validation
            cursor.execute("""
                SELECT symbol, COUNT(*) as records, 
                       AVG(monthly_trend) as avg_trend,
                       MIN(monthly_trend) as min_trend,
                       MAX(monthly_trend) as max_trend
                FROM prices 
                WHERE monthly_trend IS NOT NULL 
                GROUP BY symbol 
                ORDER BY records DESC
            """)
            
            etf_stats = cursor.fetchall()
            
            print(f"\n📈 ETF FEATURE VALIDATION:")
            print(f"{'ETF':<6} {'Records':<8} {'Avg Trend':<10} {'Min Trend':<10} {'Max Trend'}")
            print("-" * 50)
            
            for etf, records, avg_trend, min_trend, max_trend in etf_stats[:10]:
                print(f"{etf:<6} {records:<8,} {avg_trend:<10.2f} {min_trend:<10.2f} {max_trend:.2f}")
            
            conn.close()
            
        except Exception as e:
            print(f"❌ Validation error: {e}")
    
    def display_detailed_testing_results(self):
        """Display comprehensive testing results with step-by-step analysis"""
        print("\n" + "="*60)
        print("🧪 DETAILED TESTING RESULTS")
        print("="*60)
        
        for model_name, results in self.results.items():
            if 'trades' in results and results['trades']:
                self.analyze_model_testing(model_name, results)
    
    def safe_float(self, value, default=0.0):
        """Safely convert value to float"""
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def safe_str(self, value, default='N/A'):
        """Safely convert value to string"""
        try:
            return str(value) if value is not None else default
        except:
            return default
    
    def analyze_model_testing(self, model_name, results):
        """Analyze individual model testing results"""
        print(f"\n🔍 {model_name.upper()} TESTING ANALYSIS")
        print("-" * 40)
        
        # Overall performance
        print(f"📊 OVERALL PERFORMANCE:")
        print(f"   Strategy Return: {self.safe_float(results.get('strategy_return', 0)):+.2f}%")
        print(f"   SPY Benchmark:   {self.safe_float(results.get('spy_return', 0)):+.2f}%")
        print(f"   Outperformance:  {self.safe_float(results.get('outperformance', 0)):+.2f}%")
        print(f"   Final Portfolio: ${self.safe_float(results.get('final_portfolio_value', 0)):,.2f}")
        print(f"   Final SPY Value: ${self.safe_float(results.get('final_spy_value', 0)):,.2f}")
        
        # Trading analysis
        trades = results.get('trades', [])
        if trades:
            print(f"\n💼 TRADING ANALYSIS:")
            print(f"   Total Trades: {len(trades)}")
            
            # Trade details
            print(f"\n📋 TRADE-BY-TRADE ANALYSIS:")
            print(f"{'Date':<12} {'From':<5} {'To':<5} {'Confidence':<10} {'Portfolio Value':<15} {'P&L'}")
            print("-" * 70)
            
            previous_value = 10000.0  # Starting value
            
            for i, trade in enumerate(trades):
                try:
                    current_value = self.safe_float(trade.get('portfolio_value', 0))
                    pnl = current_value - previous_value
                    pnl_pct = (pnl / previous_value) * 100 if previous_value > 0 else 0
                    
                    date_str = self.safe_str(trade.get('date', ''))[:10]
                    from_etf = self.safe_str(trade.get('from_etf', 'N/A'))[:5]
                    to_etf = self.safe_str(trade.get('to_etf', 'N/A'))[:5]
                    confidence = self.safe_float(trade.get('confidence', 0))
                    
                    pnl_symbol = "📈" if pnl >= 0 else "📉"
                    
                    print(f"{date_str:<12} {from_etf:<5} {to_etf:<5} {confidence:<10.3f} ${current_value:<13,.2f} {pnl_symbol} {pnl:+.2f} ({pnl_pct:+.1f}%)")
                    
                    previous_value = current_value
                    
                except Exception as e:
                    print(f"Trade {i+1}: Data formatting error - {e}")
                    continue
            
            # Trading statistics
            try:
                portfolio_values = []
                for trade in trades:
                    pv = self.safe_float(trade.get('portfolio_value'))
                    if pv > 0:
                        portfolio_values.append(pv)
                
                if len(portfolio_values) > 1:
                    trade_returns = []
                    for i in range(1, len(portfolio_values)):
                        if portfolio_values[i-1] > 0:
                            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] * 100
                            trade_returns.append(ret)
                    
                    if trade_returns:
                        winning_trades = sum(1 for ret in trade_returns if ret > 0)
                        losing_trades = sum(1 for ret in trade_returns if ret < 0)
                        
                        print(f"\n📊 TRADING STATISTICS:")
                        print(f"   Winning Trades: {winning_trades} ({winning_trades/len(trade_returns)*100:.1f}%)")
                        print(f"   Losing Trades:  {losing_trades} ({losing_trades/len(trade_returns)*100:.1f}%)")
                        print(f"   Average Trade:  {np.mean(trade_returns):+.2f}%")
                        print(f"   Best Trade:     {max(trade_returns):+.2f}%")
                        print(f"   Worst Trade:    {min(trade_returns):+.2f}%")
                    else:
                        print(f"\n📊 TRADING STATISTICS: No valid trade returns calculated")
                else:
                    print(f"\n📊 TRADING STATISTICS: Insufficient data for analysis")
                    
            except Exception as e:
                print(f"\n📊 TRADING STATISTICS: Error calculating - {e}")
        
        # ETF allocation analysis
        if trades:
            etf_trades = {}
            for trade in trades:
                to_etf = self.safe_str(trade.get('to_etf', 'Unknown'))
                etf_trades[to_etf] = etf_trades.get(to_etf, 0) + 1
            
            print(f"\n🎯 ETF ALLOCATION ANALYSIS:")
            for etf, count in sorted(etf_trades.items(), key=lambda x: x[1], reverse=True):
                print(f"   {etf}: {count} trades ({count/len(trades)*100:.1f}%)")
    
    def create_simple_summary_chart(self):
        """Create a simple summary chart without complex plotting"""
        print(f"\n📈 CREATING PERFORMANCE SUMMARY...")
        
        # Extract performance data
        models = []
        returns = []
        outperformance = []
        trades = []
        
        for model_name, results in self.results.items():
            if 'strategy_return' in results:
                models.append(model_name.upper())
                returns.append(self.safe_float(results.get('strategy_return', 0)))
                outperformance.append(self.safe_float(results.get('outperformance', 0)))
                trades.append(results.get('total_trades', 0))
        
        if models:
            print(f"\n📊 PERFORMANCE SUMMARY TABLE:")
            print(f"{'Model':<15} {'Return':<10} {'Outperf':<10} {'Trades':<8} {'Rating'}")
            print("-" * 55)
            
            for i, model in enumerate(models):
                rating = "🏆🏆🏆" if outperformance[i] > 50 else "🏆🏆" if outperformance[i] > 20 else "🏆" if outperformance[i] > 0 else "⚠️"
                print(f"{model:<15} {returns[i]:+.1f}%{'':<4} {outperformance[i]:+.1f}%{'':<4} {trades[i]:<8} {rating}")
            
            print(f"\n🥇 CHAMPION: {models[outperformance.index(max(outperformance))]}")
            print(f"   Best Outperformance: {max(outperformance):+.1f}%")
    
    def generate_text_report(self):
        """Generate comprehensive text report"""
        print(f"\n📝 GENERATING TEXT REPORT...")
        
        report_content = f"""
ML ETF STRATEGY ANALYSIS REPORT
===============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Models Analyzed: {len(self.results)}

EXECUTIVE SUMMARY
================
"""
        
        # Add model performance
        for model_name, results in self.results.items():
            if 'strategy_return' in results:
                report_content += f"""
{model_name.upper()} RESULTS:
- Strategy Return: {self.safe_float(results.get('strategy_return', 0)):+.2f}%
- SPY Benchmark: {self.safe_float(results.get('spy_return', 0)):+.2f}%
- Outperformance: {self.safe_float(results.get('outperformance', 0)):+.2f}%
- Total Trades: {results.get('total_trades', 0)}
- Status: {'SUCCESS' if results.get('outperformance', 0) > 0 else 'UNDERPERFORMED'}
"""
        
        report_content += f"""

ANALYSIS COMPLETE
================
All models have been analyzed successfully.
Check console output for detailed trade-by-trade analysis.

Report generated by ML Result Viewer | ETF-app-1w Project
"""
        
        with open('ml_analysis_report.txt', 'w') as f:
            f.write(report_content)
        
        print("✅ Text report saved to: ml_analysis_report.txt")
    
    def run_complete_analysis(self):
        """Run complete ML result analysis"""
        print("🚀 STARTING COMPREHENSIVE ML ANALYSIS")
        print("=" * 60)
        
        # Display all phases
        self.display_training_summary()
        self.display_validation_analysis()
        self.display_detailed_testing_results()
        
        # Create simple visualizations
        self.create_simple_summary_chart()
        
        # Generate report
        self.generate_text_report()
        
        print(f"\n🎯 ANALYSIS COMPLETE!")
        print("Files generated:")
        print("  📄 ml_analysis_report.txt - Comprehensive text report")
        print("  📋 Console output - Detailed analysis")

def main():
    """Main execution function"""
    viewer = MLResultViewer()
    viewer.run_complete_analysis()

if __name__ == "__main__":
    main()