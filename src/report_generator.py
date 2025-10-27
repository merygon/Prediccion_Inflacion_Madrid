"""
Report generation system for inflation prediction analysis.
Creates visualizations, economic analysis, and technical reports.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
from pathlib import Path
import yaml
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Visualization imports
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    import seaborn as sns
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False

# PDF generation imports
try:
    from reportlab.lib.pagesizes import A4, letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False


class ReportGenerator:
    """
    Handles report generation for inflation prediction system.
    
    This class creates visualizations, economic analysis, and comprehensive
    technical reports with model results and interpretations.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the ReportGenerator with configuration.
        
        Args:
            config_path (str): Path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Report parameters from config
        self.report_params = self.config.get('reports', {})
        self.chart_dpi = self.report_params.get('chart_dpi', 300)
        self.chart_style = self.report_params.get('chart_style', 'seaborn-v0_8')
        self.pdf_format = self.report_params.get('pdf_format', 'A4')
        
        # Output paths
        self.paths = self.config.get('paths', {})
        self.reports_dir = Path(self.paths.get('reports', 'reports/'))
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Set matplotlib style
        if _MATPLOTLIB_AVAILABLE:
            try:
                plt.style.use(self.chart_style)
            except:
                plt.style.use('default')
                self.logger.warning(f"Chart style '{self.chart_style}' not available, using default")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                self.logger.info(f"Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}. Using default parameters.")
            return {}
        except Exception as e:
            self.logger.error(f"Error loading config: {e}. Using default parameters.")
            return {}
    
    def create_visualizations(self, data: pd.DataFrame, predictions: pd.DataFrame,
                            model_results: Optional[Dict[str, Any]] = None,
                            output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Create visualizations using matplotlib/seaborn.
        
        Implements time series plots for historical data and predictions,
        and model performance comparison charts.
        
        Args:
            data (pd.DataFrame): Historical inflation data
            predictions (pd.DataFrame): Model predictions
            model_results (Dict[str, Any], optional): Model evaluation results
            output_dir (str, optional): Output directory for plots
            
        Returns:
            Dict[str, str]: Dictionary mapping plot types to file paths
        """
        if not _MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib and seaborn are required for visualizations. "
                            "Install with: pip install matplotlib seaborn")
        
        if output_dir is None:
            output_dir = self.reports_dir
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Creating visualizations for inflation analysis")
        
        created_plots = {}
        
        try:
            # Plot 1: Historical data and predictions time series
            plot_path = self._create_time_series_plot(data, predictions, output_dir)
            created_plots['time_series'] = plot_path
            
            # Plot 2: Model performance comparison
            if model_results:
                plot_path = self._create_model_comparison_plot(model_results, output_dir)
                created_plots['model_comparison'] = plot_path
            
            # Plot 3: Prediction distribution and statistics
            plot_path = self._create_prediction_distribution_plot(predictions, output_dir)
            created_plots['prediction_distribution'] = plot_path
            
            # Plot 4: Confidence intervals analysis
            if 'confidence_lower' in predictions.columns and 'confidence_upper' in predictions.columns:
                plot_path = self._create_confidence_intervals_plot(predictions, output_dir)
                created_plots['confidence_intervals'] = plot_path
            
            # Plot 5: Historical inflation analysis
            plot_path = self._create_historical_analysis_plot(data, output_dir)
            created_plots['historical_analysis'] = plot_path
            
            # Plot 6: Seasonal decomposition if enough data
            if len(data) >= 24:  # At least 2 years of data
                plot_path = self._create_seasonal_decomposition_plot(data, output_dir)
                created_plots['seasonal_decomposition'] = plot_path
            
            self.logger.info(f"Created {len(created_plots)} visualization plots")
            return created_plots
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations: {e}")
            raise
    
    def _create_time_series_plot(self, data: pd.DataFrame, predictions: pd.DataFrame,
                                output_dir: Path) -> str:
        """Create time series plot for historical data and predictions."""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot historical data
        if isinstance(data.index, pd.DatetimeIndex):
            hist_dates = data.index
        elif 'fecha' in data.columns:
            hist_dates = pd.to_datetime(data['fecha'])
        else:
            hist_dates = pd.date_range(start='2010-01-01', periods=len(data), freq='M')
        
        # Find inflation rate column
        inflation_col = self._find_inflation_column(data)
        if inflation_col:
            ax.plot(hist_dates, data[inflation_col], 
                   label='Inflación Histórica', linewidth=2, color='#2E86AB', alpha=0.8)
        
        # Plot predictions
        if 'fecha' in predictions.columns:
            pred_dates = pd.to_datetime(predictions['fecha'])
        else:
            last_date = hist_dates[-1] if len(hist_dates) > 0 else pd.Timestamp('2024-01-01')
            pred_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                     periods=len(predictions), freq='M')
        
        ax.plot(pred_dates, predictions['predicted_inflation'], 
               label='Predicciones', linewidth=2.5, color='#A23B72', marker='o', markersize=4)
        
        # Add confidence intervals
        if 'confidence_lower' in predictions.columns and 'confidence_upper' in predictions.columns:
            ax.fill_between(pred_dates, 
                           predictions['confidence_lower'], 
                           predictions['confidence_upper'],
                           alpha=0.3, color='#A23B72', 
                           label=f'Intervalo de Confianza ({int(predictions.get("confidence_level", [0.95])[0]*100)}%)')
        
        # Formatting
        ax.set_xlabel('Fecha', fontsize=12)
        ax.set_ylabel('Tasa de Inflación (%)', fontsize=12)
        ax.set_title('Evolución de la Inflación en España: Histórico y Predicciones', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.xticks(rotation=45)
        
        # Add horizontal line at 0%
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=0.8)
        
        plt.tight_layout()
        plot_path = output_dir / "inflacion_historica_predicciones.png"
        plt.savefig(plot_path, dpi=self.chart_dpi, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_model_comparison_plot(self, model_results: Dict[str, Any], 
                                    output_dir: Path) -> str:
        """Create model performance comparison chart."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Extract model performance data
        models = []
        mae_scores = []
        rmse_scores = []
        mape_scores = []
        
        for model_name, results in model_results.items():
            if results.get('status') == 'success':
                metrics = results.get('metrics', {})
                models.append(model_name.replace('_', ' ').title())
                mae_scores.append(metrics.get('MAE', 0))
                rmse_scores.append(metrics.get('RMSE', 0))
                mape_scores.append(metrics.get('MAPE', 0))
        
        if not models:
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No hay resultados de modelos disponibles', 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
        else:
            # Colors for different models
            colors = ['#2E86AB', '#A23B72', '#F18F01'][:len(models)]
            
            # MAE comparison
            bars1 = axes[0].bar(models, mae_scores, color=colors, alpha=0.7)
            axes[0].set_title('Error Absoluto Medio (MAE)', fontweight='bold')
            axes[0].set_ylabel('MAE')
            axes[0].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars1, mae_scores):
                axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # RMSE comparison
            bars2 = axes[1].bar(models, rmse_scores, color=colors, alpha=0.7)
            axes[1].set_title('Raíz del Error Cuadrático Medio (RMSE)', fontweight='bold')
            axes[1].set_ylabel('RMSE')
            axes[1].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars2, rmse_scores):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            # MAPE comparison
            bars3 = axes[2].bar(models, mape_scores, color=colors, alpha=0.7)
            axes[2].set_title('Error Porcentual Absoluto Medio (MAPE)', fontweight='bold')
            axes[2].set_ylabel('MAPE (%)')
            axes[2].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars3, mape_scores):
                axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                           f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('Comparación de Rendimiento de Modelos', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = output_dir / "comparacion_modelos.png"
        plt.savefig(plot_path, dpi=self.chart_dpi, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_prediction_distribution_plot(self, predictions: pd.DataFrame, 
                                           output_dir: Path) -> str:
        """Create prediction distribution and statistics plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        pred_values = predictions['predicted_inflation'].values
        
        # Histogram
        ax1.hist(pred_values, bins=15, alpha=0.7, color='#A23B72', edgecolor='black')
        ax1.set_xlabel('Tasa de Inflación Predicha (%)')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Predicciones de Inflación')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        mean_pred = np.mean(pred_values)
        std_pred = np.std(pred_values)
        ax1.axvline(mean_pred, color='red', linestyle='--', linewidth=2, 
                   label=f'Media: {mean_pred:.2f}%')
        ax1.axvline(mean_pred + std_pred, color='orange', linestyle=':', alpha=0.7, 
                   label=f'+1σ: {mean_pred + std_pred:.2f}%')
        ax1.axvline(mean_pred - std_pred, color='orange', linestyle=':', alpha=0.7, 
                   label=f'-1σ: {mean_pred - std_pred:.2f}%')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(pred_values, vert=True, patch_artist=True,
                   boxprops=dict(facecolor='#A23B72', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax2.set_ylabel('Tasa de Inflación Predicha (%)')
        ax2.set_title('Estadísticas de Predicciones')
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Estadísticas:
Media: {mean_pred:.2f}%
Mediana: {np.median(pred_values):.2f}%
Desv. Estándar: {std_pred:.2f}%
Mín: {np.min(pred_values):.2f}%
Máx: {np.max(pred_values):.2f}%"""
        
        ax2.text(1.1, 0.5, stats_text, transform=ax2.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
                verticalalignment='center')
        
        plt.tight_layout()
        
        plot_path = output_dir / "distribucion_predicciones.png"
        plt.savefig(plot_path, dpi=self.chart_dpi, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)    

    def _create_confidence_intervals_plot(self, predictions: pd.DataFrame, 
                                        output_dir: Path) -> str:
        """Create confidence intervals analysis plot."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Confidence interval width over time
        ci_width = predictions['confidence_upper'] - predictions['confidence_lower']
        
        if 'fecha' in predictions.columns:
            dates = pd.to_datetime(predictions['fecha'])
            ax1.plot(dates, ci_width, linewidth=2, color='#F18F01', marker='o', markersize=4)
            ax1.set_xlabel('Fecha')
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax1.plot(ci_width, linewidth=2, color='#F18F01', marker='o', markersize=4)
            ax1.set_xlabel('Período de Predicción')
        
        ax1.set_ylabel('Ancho del Intervalo de Confianza (%)')
        ax1.set_title('Evolución de la Incertidumbre en las Predicciones')
        ax1.grid(True, alpha=0.3)
        
        # Add mean line
        mean_width = np.mean(ci_width)
        ax1.axhline(y=mean_width, color='red', linestyle='--', alpha=0.7,
                   label=f'Ancho Promedio: {mean_width:.2f}%')
        ax1.legend()
        
        # Plot 2: Predictions with confidence bands
        if 'fecha' in predictions.columns:
            dates = pd.to_datetime(predictions['fecha'])
            ax2.plot(dates, predictions['predicted_inflation'], 
                    label='Predicción', linewidth=2, color='#2E86AB')
            ax2.fill_between(dates, 
                           predictions['confidence_lower'], 
                           predictions['confidence_upper'],
                           alpha=0.3, color='#2E86AB', 
                           label='Intervalo de Confianza')
            ax2.set_xlabel('Fecha')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        else:
            ax2.plot(predictions['predicted_inflation'], 
                    label='Predicción', linewidth=2, color='#2E86AB')
            ax2.fill_between(range(len(predictions)), 
                           predictions['confidence_lower'], 
                           predictions['confidence_upper'],
                           alpha=0.3, color='#2E86AB', 
                           label='Intervalo de Confianza')
            ax2.set_xlabel('Período de Predicción')
        
        ax2.set_ylabel('Tasa de Inflación (%)')
        ax2.set_title('Predicciones con Intervalos de Confianza')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = output_dir / "intervalos_confianza.png"
        plt.savefig(plot_path, dpi=self.chart_dpi, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_historical_analysis_plot(self, data: pd.DataFrame, output_dir: Path) -> str:
        """Create historical inflation analysis plot."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Find inflation column
        inflation_col = self._find_inflation_column(data)
        if not inflation_col:
            # Create empty plot with message
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No se encontró columna de inflación en los datos históricos', 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            plot_path = output_dir / "analisis_historico.png"
            plt.savefig(plot_path, dpi=self.chart_dpi, bbox_inches='tight')
            plt.close()
            return str(plot_path)
        
        inflation_data = data[inflation_col].dropna()
        
        # Get dates
        if isinstance(data.index, pd.DatetimeIndex):
            dates = data.index
        elif 'fecha' in data.columns:
            dates = pd.to_datetime(data['fecha'])
        else:
            dates = pd.date_range(start='2010-01-01', periods=len(data), freq='M')
        
        # Plot 1: Time series
        axes[0, 0].plot(dates, inflation_data, linewidth=1.5, color='#2E86AB')
        axes[0, 0].set_title('Serie Temporal de Inflación')
        axes[0, 0].set_ylabel('Tasa de Inflación (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Distribution
        axes[0, 1].hist(inflation_data, bins=20, alpha=0.7, color='#A23B72', edgecolor='black')
        axes[0, 1].set_title('Distribución Histórica de Inflación')
        axes[0, 1].set_xlabel('Tasa de Inflación (%)')
        axes[0, 1].set_ylabel('Frecuencia')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Add statistics
        mean_inf = np.mean(inflation_data)
        axes[0, 1].axvline(mean_inf, color='red', linestyle='--', 
                          label=f'Media: {mean_inf:.2f}%')
        axes[0, 1].legend()
        
        # Plot 3: Monthly seasonality (if enough data)
        if len(inflation_data) >= 12:
            monthly_data = pd.DataFrame({'inflation': inflation_data, 'month': dates.month})
            monthly_avg = monthly_data.groupby('month')['inflation'].mean()
            
            axes[1, 0].bar(monthly_avg.index, monthly_avg.values, 
                          color='#F18F01', alpha=0.7)
            axes[1, 0].set_title('Patrón Estacional Mensual')
            axes[1, 0].set_xlabel('Mes')
            axes[1, 0].set_ylabel('Inflación Promedio (%)')
            axes[1, 0].set_xticks(range(1, 13))
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Rolling statistics
        if len(inflation_data) >= 12:
            rolling_mean = inflation_data.rolling(window=12).mean()
            rolling_std = inflation_data.rolling(window=12).std()
            
            axes[1, 1].plot(dates, inflation_data, alpha=0.3, color='gray', label='Datos')
            axes[1, 1].plot(dates, rolling_mean, linewidth=2, color='#2E86AB', 
                           label='Media Móvil (12m)')
            axes[1, 1].fill_between(dates, 
                                   rolling_mean - rolling_std, 
                                   rolling_mean + rolling_std,
                                   alpha=0.2, color='#2E86AB', 
                                   label='±1 Desv. Estándar')
            axes[1, 1].set_title('Tendencia y Volatilidad')
            axes[1, 1].set_ylabel('Tasa de Inflación (%)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        plot_path = output_dir / "analisis_historico.png"
        plt.savefig(plot_path, dpi=self.chart_dpi, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def _create_seasonal_decomposition_plot(self, data: pd.DataFrame, output_dir: Path) -> str:
        """Create seasonal decomposition plot."""
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            self.logger.warning("statsmodels not available for seasonal decomposition")
            return ""
        
        # Find inflation column
        inflation_col = self._find_inflation_column(data)
        if not inflation_col:
            return ""
        
        inflation_data = data[inflation_col].dropna()
        
        if len(inflation_data) < 24:  # Need at least 2 years
            return ""
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(inflation_data, model='additive', period=12)
            
            fig, axes = plt.subplots(4, 1, figsize=(14, 12))
            
            # Original series
            axes[0].plot(decomposition.observed, linewidth=1.5, color='#2E86AB')
            axes[0].set_title('Serie Original')
            axes[0].set_ylabel('Inflación (%)')
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            axes[1].plot(decomposition.trend, linewidth=2, color='#A23B72')
            axes[1].set_title('Tendencia')
            axes[1].set_ylabel('Inflación (%)')
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal
            axes[2].plot(decomposition.seasonal, linewidth=1.5, color='#F18F01')
            axes[2].set_title('Componente Estacional')
            axes[2].set_ylabel('Inflación (%)')
            axes[2].grid(True, alpha=0.3)
            
            # Residual
            axes[3].plot(decomposition.resid, linewidth=1, color='gray')
            axes[3].set_title('Residuos')
            axes[3].set_ylabel('Inflación (%)')
            axes[3].set_xlabel('Fecha')
            axes[3].grid(True, alpha=0.3)
            
            plt.suptitle('Descomposición Estacional de la Inflación', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            plot_path = output_dir / "descomposicion_estacional.png"
            plt.savefig(plot_path, dpi=self.chart_dpi, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
            
        except Exception as e:
            self.logger.warning(f"Error in seasonal decomposition: {e}")
            return ""
    
    def _find_inflation_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find the inflation rate column in the data."""
        # Priority order for inflation column detection
        priority_keywords = [
            'inflation_rate_annual', 'inflacion_anual', 'ipc_annual_rate', 
            'annual_rate', 'inflation_rate', 'inflacion', 'rate'
        ]
        
        for keyword in priority_keywords:
            matching_cols = [col for col in data.columns if keyword in col.lower()]
            if matching_cols:
                return matching_cols[0]
        
        # Fallback to numeric columns that might contain inflation data
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if any(word in col.lower() for word in ['ipc', 'inflacion', 'inflation']):
                return col
        
        return None
    
    def generate_economic_analysis(self, data: pd.DataFrame, predictions: pd.DataFrame,
                                 model_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate economic analysis with interpretation.
        
        Implements interpretation of results and economic conclusions
        about inflation trends identified in the data and predictions.
        
        Args:
            data (pd.DataFrame): Historical inflation data
            predictions (pd.DataFrame): Model predictions
            model_results (Dict[str, Any], optional): Model evaluation results
            
        Returns:
            Dict[str, Any]: Economic analysis with interpretations and conclusions
        """
        self.logger.info("Generating economic analysis and interpretation")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'historical_analysis': {},
            'prediction_analysis': {},
            'economic_interpretation': {},
            'conclusions': [],
            'recommendations': []
        }
        
        try:
            # Historical data analysis
            analysis['historical_analysis'] = self._analyze_historical_data(data)
            
            # Prediction analysis
            analysis['prediction_analysis'] = self._analyze_predictions(predictions)
            
            # Economic interpretation
            analysis['economic_interpretation'] = self._generate_economic_interpretation(
                analysis['historical_analysis'], 
                analysis['prediction_analysis'],
                model_results
            )
            
            # Generate conclusions
            analysis['conclusions'] = self._generate_conclusions(
                analysis['historical_analysis'],
                analysis['prediction_analysis'],
                analysis['economic_interpretation']
            )
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis)
            
            # Save analysis to JSON
            analysis_path = self.reports_dir / "economic_analysis.json"
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Economic analysis saved to {analysis_path}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating economic analysis: {e}")
            raise
    
    def _analyze_historical_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze historical inflation data."""
        inflation_col = self._find_inflation_column(data)
        if not inflation_col:
            return {'error': 'No inflation column found'}
        
        inflation_data = data[inflation_col].dropna()
        
        # Basic statistics
        stats = {
            'mean': float(np.mean(inflation_data)),
            'median': float(np.median(inflation_data)),
            'std': float(np.std(inflation_data)),
            'min': float(np.min(inflation_data)),
            'max': float(np.max(inflation_data)),
            'count': len(inflation_data)
        }
        
        # Periods analysis
        periods = {
            'high_inflation_periods': len(inflation_data[inflation_data > 3.0]),
            'low_inflation_periods': len(inflation_data[inflation_data < 1.0]),
            'deflation_periods': len(inflation_data[inflation_data < 0]),
            'stable_periods': len(inflation_data[(inflation_data >= 1.0) & (inflation_data <= 3.0)])
        }
        
        # Volatility analysis
        volatility = {
            'coefficient_variation': stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else 0,
            'range': stats['max'] - stats['min'],
            'iqr': float(np.percentile(inflation_data, 75) - np.percentile(inflation_data, 25))
        }
        
        # Trend analysis
        if len(inflation_data) >= 12:
            recent_12m = inflation_data.tail(12)
            previous_12m = inflation_data.tail(24).head(12) if len(inflation_data) >= 24 else inflation_data.head(12)
            
            trend = {
                'recent_12m_avg': float(np.mean(recent_12m)),
                'previous_12m_avg': float(np.mean(previous_12m)),
                'trend_direction': 'increasing' if np.mean(recent_12m) > np.mean(previous_12m) else 'decreasing',
                'trend_magnitude': float(abs(np.mean(recent_12m) - np.mean(previous_12m)))
            }
        else:
            trend = {'insufficient_data': True}
        
        return {
            'statistics': stats,
            'periods': periods,
            'volatility': volatility,
            'trend': trend
        }
    
    def _analyze_predictions(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze prediction results."""
        pred_values = predictions['predicted_inflation'].values
        
        # Basic prediction statistics
        pred_stats = {
            'mean': float(np.mean(pred_values)),
            'median': float(np.median(pred_values)),
            'std': float(np.std(pred_values)),
            'min': float(np.min(pred_values)),
            'max': float(np.max(pred_values)),
            'count': len(pred_values)
        }
        
        # Prediction trajectory
        trajectory = {
            'initial_prediction': float(pred_values[0]) if len(pred_values) > 0 else 0,
            'final_prediction': float(pred_values[-1]) if len(pred_values) > 0 else 0,
            'overall_trend': 'increasing' if pred_values[-1] > pred_values[0] else 'decreasing' if len(pred_values) > 1 else 'stable',
            'volatility': float(np.std(np.diff(pred_values))) if len(pred_values) > 1 else 0
        }
        
        # Confidence analysis
        confidence_analysis = {}
        if 'confidence_lower' in predictions.columns and 'confidence_upper' in predictions.columns:
            ci_width = predictions['confidence_upper'] - predictions['confidence_lower']
            confidence_analysis = {
                'avg_confidence_width': float(np.mean(ci_width)),
                'max_confidence_width': float(np.max(ci_width)),
                'min_confidence_width': float(np.min(ci_width)),
                'confidence_trend': 'increasing' if ci_width.iloc[-1] > ci_width.iloc[0] else 'decreasing' if len(ci_width) > 1 else 'stable'
            }
        
        # Economic classification
        economic_periods = {
            'high_inflation_months': len(pred_values[pred_values > 3.0]),
            'moderate_inflation_months': len(pred_values[(pred_values >= 1.0) & (pred_values <= 3.0)]),
            'low_inflation_months': len(pred_values[(pred_values >= 0) & (pred_values < 1.0)]),
            'deflation_months': len(pred_values[pred_values < 0])
        }
        
        return {
            'statistics': pred_stats,
            'trajectory': trajectory,
            'confidence': confidence_analysis,
            'economic_periods': economic_periods
        }
    
    def _generate_economic_interpretation(self, historical: Dict[str, Any], 
                                        predictions: Dict[str, Any],
                                        model_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate economic interpretation of results."""
        interpretation = {
            'inflation_regime': self._classify_inflation_regime(historical, predictions),
            'economic_outlook': self._assess_economic_outlook(predictions),
            'policy_implications': self._assess_policy_implications(historical, predictions),
            'risk_assessment': self._assess_risks(historical, predictions),
            'model_reliability': self._assess_model_reliability(model_results) if model_results else {}
        }
        
        return interpretation
    
    def _classify_inflation_regime(self, historical: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, str]:
        """Classify the inflation regime based on historical and predicted data."""
        hist_mean = historical.get('statistics', {}).get('mean', 0)
        pred_mean = predictions.get('statistics', {}).get('mean', 0)
        
        # Historical regime
        if hist_mean < 0:
            hist_regime = "deflacionario"
        elif hist_mean < 1:
            hist_regime = "inflación muy baja"
        elif hist_mean <= 2:
            hist_regime = "inflación baja y estable"
        elif hist_mean <= 4:
            hist_regime = "inflación moderada"
        else:
            hist_regime = "inflación alta"
        
        # Predicted regime
        if pred_mean < 0:
            pred_regime = "deflacionario"
        elif pred_mean < 1:
            pred_regime = "inflación muy baja"
        elif pred_mean <= 2:
            pred_regime = "inflación baja y estable"
        elif pred_mean <= 4:
            pred_regime = "inflación moderada"
        else:
            pred_regime = "inflación alta"
        
        return {
            'historical_regime': hist_regime,
            'predicted_regime': pred_regime,
            'regime_change': 'Sí' if hist_regime != pred_regime else 'No'
        }
    
    def _assess_economic_outlook(self, predictions: Dict[str, Any]) -> Dict[str, str]:
        """Assess economic outlook based on predictions."""
        trajectory = predictions.get('trajectory', {})
        stats = predictions.get('statistics', {})
        
        trend = trajectory.get('overall_trend', 'stable')
        volatility = trajectory.get('volatility', 0)
        mean_pred = stats.get('mean', 0)
        
        if trend == 'increasing' and mean_pred > 3:
            outlook = "Presiones inflacionarias crecientes"
            risk_level = "Alto"
        elif trend == 'decreasing' and mean_pred < 1:
            outlook = "Riesgo de deflación"
            risk_level = "Moderado-Alto"
        elif 1 <= mean_pred <= 2:
            outlook = "Estabilidad de precios"
            risk_level = "Bajo"
        else:
            outlook = "Inflación moderada"
            risk_level = "Moderado"
        
        volatility_assessment = "Alta" if volatility > 0.5 else "Moderada" if volatility > 0.2 else "Baja"
        
        return {
            'outlook': outlook,
            'risk_level': risk_level,
            'volatility_assessment': volatility_assessment,
            'stability': "Estable" if volatility < 0.3 else "Inestable"
        }
    
    def _assess_policy_implications(self, historical: Dict[str, Any], predictions: Dict[str, Any]) -> List[str]:
        """Assess monetary policy implications."""
        implications = []
        
        pred_mean = predictions.get('statistics', {}).get('mean', 0)
        trend = predictions.get('trajectory', {}).get('overall_trend', 'stable')
        
        # ECB target is around 2%
        if pred_mean < 1:
            implications.append("Política monetaria expansiva podría ser necesaria para estimular la inflación")
        elif pred_mean > 3:
            implications.append("Política monetaria restrictiva podría ser necesaria para controlar la inflación")
        else:
            implications.append("Inflación cerca del objetivo del BCE (2%), política monetaria neutral apropiada")
        
        if trend == 'increasing':
            implications.append("Tendencia creciente sugiere vigilancia de presiones inflacionarias")
        elif trend == 'decreasing':
            implications.append("Tendencia decreciente requiere monitoreo de riesgos deflacionarios")
        
        volatility = predictions.get('trajectory', {}).get('volatility', 0)
        if volatility > 0.5:
            implications.append("Alta volatilidad sugiere incertidumbre económica elevada")
        
        return implications
    
    def _assess_risks(self, historical: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, List[str]]:
        """Assess economic risks based on analysis."""
        risks = {
            'upside_risks': [],
            'downside_risks': [],
            'structural_risks': []
        }
        
        pred_mean = predictions.get('statistics', {}).get('mean', 0)
        pred_max = predictions.get('statistics', {}).get('max', 0)
        pred_min = predictions.get('statistics', {}).get('min', 0)
        
        # Upside risks (higher inflation)
        if pred_max > 4:
            risks['upside_risks'].append("Riesgo de inflación alta en algunos períodos")
        if predictions.get('trajectory', {}).get('overall_trend') == 'increasing':
            risks['upside_risks'].append("Tendencia creciente podría acelerar más de lo previsto")
        
        # Downside risks (lower inflation/deflation)
        if pred_min < 0:
            risks['downside_risks'].append("Riesgo de períodos deflacionarios")
        if pred_mean < 1:
            risks['downside_risks'].append("Inflación persistentemente baja")
        
        # Structural risks
        volatility = predictions.get('trajectory', {}).get('volatility', 0)
        if volatility > 0.5:
            risks['structural_risks'].append("Alta volatilidad indica incertidumbre estructural")
        
        confidence = predictions.get('confidence', {})
        if confidence.get('avg_confidence_width', 0) > 2:
            risks['structural_risks'].append("Amplios intervalos de confianza indican alta incertidumbre")
        
        return risks
    
    def _assess_model_reliability(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model reliability based on performance metrics."""
        reliability = {
            'overall_assessment': 'No disponible',
            'best_model': 'No identificado',
            'performance_summary': {}
        }
        
        if not model_results:
            return reliability
        
        # Find best performing model
        best_model = None
        best_mae = float('inf')
        
        for model_name, results in model_results.items():
            if results.get('status') == 'success':
                metrics = results.get('metrics', {})
                mae = metrics.get('MAE', float('inf'))
                if mae < best_mae:
                    best_mae = mae
                    best_model = model_name
        
        if best_model:
            reliability['best_model'] = best_model.replace('_', ' ').title()
            best_metrics = model_results[best_model].get('metrics', {})
            
            # Assess performance
            mae = best_metrics.get('MAE', 0)
            mape = best_metrics.get('MAPE', 0)
            
            if mae < 0.5 and mape < 10:
                assessment = "Excelente"
            elif mae < 1.0 and mape < 20:
                assessment = "Buena"
            elif mae < 2.0 and mape < 30:
                assessment = "Moderada"
            else:
                assessment = "Limitada"
            
            reliability['overall_assessment'] = assessment
            reliability['performance_summary'] = {
                'MAE': f"{mae:.3f}",
                'RMSE': f"{best_metrics.get('RMSE', 0):.3f}",
                'MAPE': f"{mape:.1f}%"
            }
        
        return reliability
    
    def _generate_conclusions(self, historical: Dict[str, Any], 
                            predictions: Dict[str, Any],
                            interpretation: Dict[str, Any]) -> List[str]:
        """Generate conclusions about inflation trends."""
        conclusions = []
        
        # Historical conclusions
        hist_stats = historical.get('statistics', {})
        if hist_stats.get('mean', 0) > 0:
            conclusions.append(f"La inflación histórica promedio fue de {hist_stats['mean']:.2f}%, "
                             f"con una volatilidad de {hist_stats['std']:.2f}%")
        
        # Prediction conclusions
        pred_stats = predictions.get('statistics', {})
        trajectory = predictions.get('trajectory', {})
        
        conclusions.append(f"Las predicciones indican una inflación promedio de {pred_stats.get('mean', 0):.2f}% "
                         f"para los próximos 12 meses")
        
        if trajectory.get('overall_trend') == 'increasing':
            conclusions.append("Se espera una tendencia creciente en la inflación durante el período de predicción")
        elif trajectory.get('overall_trend') == 'decreasing':
            conclusions.append("Se espera una tendencia decreciente en la inflación durante el período de predicción")
        else:
            conclusions.append("Se espera una inflación relativamente estable durante el período de predicción")
        
        # Regime conclusions
        regime = interpretation.get('inflation_regime', {})
        if regime.get('regime_change') == 'Sí':
            conclusions.append(f"Se anticipa un cambio de régimen inflacionario: "
                             f"de {regime.get('historical_regime', 'N/A')} "
                             f"a {regime.get('predicted_regime', 'N/A')}")
        
        # Risk conclusions
        outlook = interpretation.get('economic_outlook', {})
        risk_level = outlook.get('risk_level', 'Desconocido')
        conclusions.append(f"El nivel de riesgo económico se evalúa como: {risk_level}")
        
        return conclusions
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate policy and economic recommendations."""
        recommendations = []
        
        interpretation = analysis.get('economic_interpretation', {})
        predictions = analysis.get('prediction_analysis', {})
        
        # Policy recommendations
        policy_implications = interpretation.get('policy_implications', [])
        if policy_implications:
            recommendations.extend(policy_implications)
        
        # Monitoring recommendations
        outlook = interpretation.get('economic_outlook', {})
        if outlook.get('volatility_assessment') == 'Alta':
            recommendations.append("Se recomienda monitoreo frecuente debido a la alta volatilidad esperada")
        
        confidence = predictions.get('confidence', {})
        if confidence.get('avg_confidence_width', 0) > 2:
            recommendations.append("Los amplios intervalos de confianza sugieren cautela en la interpretación")
        
        # Risk management recommendations
        risks = interpretation.get('risk_assessment', {})
        if risks.get('upside_risks'):
            recommendations.append("Preparar medidas preventivas contra presiones inflacionarias")
        if risks.get('downside_risks'):
            recommendations.append("Considerar estímulos económicos ante riesgos deflacionarios")
        
        return recommendations
    
    def create_technical_report(self, analysis: Dict[str, Any], 
                              visualizations: Dict[str, str],
                              model_results: Optional[Dict[str, Any]] = None,
                              output_filename: Optional[str] = None) -> str:
        """
        Create technical report for PDF generation using reportlab.
        
        Generates a comprehensive PDF report with methodology, results,
        and economic analysis.
        
        Args:
            analysis (Dict[str, Any]): Economic analysis results
            visualizations (Dict[str, str]): Dictionary of plot file paths
            model_results (Dict[str, Any], optional): Model evaluation results
            output_filename (str, optional): Output PDF filename
            
        Returns:
            str: Path to generated PDF report
        """
        if not _REPORTLAB_AVAILABLE:
            raise ImportError("reportlab is required for PDF generation. "
                            "Install with: pip install reportlab")
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"informe_tecnico_inflacion_{timestamp}.pdf"
        
        output_path = self.reports_dir / output_filename
        
        self.logger.info(f"Creating technical PDF report: {output_path}")
        
        try:
            # Create PDF document
            doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                                  rightMargin=72, leftMargin=72,
                                  topMargin=72, bottomMargin=18)
            
            # Build story (content)
            story = []
            styles = getSampleStyleSheet()
            
            # Custom styles
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=18,
                spaceAfter=30,
                alignment=1,  # Center
                textColor=colors.darkblue
            )
            
            heading_style = ParagraphStyle(
                'CustomHeading',
                parent=styles['Heading2'],
                fontSize=14,
                spaceAfter=12,
                textColor=colors.darkblue
            )
            
            # Title page
            story.append(Paragraph("INFORME TÉCNICO", title_style))
            story.append(Paragraph("Predicción de Inflación en España", title_style))
            story.append(Spacer(1, 20))
            
            # Date and summary
            story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']))
            story.append(Paragraph(f"Generado por: Sistema de Predicción de Inflación IA", styles['Normal']))
            story.append(Spacer(1, 30))
            
            # Executive Summary
            story.append(Paragraph("RESUMEN EJECUTIVO", heading_style))
            
            pred_analysis = analysis.get('prediction_analysis', {})
            pred_stats = pred_analysis.get('statistics', {})
            
            summary_text = f"""
            Este informe presenta los resultados del análisis predictivo de la inflación en España 
            utilizando modelos de inteligencia artificial. Las predicciones indican una inflación 
            promedio de {pred_stats.get('mean', 0):.2f}% para los próximos 12 meses, con un rango 
            de {pred_stats.get('min', 0):.2f}% a {pred_stats.get('max', 0):.2f}%.
            """
            story.append(Paragraph(summary_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Methodology section
            story.append(Paragraph("1. METODOLOGÍA", heading_style))
            
            methodology_text = """
            <b>Fuente de Datos:</b> Instituto Nacional de Estadística (INE) - Índice de Precios al Consumo (IPC)<br/>
            <b>Modelos Utilizados:</b> ARIMA, Random Forest, LSTM (Long Short-Term Memory)<br/>
            <b>Período de Análisis:</b> Datos históricos desde 2002<br/>
            <b>Horizonte de Predicción:</b> 12 meses<br/>
            <b>Métricas de Evaluación:</b> MAE (Error Absoluto Medio), RMSE (Raíz del Error Cuadrático Medio), MAPE (Error Porcentual Absoluto Medio)
            """
            story.append(Paragraph(methodology_text, styles['Normal']))
            story.append(Spacer(1, 20))
            
            # Model Performance section
            if model_results:
                story.append(Paragraph("2. RENDIMIENTO DE MODELOS", heading_style))
                
                # Create performance table
                model_data = [['Modelo', 'MAE', 'RMSE', 'MAPE (%)', 'Estado']]
                
                for model_name, results in model_results.items():
                    if results.get('status') == 'success':
                        metrics = results.get('metrics', {})
                        model_data.append([
                            model_name.replace('_', ' ').title(),
                            f"{metrics.get('MAE', 0):.3f}",
                            f"{metrics.get('RMSE', 0):.3f}",
                            f"{metrics.get('MAPE', 0):.1f}",
                            "Exitoso"
                        ])
                    else:
                        model_data.append([
                            model_name.replace('_', ' ').title(),
                            "N/A", "N/A", "N/A", "Error"
                        ])
                
                model_table = Table(model_data)
                model_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(model_table)
                story.append(Spacer(1, 20))
            
            # Add visualizations
            story.append(Paragraph("3. ANÁLISIS VISUAL", heading_style))
            
            for plot_type, plot_path in visualizations.items():
                if Path(plot_path).exists():
                    try:
                        # Add plot title
                        plot_titles = {
                            'time_series': 'Evolución Histórica y Predicciones',
                            'model_comparison': 'Comparación de Rendimiento de Modelos',
                            'prediction_distribution': 'Distribución de Predicciones',
                            'confidence_intervals': 'Intervalos de Confianza',
                            'historical_analysis': 'Análisis Histórico',
                            'seasonal_decomposition': 'Descomposición Estacional'
                        }
                        
                        title = plot_titles.get(plot_type, plot_type.replace('_', ' ').title())
                        story.append(Paragraph(f"3.{len([p for p in visualizations.keys() if visualizations[p] and Path(visualizations[p]).exists() and list(visualizations.keys()).index(p) <= list(visualizations.keys()).index(plot_type)])}. {title}", styles['Heading3']))
                        
                        # Add image
                        img = Image(plot_path, width=6*inch, height=4*inch)
                        story.append(img)
                        story.append(Spacer(1, 15))
                        
                    except Exception as e:
                        self.logger.warning(f"Could not add plot {plot_type}: {e}")
            
            # Economic Analysis section
            story.append(Paragraph("4. ANÁLISIS ECONÓMICO", heading_style))
            
            # Historical analysis
            hist_analysis = analysis.get('historical_analysis', {})
            hist_stats = hist_analysis.get('statistics', {})
            
            if hist_stats:
                hist_text = f"""
                <b>Análisis Histórico:</b><br/>
                • Inflación promedio histórica: {hist_stats.get('mean', 0):.2f}%<br/>
                • Desviación estándar: {hist_stats.get('std', 0):.2f}%<br/>
                • Rango: {hist_stats.get('min', 0):.2f}% - {hist_stats.get('max', 0):.2f}%<br/>
                • Número de observaciones: {hist_stats.get('count', 0)}
                """
                story.append(Paragraph(hist_text, styles['Normal']))
                story.append(Spacer(1, 10))
            
            # Prediction analysis
            if pred_stats:
                pred_text = f"""
                <b>Análisis de Predicciones:</b><br/>
                • Inflación promedio predicha: {pred_stats.get('mean', 0):.2f}%<br/>
                • Desviación estándar: {pred_stats.get('std', 0):.2f}%<br/>
                • Rango predicho: {pred_stats.get('min', 0):.2f}% - {pred_stats.get('max', 0):.2f}%<br/>
                • Horizonte de predicción: {pred_stats.get('count', 0)} meses
                """
                story.append(Paragraph(pred_text, styles['Normal']))
                story.append(Spacer(1, 10))
            
            # Economic interpretation
            interpretation = analysis.get('economic_interpretation', {})
            
            # Inflation regime
            regime = interpretation.get('inflation_regime', {})
            if regime:
                regime_text = f"""
                <b>Régimen Inflacionario:</b><br/>
                • Régimen histórico: {regime.get('historical_regime', 'N/A')}<br/>
                • Régimen predicho: {regime.get('predicted_regime', 'N/A')}<br/>
                • Cambio de régimen: {regime.get('regime_change', 'N/A')}
                """
                story.append(Paragraph(regime_text, styles['Normal']))
                story.append(Spacer(1, 10))
            
            # Economic outlook
            outlook = interpretation.get('economic_outlook', {})
            if outlook:
                outlook_text = f"""
                <b>Perspectiva Económica:</b><br/>
                • Evaluación: {outlook.get('outlook', 'N/A')}<br/>
                • Nivel de riesgo: {outlook.get('risk_level', 'N/A')}<br/>
                • Volatilidad: {outlook.get('volatility_assessment', 'N/A')}<br/>
                • Estabilidad: {outlook.get('stability', 'N/A')}
                """
                story.append(Paragraph(outlook_text, styles['Normal']))
                story.append(Spacer(1, 20))
            
            # Conclusions section
            story.append(Paragraph("5. CONCLUSIONES", heading_style))
            
            conclusions = analysis.get('conclusions', [])
            if conclusions:
                for i, conclusion in enumerate(conclusions, 1):
                    story.append(Paragraph(f"{i}. {conclusion}", styles['Normal']))
                    story.append(Spacer(1, 8))
            else:
                story.append(Paragraph("No se generaron conclusiones específicas.", styles['Normal']))
            
            story.append(Spacer(1, 20))
            
            # Recommendations section
            story.append(Paragraph("6. RECOMENDACIONES", heading_style))
            
            recommendations = analysis.get('recommendations', [])
            if recommendations:
                for i, recommendation in enumerate(recommendations, 1):
                    story.append(Paragraph(f"{i}. {recommendation}", styles['Normal']))
                    story.append(Spacer(1, 8))
            else:
                story.append(Paragraph("No se generaron recomendaciones específicas.", styles['Normal']))
            
            # Risk Assessment section
            risks = interpretation.get('risk_assessment', {})
            if risks:
                story.append(Spacer(1, 20))
                story.append(Paragraph("7. EVALUACIÓN DE RIESGOS", heading_style))
                
                upside_risks = risks.get('upside_risks', [])
                if upside_risks:
                    story.append(Paragraph("<b>Riesgos al Alza:</b>", styles['Normal']))
                    for risk in upside_risks:
                        story.append(Paragraph(f"• {risk}", styles['Normal']))
                    story.append(Spacer(1, 8))
                
                downside_risks = risks.get('downside_risks', [])
                if downside_risks:
                    story.append(Paragraph("<b>Riesgos a la Baja:</b>", styles['Normal']))
                    for risk in downside_risks:
                        story.append(Paragraph(f"• {risk}", styles['Normal']))
                    story.append(Spacer(1, 8))
                
                structural_risks = risks.get('structural_risks', [])
                if structural_risks:
                    story.append(Paragraph("<b>Riesgos Estructurales:</b>", styles['Normal']))
                    for risk in structural_risks:
                        story.append(Paragraph(f"• {risk}", styles['Normal']))
            
            # Footer
            story.append(Spacer(1, 30))
            story.append(Paragraph("---", styles['Normal']))
            story.append(Paragraph(f"Informe generado automáticamente el {datetime.now().strftime('%d/%m/%Y a las %H:%M')}", 
                                 styles['Normal']))
            
            # Build PDF
            doc.build(story)
            
            self.logger.info(f"Technical report created successfully: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Error creating technical report: {e}")
            raise
    
    def export_code_screenshots(self, source_dir: str = "src", 
                              output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Export code screenshots for process documentation.
        
        Creates documentation of the source code by capturing
        code snippets and saving them as formatted text files.
        
        Args:
            source_dir (str): Directory containing source code
            output_dir (str, optional): Output directory for code documentation
            
        Returns:
            Dict[str, str]: Dictionary mapping file names to documentation paths
        """
        if output_dir is None:
            output_dir = self.reports_dir / "code_documentation"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Exporting code documentation from {source_dir} to {output_dir}")
        
        documented_files = {}
        source_path = Path(source_dir)
        
        if not source_path.exists():
            self.logger.warning(f"Source directory {source_dir} does not exist")
            return documented_files
        
        try:
            # Find all Python files
            python_files = list(source_path.glob("*.py"))
            
            for py_file in python_files:
                try:
                    # Read source code
                    with open(py_file, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                    
                    # Create formatted documentation
                    doc_content = self._format_code_documentation(py_file.name, code_content)
                    
                    # Save documentation
                    doc_filename = f"{py_file.stem}_documentation.txt"
                    doc_path = output_dir / doc_filename
                    
                    with open(doc_path, 'w', encoding='utf-8') as f:
                        f.write(doc_content)
                    
                    documented_files[py_file.name] = str(doc_path)
                    self.logger.info(f"Documented {py_file.name}")
                    
                except Exception as e:
                    self.logger.warning(f"Could not document {py_file}: {e}")
            
            # Create summary documentation
            summary_path = self._create_code_summary(documented_files, output_dir)
            if summary_path:
                documented_files['_summary'] = summary_path
            
            # Create process flow documentation
            process_path = self._create_process_documentation(output_dir)
            if process_path:
                documented_files['_process_flow'] = process_path
            
            self.logger.info(f"Code documentation completed. {len(documented_files)} files documented.")
            return documented_files
            
        except Exception as e:
            self.logger.error(f"Error exporting code documentation: {e}")
            raise
    
    def _format_code_documentation(self, filename: str, code_content: str) -> str:
        """Format code content for documentation."""
        lines = code_content.split('\n')
        
        doc_lines = [
            "=" * 80,
            f"DOCUMENTACIÓN DE CÓDIGO: {filename}",
            "=" * 80,
            f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            f"Número de líneas: {len(lines)}",
            "",
            "CONTENIDO DEL ARCHIVO:",
            "-" * 40,
            ""
        ]
        
        # Add line numbers and content
        for i, line in enumerate(lines, 1):
            doc_lines.append(f"{i:4d}: {line}")
        
        doc_lines.extend([
            "",
            "-" * 40,
            "FIN DEL ARCHIVO",
            "=" * 80
        ])
        
        return '\n'.join(doc_lines)
    
    def _create_code_summary(self, documented_files: Dict[str, str], output_dir: Path) -> Optional[str]:
        """Create a summary of all documented code files."""
        try:
            summary_content = [
                "=" * 80,
                "RESUMEN DE DOCUMENTACIÓN DE CÓDIGO",
                "=" * 80,
                f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
                f"Total de archivos documentados: {len([f for f in documented_files.keys() if not f.startswith('_')])}",
                "",
                "ARCHIVOS DOCUMENTADOS:",
                "-" * 40
            ]
            
            for filename, doc_path in documented_files.items():
                if not filename.startswith('_'):
                    summary_content.append(f"• {filename} -> {Path(doc_path).name}")
            
            summary_content.extend([
                "",
                "ESTRUCTURA DEL SISTEMA:",
                "-" * 40,
                "• ine_extractor.py: Descarga de datos del INE",
                "• data_cleaner.py: Limpieza y procesamiento de datos",
                "• feature_engineering.py: Ingeniería de características",
                "• model_trainer.py: Entrenamiento de modelos de IA",
                "• predictor.py: Generación de predicciones",
                "• report_generator.py: Generación de informes y visualizaciones",
                "",
                "FLUJO DE PROCESAMIENTO:",
                "-" * 40,
                "1. Extracción de datos (INE) -> datos brutos",
                "2. Limpieza de datos -> datos procesados",
                "3. Ingeniería de características -> características para ML",
                "4. Entrenamiento de modelos -> modelos entrenados",
                "5. Generación de predicciones -> predicciones futuras",
                "6. Generación de informes -> análisis y visualizaciones",
                "",
                "=" * 80
            ])
            
            summary_path = output_dir / "resumen_documentacion.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(summary_content))
            
            return str(summary_path)
            
        except Exception as e:
            self.logger.warning(f"Could not create code summary: {e}")
            return None
    
    def _create_process_documentation(self, output_dir: Path) -> Optional[str]:
        """Create process flow documentation."""
        try:
            process_content = [
                "=" * 80,
                "DOCUMENTACIÓN DEL PROCESO DE PREDICCIÓN DE INFLACIÓN",
                "=" * 80,
                f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
                "",
                "DESCRIPCIÓN GENERAL:",
                "-" * 40,
                "Sistema automatizado para la predicción de la tasa de inflación en España",
                "utilizando datos del Instituto Nacional de Estadística (INE) y modelos de",
                "inteligencia artificial (ARIMA, Random Forest, LSTM).",
                "",
                "COMPONENTES PRINCIPALES:",
                "-" * 40,
                "",
                "1. EXTRACTOR DE DATOS INE (ine_extractor.py)",
                "   • Conecta con la API del INE",
                "   • Descarga datos de IPC general, por grupos e IPCA",
                "   • Maneja errores de conexión con reintentos",
                "   • Guarda datos en formato CSV",
                "",
                "2. PROCESADOR DE DATOS (data_cleaner.py)",
                "   • Carga datos brutos desde CSV",
                "   • Detecta y corrige valores faltantes",
                "   • Identifica y trata outliers estadísticos",
                "   • Normaliza fechas y calcula tasas de inflación",
                "",
                "3. INGENIERO DE CARACTERÍSTICAS (feature_engineering.py)",
                "   • Crea características lag (1, 3, 6, 12 meses)",
                "   • Genera medias móviles y componentes estacionales",
                "   • Prepara datos para modelos de machine learning",
                "",
                "4. ENTRENADOR DE MODELOS (model_trainer.py)",
                "   • Implementa modelos ARIMA, Random Forest y LSTM",
                "   • Divide datos en entrenamiento y validación",
                "   • Evalúa modelos con métricas MAE, RMSE, MAPE",
                "   • Selecciona el mejor modelo automáticamente",
                "",
                "5. PREDICTOR (predictor.py)",
                "   • Carga el mejor modelo entrenado",
                "   • Genera predicciones para 12 meses",
                "   • Calcula intervalos de confianza",
                "   • Valida y exporta predicciones",
                "",
                "6. GENERADOR DE INFORMES (report_generator.py)",
                "   • Crea visualizaciones con matplotlib/seaborn",
                "   • Genera análisis económico interpretativo",
                "   • Produce informes técnicos en PDF",
                "   • Documenta código fuente del proceso",
                "",
                "FLUJO DE EJECUCIÓN:",
                "-" * 40,
                "",
                "Paso 1: Descarga de Datos",
                "├── Conectar con API del INE",
                "├── Descargar IPC general (serie principal)",
                "├── Descargar IPC por grupos (sectorial)",
                "├── Descargar IPCA (armonizado europeo)",
                "└── Guardar en data/raw/",
                "",
                "Paso 2: Procesamiento de Datos",
                "├── Cargar datos brutos",
                "├── Limpiar valores faltantes (interpolación)",
                "├── Detectar outliers (IQR, Z-score)",
                "├── Calcular tasas de inflación",
                "└── Guardar en data/processed/",
                "",
                "Paso 3: Ingeniería de Características",
                "├── Crear características lag",
                "├── Generar medias móviles",
                "├── Añadir componentes estacionales",
                "└── Preparar datasets para ML",
                "",
                "Paso 4: Entrenamiento de Modelos",
                "├── Dividir datos (80% entrenamiento, 20% validación)",
                "├── Entrenar ARIMA (series temporales)",
                "├── Entrenar Random Forest (no lineal)",
                "├── Entrenar LSTM (redes neuronales)",
                "├── Evaluar con métricas de error",
                "└── Seleccionar mejor modelo",
                "",
                "Paso 5: Generación de Predicciones",
                "├── Cargar mejor modelo",
                "├── Generar predicciones 12 meses",
                "├── Calcular intervalos de confianza",
                "└── Exportar resultados",
                "",
                "Paso 6: Generación de Informes",
                "├── Crear visualizaciones (6 tipos de gráficos)",
                "├── Generar análisis económico interpretativo",
                "├── Producir informe técnico PDF",
                "└── Documentar proceso y código",
                "",
                "MÉTRICAS DE EVALUACIÓN:",
                "-" * 40,
                "• MAE (Mean Absolute Error): Error absoluto promedio",
                "• RMSE (Root Mean Square Error): Raíz del error cuadrático medio",
                "• MAPE (Mean Absolute Percentage Error): Error porcentual absoluto medio",
                "",
                "OUTPUTS GENERADOS:",
                "-" * 40,
                "• data/processed/: Datos limpios y procesados",
                "• models/: Modelos entrenados guardados",
                "• reports/: Visualizaciones, análisis y documentación",
                "• reports/informe_tecnico_*.pdf: Informe técnico completo",
                "",
                "CONFIGURACIÓN:",
                "-" * 40,
                "• config/config.yaml: Parámetros del sistema",
                "• requirements.txt: Dependencias de Python",
                "",
                "=" * 80
            ]
            
            process_path = output_dir / "documentacion_proceso.txt"
            with open(process_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(process_content))
            
            return str(process_path)
            
        except Exception as e:
            self.logger.warning(f"Could not create process documentation: {e}")
            return None    
def generate_economic_analysis(self, data: pd.DataFrame, predictions: pd.DataFrame,
                                 model_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate economic analysis with interpretation.
        
        Implements interpretation of results and economic conclusions
        about inflation trends identified in the data and predictions.
        
        Args:
            data (pd.DataFrame): Historical inflation data
            predictions (pd.DataFrame): Model predictions
            model_results (Dict[str, Any], optional): Model evaluation results
            
        Returns:
            Dict[str, Any]: Economic analysis with interpretations and conclusions
        """
        self.logger.info("Generating economic analysis and interpretation")
        
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'historical_analysis': {},
            'prediction_analysis': {},
            'economic_interpretation': {},
            'conclusions': [],
            'recommendations': []
        }
        
        try:
            # Historical data analysis
            analysis['historical_analysis'] = self._analyze_historical_data(data)
            
            # Prediction analysis
            analysis['prediction_analysis'] = self._analyze_predictions(predictions)
            
            # Economic interpretation
            analysis['economic_interpretation'] = self._generate_economic_interpretation(
                analysis['historical_analysis'], 
                analysis['prediction_analysis'],
                model_results
            )
            
            # Generate conclusions
            analysis['conclusions'] = self._generate_conclusions(
                analysis['historical_analysis'],
                analysis['prediction_analysis'],
                analysis['economic_interpretation']
            )
            
            # Generate recommendations
            analysis['recommendations'] = self._generate_recommendations(analysis)
            
            # Save analysis to JSON
            analysis_path = self.reports_dir / "economic_analysis.json"
            with open(analysis_path, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Economic analysis saved to {analysis_path}")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error generating economic analysis: {e}")
            raise
    
def _analyze_historical_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze historical inflation data."""
        inflation_col = self._find_inflation_column(data)
        if not inflation_col:
            return {'error': 'No inflation column found'}
        
        inflation_data = data[inflation_col].dropna()
        
        # Basic statistics
        stats = {
            'mean': float(np.mean(inflation_data)),
            'median': float(np.median(inflation_data)),
            'std': float(np.std(inflation_data)),
            'min': float(np.min(inflation_data)),
            'max': float(np.max(inflation_data)),
            'count': len(inflation_data)
        }
        
        # Periods analysis
        periods = {
            'high_inflation_periods': len(inflation_data[inflation_data > 3.0]),
            'low_inflation_periods': len(inflation_data[inflation_data < 1.0]),
            'deflation_periods': len(inflation_data[inflation_data < 0]),
            'stable_periods': len(inflation_data[(inflation_data >= 1.0) & (inflation_data <= 3.0)])
        }
        
        # Volatility analysis
        volatility = {
            'coefficient_variation': stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else 0,
            'range': stats['max'] - stats['min'],
            'iqr': float(np.percentile(inflation_data, 75) - np.percentile(inflation_data, 25))
        }
        
        # Trend analysis
        if len(inflation_data) >= 12:
            recent_12m = inflation_data.tail(12)
            previous_12m = inflation_data.tail(24).head(12) if len(inflation_data) >= 24 else inflation_data.head(12)
            
            trend = {
                'recent_12m_avg': float(np.mean(recent_12m)),
                'previous_12m_avg': float(np.mean(previous_12m)),
                'trend_direction': 'increasing' if np.mean(recent_12m) > np.mean(previous_12m) else 'decreasing',
                'trend_magnitude': float(abs(np.mean(recent_12m) - np.mean(previous_12m)))
            }
        else:
            trend = {'insufficient_data': True}
        
        return {
            'statistics': stats,
            'periods': periods,
            'volatility': volatility,
            'trend': trend
        }
    
def _analyze_predictions(self, predictions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze prediction results."""
        pred_values = predictions['predicted_inflation'].values
        
        # Basic prediction statistics
        pred_stats = {
            'mean': float(np.mean(pred_values)),
            'median': float(np.median(pred_values)),
            'std': float(np.std(pred_values)),
            'min': float(np.min(pred_values)),
            'max': float(np.max(pred_values)),
            'count': len(pred_values)
        }
        
        # Prediction trajectory
        trajectory = {
            'initial_prediction': float(pred_values[0]) if len(pred_values) > 0 else 0,
            'final_prediction': float(pred_values[-1]) if len(pred_values) > 0 else 0,
            'overall_trend': 'increasing' if pred_values[-1] > pred_values[0] else 'decreasing' if len(pred_values) > 1 else 'stable',
            'volatility': float(np.std(np.diff(pred_values))) if len(pred_values) > 1 else 0
        }
        
        # Confidence analysis
        confidence_analysis = {}
        if 'confidence_lower' in predictions.columns and 'confidence_upper' in predictions.columns:
            ci_width = predictions['confidence_upper'] - predictions['confidence_lower']
            confidence_analysis = {
                'avg_confidence_width': float(np.mean(ci_width)),
                'max_confidence_width': float(np.max(ci_width)),
                'min_confidence_width': float(np.min(ci_width)),
                'confidence_trend': 'increasing' if ci_width.iloc[-1] > ci_width.iloc[0] else 'decreasing' if len(ci_width) > 1 else 'stable'
            }
        
        # Economic classification
        economic_periods = {
            'high_inflation_months': len(pred_values[pred_values > 3.0]),
            'moderate_inflation_months': len(pred_values[(pred_values >= 1.0) & (pred_values <= 3.0)]),
            'low_inflation_months': len(pred_values[(pred_values >= 0) & (pred_values < 1.0)]),
            'deflation_months': len(pred_values[pred_values < 0])
        }
        
        return {
            'statistics': pred_stats,
            'trajectory': trajectory,
            'confidence': confidence_analysis,
            'economic_periods': economic_periods
        }
    
def _generate_economic_interpretation(self, historical: Dict[str, Any], 
                                        predictions: Dict[str, Any],
                                        model_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate economic interpretation of results."""
        interpretation = {
            'inflation_regime': self._classify_inflation_regime(historical, predictions),
            'economic_outlook': self._assess_economic_outlook(predictions),
            'policy_implications': self._assess_policy_implications(historical, predictions),
            'risk_assessment': self._assess_risks(historical, predictions),
            'model_reliability': self._assess_model_reliability(model_results) if model_results else {}
        }
        
        return interpretation
    
def _classify_inflation_regime(self, historical: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, str]:
        """Classify the inflation regime based on historical and predicted data."""
        hist_mean = historical.get('statistics', {}).get('mean', 0)
        pred_mean = predictions.get('statistics', {}).get('mean', 0)
        
        # Historical regime
        if hist_mean < 0:
            hist_regime = "deflacionario"
        elif hist_mean < 1:
            hist_regime = "inflación muy baja"
        elif hist_mean <= 2:
            hist_regime = "inflación baja y estable"
        elif hist_mean <= 4:
            hist_regime = "inflación moderada"
        else:
            hist_regime = "inflación alta"
        
        # Predicted regime
        if pred_mean < 0:
            pred_regime = "deflacionario"
        elif pred_mean < 1:
            pred_regime = "inflación muy baja"
        elif pred_mean <= 2:
            pred_regime = "inflación baja y estable"
        elif pred_mean <= 4:
            pred_regime = "inflación moderada"
        else:
            pred_regime = "inflación alta"
        
        return {
            'historical_regime': hist_regime,
            'predicted_regime': pred_regime,
            'regime_change': 'Sí' if hist_regime != pred_regime else 'No'
        }
    
def _assess_economic_outlook(self, predictions: Dict[str, Any]) -> Dict[str, str]:
        """Assess economic outlook based on predictions."""
        trajectory = predictions.get('trajectory', {})
        stats = predictions.get('statistics', {})
        
        trend = trajectory.get('overall_trend', 'stable')
        volatility = trajectory.get('volatility', 0)
        mean_pred = stats.get('mean', 0)
        
        if trend == 'increasing' and mean_pred > 3:
            outlook = "Presiones inflacionarias crecientes"
            risk_level = "Alto"
        elif trend == 'decreasing' and mean_pred < 1:
            outlook = "Riesgo de deflación"
            risk_level = "Moderado-Alto"
        elif 1 <= mean_pred <= 2:
            outlook = "Estabilidad de precios"
            risk_level = "Bajo"
        else:
            outlook = "Inflación moderada"
            risk_level = "Moderado"
        
        volatility_assessment = "Alta" if volatility > 0.5 else "Moderada" if volatility > 0.2 else "Baja"
        
        return {
            'outlook': outlook,
            'risk_level': risk_level,
            'volatility_assessment': volatility_assessment,
            'stability': "Estable" if volatility < 0.3 else "Inestable"
        }
    
def _assess_policy_implications(self, historical: Dict[str, Any], predictions: Dict[str, Any]) -> List[str]:
        """Assess monetary policy implications."""
        implications = []
        
        pred_mean = predictions.get('statistics', {}).get('mean', 0)
        trend = predictions.get('trajectory', {}).get('overall_trend', 'stable')
        
        # ECB target is around 2%
        if pred_mean < 1:
            implications.append("Política monetaria expansiva podría ser necesaria para estimular la inflación")
        elif pred_mean > 3:
            implications.append("Política monetaria restrictiva podría ser necesaria para controlar la inflación")
        else:
            implications.append("Inflación cerca del objetivo del BCE (2%), política monetaria neutral apropiada")
        
        if trend == 'increasing':
            implications.append("Tendencia creciente sugiere vigilancia de presiones inflacionarias")
        elif trend == 'decreasing':
            implications.append("Tendencia decreciente requiere monitoreo de riesgos deflacionarios")
        
        volatility = predictions.get('trajectory', {}).get('volatility', 0)
        if volatility > 0.5:
            implications.append("Alta volatilidad sugiere incertidumbre económica elevada")
        
        return implications
    
def _assess_risks(self, historical: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, List[str]]:
        """Assess economic risks based on analysis."""
        risks = {
            'upside_risks': [],
            'downside_risks': [],
            'structural_risks': []
        }
        
        pred_mean = predictions.get('statistics', {}).get('mean', 0)
        pred_max = predictions.get('statistics', {}).get('max', 0)
        pred_min = predictions.get('statistics', {}).get('min', 0)
        
        # Upside risks (higher inflation)
        if pred_max > 4:
            risks['upside_risks'].append("Riesgo de inflación alta en algunos períodos")
        if predictions.get('trajectory', {}).get('overall_trend') == 'increasing':
            risks['upside_risks'].append("Tendencia creciente podría acelerar más de lo previsto")
        
        # Downside risks (lower inflation/deflation)
        if pred_min < 0:
            risks['downside_risks'].append("Riesgo de períodos deflacionarios")
        if pred_mean < 1:
            risks['downside_risks'].append("Inflación persistentemente baja")
        
        # Structural risks
        volatility = predictions.get('trajectory', {}).get('volatility', 0)
        if volatility > 0.5:
            risks['structural_risks'].append("Alta volatilidad indica incertidumbre estructural")
        
        confidence = predictions.get('confidence', {})
        if confidence.get('avg_confidence_width', 0) > 2:
            risks['structural_risks'].append("Amplios intervalos de confianza indican alta incertidumbre")
        
        return risks
    
def _assess_model_reliability(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess model reliability based on performance metrics."""
        reliability = {
            'overall_assessment': 'No disponible',
            'best_model': 'No identificado',
            'performance_summary': {}
        }
        
        if not model_results:
            return reliability
        
        # Find best performing model
        best_model = None
        best_mae = float('inf')
        
        for model_name, results in model_results.items():
            if results.get('status') == 'success':
                metrics = results.get('metrics', {})
                mae = metrics.get('MAE', float('inf'))
                if mae < best_mae:
                    best_mae = mae
                    best_model = model_name
        
        if best_model:
            reliability['best_model'] = best_model.replace('_', ' ').title()
            best_metrics = model_results[best_model].get('metrics', {})
            
            # Assess performance
            mae = best_metrics.get('MAE', 0)
            mape = best_metrics.get('MAPE', 0)
            
            if mae < 0.5 and mape < 10:
                assessment = "Excelente"
            elif mae < 1.0 and mape < 20:
                assessment = "Buena"
            elif mae < 2.0 and mape < 30:
                assessment = "Moderada"
            else:
                assessment = "Limitada"
            
            reliability['overall_assessment'] = assessment
            reliability['performance_summary'] = {
                'MAE': f"{mae:.3f}",
                'RMSE': f"{best_metrics.get('RMSE', 0):.3f}",
                'MAPE': f"{mape:.1f}%"
            }
        
        return reliability
    
def _generate_conclusions(self, historical: Dict[str, Any], 
                            predictions: Dict[str, Any],
                            interpretation: Dict[str, Any]) -> List[str]:
        """Generate conclusions about inflation trends."""
        conclusions = []
        
        # Historical conclusions
        hist_stats = historical.get('statistics', {})
        if hist_stats.get('mean', 0) > 0:
            conclusions.append(f"La inflación histórica promedio fue de {hist_stats['mean']:.2f}%, "
                             f"con una volatilidad de {hist_stats['std']:.2f}%")
        
        # Prediction conclusions
        pred_stats = predictions.get('statistics', {})
        trajectory = predictions.get('trajectory', {})
        
        conclusions.append(f"Las predicciones indican una inflación promedio de {pred_stats.get('mean', 0):.2f}% "
                         f"para los próximos 12 meses")
        
        if trajectory.get('overall_trend') == 'increasing':
            conclusions.append("Se espera una tendencia creciente en la inflación durante el período de predicción")
        elif trajectory.get('overall_trend') == 'decreasing':
            conclusions.append("Se espera una tendencia decreciente en la inflación durante el período de predicción")
        else:
            conclusions.append("Se espera una inflación relativamente estable durante el período de predicción")
        
        # Regime conclusions
        regime = interpretation.get('inflation_regime', {})
        if regime.get('regime_change') == 'Sí':
            conclusions.append(f"Se anticipa un cambio de régimen inflacionario: "
                             f"de {regime.get('historical_regime', 'N/A')} "
                             f"a {regime.get('predicted_regime', 'N/A')}")
        
        # Risk conclusions
        outlook = interpretation.get('economic_outlook', {})
        risk_level = outlook.get('risk_level', 'Desconocido')
        conclusions.append(f"El nivel de riesgo económico se evalúa como: {risk_level}")
        
        return conclusions
    
def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate policy and economic recommendations."""
        recommendations = []
        
        interpretation = analysis.get('economic_interpretation', {})
        predictions = analysis.get('prediction_analysis', {})
        
        # Policy recommendations
        policy_implications = interpretation.get('policy_implications', [])
        if policy_implications:
            recommendations.extend(policy_implications)
        
        # Monitoring recommendations
        outlook = interpretation.get('economic_outlook', {})
        if outlook.get('volatility_assessment') == 'Alta':
            recommendations.append("Se recomienda monitoreo frecuente debido a la alta volatilidad esperada")
        
        confidence = predictions.get('confidence', {})
        if confidence.get('avg_confidence_width', 0) > 2:
            recommendations.append("Los amplios intervalos de confianza sugieren cautela en la interpretación")
        
        # Risk management recommendations
        risks = interpretation.get('risk_assessment', {})
        if risks.get('upside_risks'):
            recommendations.append("Preparar medidas preventivas contra presiones inflacionarias")
        if risks.get('downside_risks'):
            recommendations.append("Considerar estímulos económicos ante riesgos deflacionarios")
        
        return recommendations
    
def create_technical_report(self, analysis: Dict[str, Any], 
                            visualizations: Dict[str, str],
                            model_results: Optional[Dict[str, Any]] = None,
                            output_filename: Optional[str] = None) -> str:
    """
    Create technical report for PDF generation using reportlab.
    
    Generates a comprehensive PDF report with methodology, results,
    and economic analysis.
    
    Args:
        analysis (Dict[str, Any]): Economic analysis results
        visualizations (Dict[str, str]): Dictionary of plot file paths
        model_results (Dict[str, Any], optional): Model evaluation results
        output_filename (str, optional): Output PDF filename
        
    Returns:
        str: Path to generated PDF report
    """
    if not _REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is required for PDF generation. "
                        "Install with: pip install reportlab")
    
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"informe_tecnico_inflacion_{timestamp}.pdf"
    
    output_path = self.reports_dir / output_filename
    
    self.logger.info(f"Creating technical PDF report: {output_path}")
    
    try:
        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        # Build story (content)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Title page
        story.append(Paragraph("INFORME TÉCNICO", title_style))
        story.append(Paragraph("Predicción de Inflación en España", title_style))
        story.append(Spacer(1, 20))
        
        # Date and summary
        story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']))
        story.append(Paragraph(f"Generado por: Sistema de Predicción de Inflación IA", styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Executive Summary
        story.append(Paragraph("RESUMEN EJECUTIVO", heading_style))
        
        pred_analysis = analysis.get('prediction_analysis', {})
        pred_stats = pred_analysis.get('statistics', {})
        
        summary_text = f"""
        Este informe presenta los resultados del análisis predictivo de la inflación en España 
        utilizando modelos de inteligencia artificial. Las predicciones indican una inflación 
        promedio de {pred_stats.get('mean', 0):.2f}% para los próximos 12 meses, con un rango 
        de {pred_stats.get('min', 0):.2f}% a {pred_stats.get('max', 0):.2f}%.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Methodology section
        story.append(Paragraph("1. METODOLOGÍA", heading_style))
        
        methodology_text = """
        <b>Fuente de Datos:</b> Instituto Nacional de Estadística (INE) - Índice de Precios al Consumo (IPC)<br/>
        <b>Modelos Utilizados:</b> ARIMA, Random Forest, LSTM (Long Short-Term Memory)<br/>
        <b>Período de Análisis:</b> Datos históricos desde 2002<br/>
        <b>Horizonte de Predicción:</b> 12 meses<br/>
        <b>Métricas de Evaluación:</b> MAE (Error Absoluto Medio), RMSE (Raíz del Error Cuadrático Medio), MAPE (Error Porcentual Absoluto Medio)
        """
        story.append(Paragraph(methodology_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Model Performance section
        if model_results:
            story.append(Paragraph("2. RENDIMIENTO DE MODELOS", heading_style))
            
            # Create performance table
            model_data = [['Modelo', 'MAE', 'RMSE', 'MAPE (%)', 'Estado']]
            
            for model_name, results in model_results.items():
                if results.get('status') == 'success':
                    metrics = results.get('metrics', {})
                    model_data.append([
                        model_name.replace('_', ' ').title(),
                        f"{metrics.get('MAE', 0):.3f}",
                        f"{metrics.get('RMSE', 0):.3f}",
                        f"{metrics.get('MAPE', 0):.1f}",
                        "Exitoso"
                    ])
                else:
                    model_data.append([
                        model_name.replace('_', ' ').title(),
                        "N/A", "N/A", "N/A", "Error"
                    ])
            
            model_table = Table(model_data)
            model_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(model_table)
            story.append(Spacer(1, 20))
        
        # Add visualizations
        story.append(Paragraph("3. ANÁLISIS VISUAL", heading_style))
        
        for plot_type, plot_path in visualizations.items():
            if Path(plot_path).exists():
                try:
                    # Add plot title
                    plot_titles = {
                        'time_series': 'Evolución Histórica y Predicciones',
                        'model_comparison': 'Comparación de Rendimiento de Modelos',
                        'prediction_distribution': 'Distribución de Predicciones',
                        'confidence_intervals': 'Intervalos de Confianza',
                        'historical_analysis': 'Análisis Histórico',
                        'seasonal_decomposition': 'Descomposición Estacional'
                    }
                    
                    title = plot_titles.get(plot_type, plot_type.replace('_', ' ').title())
                    story.append(Paragraph(f"3.{len([p for p in visualizations.keys() if visualizations[p] and Path(visualizations[p]).exists() and list(visualizations.keys()).index(p) <= list(visualizations.keys()).index(plot_type)])}. {title}", styles['Heading3']))
                    
                    # Add image
                    img = Image(plot_path, width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 15))
                    
                except Exception as e:
                    self.logger.warning(f"Could not add plot {plot_type}: {e}")
        
        # Economic Analysis section
        story.append(Paragraph("4. ANÁLISIS ECONÓMICO", heading_style))
        
        # Historical analysis
        hist_analysis = analysis.get('historical_analysis', {})
        hist_stats = hist_analysis.get('statistics', {})
        
        if hist_stats:
            hist_text = f"""
            <b>Análisis Histórico:</b><br/>
            • Inflación promedio histórica: {hist_stats.get('mean', 0):.2f}%<br/>
            • Desviación estándar: {hist_stats.get('std', 0):.2f}%<br/>
            • Rango: {hist_stats.get('min', 0):.2f}% - {hist_stats.get('max', 0):.2f}%<br/>
            • Número de observaciones: {hist_stats.get('count', 0)}
            """
            story.append(Paragraph(hist_text, styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Prediction analysis
        if pred_stats:
            pred_text = f"""
            <b>Análisis de Predicciones:</b><br/>
            • Inflación promedio predicha: {pred_stats.get('mean', 0):.2f}%<br/>
            • Desviación estándar: {pred_stats.get('std', 0):.2f}%<br/>
            • Rango predicho: {pred_stats.get('min', 0):.2f}% - {pred_stats.get('max', 0):.2f}%<br/>
            • Horizonte de predicción: {pred_stats.get('count', 0)} meses
            """
            story.append(Paragraph(pred_text, styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Economic interpretation
        interpretation = analysis.get('economic_interpretation', {})
        
        # Inflation regime
        regime = interpretation.get('inflation_regime', {})
        if regime:
            regime_text = f"""
            <b>Régimen Inflacionario:</b><br/>
            • Régimen histórico: {regime.get('historical_regime', 'N/A')}<br/>
            • Régimen predicho: {regime.get('predicted_regime', 'N/A')}<br/>
            • Cambio de régimen: {regime.get('regime_change', 'N/A')}
            """
            story.append(Paragraph(regime_text, styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Economic outlook
        outlook = interpretation.get('economic_outlook', {})
        if outlook:
            outlook_text = f"""
            <b>Perspectiva Económica:</b><br/>
            • Evaluación: {outlook.get('outlook', 'N/A')}<br/>
            • Nivel de riesgo: {outlook.get('risk_level', 'N/A')}<br/>
            • Volatilidad: {outlook.get('volatility_assessment', 'N/A')}<br/>
            • Estabilidad: {outlook.get('stability', 'N/A')}
            """
            story.append(Paragraph(outlook_text, styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Conclusions section
        story.append(Paragraph("5. CONCLUSIONES", heading_style))
        
        conclusions = analysis.get('conclusions', [])
        if conclusions:
            for i, conclusion in enumerate(conclusions, 1):
                story.append(Paragraph(f"{i}. {conclusion}", styles['Normal']))
                story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("No se generaron conclusiones específicas.", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Recommendations section
        story.append(Paragraph("6. RECOMENDACIONES", heading_style))
        
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            for i, recommendation in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {recommendation}", styles['Normal']))
                story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("No se generaron recomendaciones específicas.", styles['Normal']))
        
        # Risk Assessment section
        risks = interpretation.get('risk_assessment', {})
        if risks:
            story.append(Spacer(1, 20))
            story.append(Paragraph("7. EVALUACIÓN DE RIESGOS", heading_style))
            
            upside_risks = risks.get('upside_risks', [])
            if upside_risks:
                story.append(Paragraph("<b>Riesgos al Alza:</b>", styles['Normal']))
                for risk in upside_risks:
                    story.append(Paragraph(f"• {risk}", styles['Normal']))
                story.append(Spacer(1, 8))
            
            downside_risks = risks.get('downside_risks', [])
            if downside_risks:
                story.append(Paragraph("<b>Riesgos a la Baja:</b>", styles['Normal']))
                for risk in downside_risks:
                    story.append(Paragraph(f"• {risk}", styles['Normal']))
                story.append(Spacer(1, 8))
            
            structural_risks = risks.get('structural_risks', [])
            if structural_risks:
                story.append(Paragraph("<b>Riesgos Estructurales:</b>", styles['Normal']))
                for risk in structural_risks:
                    story.append(Paragraph(f"• {risk}", styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("---", styles['Normal']))
        story.append(Paragraph(f"Informe generado automáticamente el {datetime.now().strftime('%d/%m/%Y a las %H:%M')}", 
                                styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        self.logger.info(f"Technical report created successfully: {output_path}")
        return str(output_path)
        
    except Exception as e:
        self.logger.error(f"Error creating technical report: {e}")
        raise

def export_code_screenshots(self, source_dir: str = "src", 
                            output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Export code screenshots for process documentation.
    
    Creates documentation of the source code by capturing
    code snippets and saving them as formatted text files.
    
    Args:
        source_dir (str): Directory containing source code
        output_dir (str, optional): Output directory for code documentation
        
    Returns:
        Dict[str, str]: Dictionary mapping file names to documentation paths
    """
    if output_dir is None:
        output_dir = self.reports_dir / "code_documentation"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    self.logger.info(f"Exporting code documentation from {source_dir} to {output_dir}")
    
    documented_files = {}
    source_path = Path(source_dir)
    
    if not source_path.exists():
        self.logger.warning(f"Source directory {source_dir} does not exist")
        return documented_files
    
    try:
        # Find all Python files
        python_files = list(source_path.glob("*.py"))
        
        for py_file in python_files:
            try:
                # Read source code
                with open(py_file, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                
                # Create formatted documentation
                doc_content = self._format_code_documentation(py_file.name, code_content)
                
                # Save documentation
                doc_filename = f"{py_file.stem}_documentation.txt"
                doc_path = output_dir / doc_filename
                
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(doc_content)
                
                documented_files[py_file.name] = str(doc_path)
                self.logger.info(f"Documented {py_file.name}")
                
            except Exception as e:
                self.logger.warning(f"Could not document {py_file}: {e}")
        
        # Create summary documentation
        summary_path = self._create_code_summary(documented_files, output_dir)
        if summary_path:
            documented_files['_summary'] = summary_path
        
        # Create process flow documentation
        process_path = self._create_process_documentation(output_dir)
        if process_path:
            documented_files['_process_flow'] = process_path
        
        self.logger.info(f"Code documentation completed. {len(documented_files)} files documented.")
        return documented_files
        
    except Exception as e:
        self.logger.error(f"Error exporting code documentation: {e}")
        raise

def _format_code_documentation(self, filename: str, code_content: str) -> str:
    """Format code content for documentation."""
    lines = code_content.split('\n')
    
    doc_lines = [
        "=" * 80,
        f"DOCUMENTACIÓN DE CÓDIGO: {filename}",
        "=" * 80,
        f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        f"Número de líneas: {len(lines)}",
        "",
        "CONTENIDO DEL ARCHIVO:",
        "-" * 40,
        ""
    ]
    
    # Add line numbers and content
    for i, line in enumerate(lines, 1):
        doc_lines.append(f"{i:4d}: {line}")
    
    doc_lines.extend([
        "",
        "-" * 40,
        "FIN DEL ARCHIVO",
        "=" * 80
    ])
    
    return '\n'.join(doc_lines)

def _create_code_summary(self, documented_files: Dict[str, str], output_dir: Path) -> Optional[str]:
    """Create a summary of all documented code files."""
    try:
        summary_content = [
            "=" * 80,
            "RESUMEN DE DOCUMENTACIÓN DE CÓDIGO",
            "=" * 80,
            f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            f"Total de archivos documentados: {len([f for f in documented_files.keys() if not f.startswith('_')])}",
            "",
            "ARCHIVOS DOCUMENTADOS:",
            "-" * 40
        ]
        
        for filename, doc_path in documented_files.items():
            if not filename.startswith('_'):
                summary_content.append(f"• {filename} -> {Path(doc_path).name}")
        
        summary_content.extend([
            "",
            "ESTRUCTURA DEL SISTEMA:",
            "-" * 40,
            "• ine_extractor.py: Descarga de datos del INE",
            "• data_cleaner.py: Limpieza y procesamiento de datos",
            "• feature_engineering.py: Ingeniería de características",
            "• model_trainer.py: Entrenamiento de modelos de IA",
            "• predictor.py: Generación de predicciones",
            "• report_generator.py: Generación de informes y visualizaciones",
            "",
            "FLUJO DE PROCESAMIENTO:",
            "-" * 40,
            "1. Extracción de datos (INE) -> datos brutos",
            "2. Limpieza de datos -> datos procesados",
            "3. Ingeniería de características -> características para ML",
            "4. Entrenamiento de modelos -> modelos entrenados",
            "5. Generación de predicciones -> predicciones futuras",
            "6. Generación de informes -> análisis y visualizaciones",
            "",
            "=" * 80
        ])
        
        summary_path = output_dir / "resumen_documentacion.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_content))
        
        return str(summary_path)
        
    except Exception as e:
        self.logger.warning(f"Could not create code summary: {e}")
        return None

def _create_process_documentation(self, output_dir: Path) -> Optional[str]:
    """Create process flow documentation."""
    try:
        process_content = [
            "=" * 80,
            "DOCUMENTACIÓN DEL PROCESO DE PREDICCIÓN DE INFLACIÓN",
            "=" * 80,
            f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            "",
            "DESCRIPCIÓN GENERAL:",
            "-" * 40,
            "Sistema automatizado para la predicción de la tasa de inflación en España",
            "utilizando datos del Instituto Nacional de Estadística (INE) y modelos de",
            "inteligencia artificial (ARIMA, Random Forest, LSTM).",
            "",
            "COMPONENTES PRINCIPALES:",
            "-" * 40,
            "",
            "1. EXTRACTOR DE DATOS INE (ine_extractor.py)",
            "   • Conecta con la API del INE",
            "   • Descarga datos de IPC general, por grupos e IPCA",
            "   • Maneja errores de conexión con reintentos",
            "   • Guarda datos en formato CSV",
            "",
            "2. PROCESADOR DE DATOS (data_cleaner.py)",
            "   • Carga datos brutos desde CSV",
            "   • Detecta y corrige valores faltantes",
            "   • Identifica y trata outliers estadísticos",
            "   • Normaliza fechas y calcula tasas de inflación",
            "",
            "3. INGENIERO DE CARACTERÍSTICAS (feature_engineering.py)",
            "   • Crea características lag (1, 3, 6, 12 meses)",
            "   • Genera medias móviles y componentes estacionales",
            "   • Prepara datos para modelos de machine learning",
            "",
            "4. ENTRENADOR DE MODELOS (model_trainer.py)",
            "   • Implementa modelos ARIMA, Random Forest y LSTM",
            "   • Divide datos en entrenamiento y validación",
            "   • Evalúa modelos con métricas MAE, RMSE, MAPE",
            "   • Selecciona el mejor modelo automáticamente",
            "",
            "5. PREDICTOR (predictor.py)",
            "   • Carga el mejor modelo entrenado",
            "   • Genera predicciones para 12 meses",
            "   • Calcula intervalos de confianza",
            "   • Valida y exporta predicciones",
            "",
            "6. GENERADOR DE INFORMES (report_generator.py)",
            "   • Crea visualizaciones con matplotlib/seaborn",
            "   • Genera análisis económico interpretativo",
            "   • Produce informes técnicos en PDF",
            "   • Documenta código fuente del proceso",
            "",
            "FLUJO DE EJECUCIÓN:",
            "-" * 40,
            "",
            "Paso 1: Descarga de Datos",
            "├── Conectar con API del INE",
            "├── Descargar IPC general (serie principal)",
            "├── Descargar IPC por grupos (sectorial)",
            "├── Descargar IPCA (armonizado europeo)",
            "└── Guardar en data/raw/",
            "",
            "Paso 2: Procesamiento de Datos",
            "├── Cargar datos brutos",
            "├── Limpiar valores faltantes (interpolación)",
            "├── Detectar outliers (IQR, Z-score)",
            "├── Calcular tasas de inflación",
            "└── Guardar en data/processed/",
            "",
            "Paso 3: Ingeniería de Características",
            "├── Crear características lag",
            "├── Generar medias móviles",
            "├── Añadir componentes estacionales",
            "└── Preparar datasets para ML",
            "",
            "Paso 4: Entrenamiento de Modelos",
            "├── Dividir datos (80% entrenamiento, 20% validación)",
            "├── Entrenar ARIMA (series temporales)",
            "├── Entrenar Random Forest (no lineal)",
            "├── Entrenar LSTM (redes neuronales)",
            "├── Evaluar con métricas de error",
            "└── Seleccionar mejor modelo",
            "",
            "Paso 5: Generación de Predicciones",
            "├── Cargar mejor modelo",
            "├── Generar predicciones 12 meses",
            "├── Calcular intervalos de confianza",
            "└── Exportar resultados",
            "",
            "Paso 6: Generación de Informes",
            "├── Crear visualizaciones (6 tipos de gráficos)",
            "├── Generar análisis económico interpretativo",
            "├── Producir informe técnico PDF",
            "└── Documentar proceso y código",
            "",
            "MÉTRICAS DE EVALUACIÓN:",
            "-" * 40,
            "• MAE (Mean Absolute Error): Error absoluto promedio",
            "• RMSE (Root Mean Square Error): Raíz del error cuadrático medio",
            "• MAPE (Mean Absolute Percentage Error): Error porcentual absoluto medio",
            "",
            "OUTPUTS GENERADOS:",
            "-" * 40,
            "• data/processed/: Datos limpios y procesados",
            "• models/: Modelos entrenados guardados",
            "• reports/: Visualizaciones, análisis y documentación",
            "• reports/informe_tecnico_*.pdf: Informe técnico completo",
            "",
            "CONFIGURACIÓN:",
            "-" * 40,
            "• config/config.yaml: Parámetros del sistema",
            "• requirements.txt: Dependencias de Python",
            "",
            "=" * 80
        ]
        
        process_path = output_dir / "documentacion_proceso.txt"
        with open(process_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(process_content))
        
        return str(process_path)
        
    except Exception as e:
        self.logger.warning(f"Could not create process documentation: {e}")
        return None   
def generate_economic_analysis(self, data: pd.DataFrame, predictions: pd.DataFrame,
                                model_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Generate economic analysis with interpretation.
    
    Implements interpretation of results and economic conclusions
    about inflation trends identified in the data and predictions.
    
    Args:
        data (pd.DataFrame): Historical inflation data
        predictions (pd.DataFrame): Model predictions
        model_results (Dict[str, Any], optional): Model evaluation results
        
    Returns:
        Dict[str, Any]: Economic analysis with interpretations and conclusions
    """
    self.logger.info("Generating economic analysis and interpretation")
    
    analysis = {
        'timestamp': datetime.now().isoformat(),
        'historical_analysis': {},
        'prediction_analysis': {},
        'economic_interpretation': {},
        'conclusions': [],
        'recommendations': []
    }
    
    try:
        # Historical data analysis
        analysis['historical_analysis'] = self._analyze_historical_data(data)
        
        # Prediction analysis
        analysis['prediction_analysis'] = self._analyze_predictions(predictions)
        
        # Economic interpretation
        analysis['economic_interpretation'] = self._generate_economic_interpretation(
            analysis['historical_analysis'], 
            analysis['prediction_analysis'],
            model_results
        )
        
        # Generate conclusions
        analysis['conclusions'] = self._generate_conclusions(
            analysis['historical_analysis'],
            analysis['prediction_analysis'],
            analysis['economic_interpretation']
        )
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        # Save analysis to JSON
        analysis_path = self.reports_dir / "economic_analysis.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Economic analysis saved to {analysis_path}")
        return analysis
        
    except Exception as e:
        self.logger.error(f"Error generating economic analysis: {e}")
        raise

def _analyze_historical_data(self, data: pd.DataFrame) -> Dict[str, Any]:
    """Analyze historical inflation data."""
    inflation_col = self._find_inflation_column(data)
    if not inflation_col:
        return {'error': 'No inflation column found'}
    
    inflation_data = data[inflation_col].dropna()
    
    # Basic statistics
    stats = {
        'mean': float(np.mean(inflation_data)),
        'median': float(np.median(inflation_data)),
        'std': float(np.std(inflation_data)),
        'min': float(np.min(inflation_data)),
        'max': float(np.max(inflation_data)),
        'count': len(inflation_data)
    }
    
    # Periods analysis
    periods = {
        'high_inflation_periods': len(inflation_data[inflation_data > 3.0]),
        'low_inflation_periods': len(inflation_data[inflation_data < 1.0]),
        'deflation_periods': len(inflation_data[inflation_data < 0]),
        'stable_periods': len(inflation_data[(inflation_data >= 1.0) & (inflation_data <= 3.0)])
    }
    
    # Volatility analysis
    volatility = {
        'coefficient_variation': stats['std'] / abs(stats['mean']) if stats['mean'] != 0 else 0,
        'range': stats['max'] - stats['min'],
        'iqr': float(np.percentile(inflation_data, 75) - np.percentile(inflation_data, 25))
    }
    
    # Trend analysis
    if len(inflation_data) >= 12:
        recent_12m = inflation_data.tail(12)
        previous_12m = inflation_data.tail(24).head(12) if len(inflation_data) >= 24 else inflation_data.head(12)
        
        trend = {
            'recent_12m_avg': float(np.mean(recent_12m)),
            'previous_12m_avg': float(np.mean(previous_12m)),
            'trend_direction': 'increasing' if np.mean(recent_12m) > np.mean(previous_12m) else 'decreasing',
            'trend_magnitude': float(abs(np.mean(recent_12m) - np.mean(previous_12m)))
        }
    else:
        trend = {'insufficient_data': True}
    
    return {
        'statistics': stats,
        'periods': periods,
        'volatility': volatility,
        'trend': trend
    }

def _analyze_predictions(self, predictions: pd.DataFrame) -> Dict[str, Any]:
    """Analyze prediction results."""
    pred_values = predictions['predicted_inflation'].values
    
    # Basic prediction statistics
    pred_stats = {
        'mean': float(np.mean(pred_values)),
        'median': float(np.median(pred_values)),
        'std': float(np.std(pred_values)),
        'min': float(np.min(pred_values)),
        'max': float(np.max(pred_values)),
        'count': len(pred_values)
    }
    
    # Prediction trajectory
    trajectory = {
        'initial_prediction': float(pred_values[0]) if len(pred_values) > 0 else 0,
        'final_prediction': float(pred_values[-1]) if len(pred_values) > 0 else 0,
        'overall_trend': 'increasing' if pred_values[-1] > pred_values[0] else 'decreasing' if len(pred_values) > 1 else 'stable',
        'volatility': float(np.std(np.diff(pred_values))) if len(pred_values) > 1 else 0
    }
    
    # Confidence analysis
    confidence_analysis = {}
    if 'confidence_lower' in predictions.columns and 'confidence_upper' in predictions.columns:
        ci_width = predictions['confidence_upper'] - predictions['confidence_lower']
        confidence_analysis = {
            'avg_confidence_width': float(np.mean(ci_width)),
            'max_confidence_width': float(np.max(ci_width)),
            'min_confidence_width': float(np.min(ci_width)),
            'confidence_trend': 'increasing' if ci_width.iloc[-1] > ci_width.iloc[0] else 'decreasing' if len(ci_width) > 1 else 'stable'
        }
    
    # Economic classification
    economic_periods = {
        'high_inflation_months': len(pred_values[pred_values > 3.0]),
        'moderate_inflation_months': len(pred_values[(pred_values >= 1.0) & (pred_values <= 3.0)]),
        'low_inflation_months': len(pred_values[(pred_values >= 0) & (pred_values < 1.0)]),
        'deflation_months': len(pred_values[pred_values < 0])
    }
    
    return {
        'statistics': pred_stats,
        'trajectory': trajectory,
        'confidence': confidence_analysis,
        'economic_periods': economic_periods
    }

def _generate_economic_interpretation(self, historical: Dict[str, Any], 
                                    predictions: Dict[str, Any],
                                    model_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate economic interpretation of results."""
    interpretation = {
        'inflation_regime': self._classify_inflation_regime(historical, predictions),
        'economic_outlook': self._assess_economic_outlook(predictions),
        'policy_implications': self._assess_policy_implications(historical, predictions),
        'risk_assessment': self._assess_risks(historical, predictions),
        'model_reliability': self._assess_model_reliability(model_results) if model_results else {}
    }
    
    return interpretation

def _classify_inflation_regime(self, historical: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, str]:
    """Classify the inflation regime based on historical and predicted data."""
    hist_mean = historical.get('statistics', {}).get('mean', 0)
    pred_mean = predictions.get('statistics', {}).get('mean', 0)
    
    # Historical regime
    if hist_mean < 0:
        hist_regime = "deflacionario"
    elif hist_mean < 1:
        hist_regime = "inflación muy baja"
    elif hist_mean <= 2:
        hist_regime = "inflación baja y estable"
    elif hist_mean <= 4:
        hist_regime = "inflación moderada"
    else:
        hist_regime = "inflación alta"
    
    # Predicted regime
    if pred_mean < 0:
        pred_regime = "deflacionario"
    elif pred_mean < 1:
        pred_regime = "inflación muy baja"
    elif pred_mean <= 2:
        pred_regime = "inflación baja y estable"
    elif pred_mean <= 4:
        pred_regime = "inflación moderada"
    else:
        pred_regime = "inflación alta"
    
    return {
        'historical_regime': hist_regime,
        'predicted_regime': pred_regime,
        'regime_change': 'Sí' if hist_regime != pred_regime else 'No'
    }

def _assess_economic_outlook(self, predictions: Dict[str, Any]) -> Dict[str, str]:
    """Assess economic outlook based on predictions."""
    trajectory = predictions.get('trajectory', {})
    stats = predictions.get('statistics', {})
    
    trend = trajectory.get('overall_trend', 'stable')
    volatility = trajectory.get('volatility', 0)
    mean_pred = stats.get('mean', 0)
    
    if trend == 'increasing' and mean_pred > 3:
        outlook = "Presiones inflacionarias crecientes"
        risk_level = "Alto"
    elif trend == 'decreasing' and mean_pred < 1:
        outlook = "Riesgo de deflación"
        risk_level = "Moderado-Alto"
    elif 1 <= mean_pred <= 2:
        outlook = "Estabilidad de precios"
        risk_level = "Bajo"
    else:
        outlook = "Inflación moderada"
        risk_level = "Moderado"
    
    volatility_assessment = "Alta" if volatility > 0.5 else "Moderada" if volatility > 0.2 else "Baja"
    
    return {
        'outlook': outlook,
        'risk_level': risk_level,
        'volatility_assessment': volatility_assessment,
        'stability': "Estable" if volatility < 0.3 else "Inestable"
    }

def _assess_policy_implications(self, historical: Dict[str, Any], predictions: Dict[str, Any]) -> List[str]:
    """Assess monetary policy implications."""
    implications = []
    
    pred_mean = predictions.get('statistics', {}).get('mean', 0)
    trend = predictions.get('trajectory', {}).get('overall_trend', 'stable')
    
    # ECB target is around 2%
    if pred_mean < 1:
        implications.append("Política monetaria expansiva podría ser necesaria para estimular la inflación")
    elif pred_mean > 3:
        implications.append("Política monetaria restrictiva podría ser necesaria para controlar la inflación")
    else:
        implications.append("Inflación cerca del objetivo del BCE (2%), política monetaria neutral apropiada")
    
    if trend == 'increasing':
        implications.append("Tendencia creciente sugiere vigilancia de presiones inflacionarias")
    elif trend == 'decreasing':
        implications.append("Tendencia decreciente requiere monitoreo de riesgos deflacionarios")
    
    volatility = predictions.get('trajectory', {}).get('volatility', 0)
    if volatility > 0.5:
        implications.append("Alta volatilidad sugiere incertidumbre económica elevada")
    
    return implications

def _assess_risks(self, historical: Dict[str, Any], predictions: Dict[str, Any]) -> Dict[str, List[str]]:
    """Assess economic risks based on analysis."""
    risks = {
        'upside_risks': [],
        'downside_risks': [],
        'structural_risks': []
    }
    
    pred_mean = predictions.get('statistics', {}).get('mean', 0)
    pred_max = predictions.get('statistics', {}).get('max', 0)
    pred_min = predictions.get('statistics', {}).get('min', 0)
    
    # Upside risks (higher inflation)
    if pred_max > 4:
        risks['upside_risks'].append("Riesgo de inflación alta en algunos períodos")
    if predictions.get('trajectory', {}).get('overall_trend') == 'increasing':
        risks['upside_risks'].append("Tendencia creciente podría acelerar más de lo previsto")
    
    # Downside risks (lower inflation/deflation)
    if pred_min < 0:
        risks['downside_risks'].append("Riesgo de períodos deflacionarios")
    if pred_mean < 1:
        risks['downside_risks'].append("Inflación persistentemente baja")
    
    # Structural risks
    volatility = predictions.get('trajectory', {}).get('volatility', 0)
    if volatility > 0.5:
        risks['structural_risks'].append("Alta volatilidad indica incertidumbre estructural")
    
    confidence = predictions.get('confidence', {})
    if confidence.get('avg_confidence_width', 0) > 2:
        risks['structural_risks'].append("Amplios intervalos de confianza indican alta incertidumbre")
    
    return risks

def _assess_model_reliability(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
    """Assess model reliability based on performance metrics."""
    reliability = {
        'overall_assessment': 'No disponible',
        'best_model': 'No identificado',
        'performance_summary': {}
    }
    
    if not model_results:
        return reliability
    
    # Find best performing model
    best_model = None
    best_mae = float('inf')
    
    for model_name, results in model_results.items():
        if results.get('status') == 'success':
            metrics = results.get('metrics', {})
            mae = metrics.get('MAE', float('inf'))
            if mae < best_mae:
                best_mae = mae
                best_model = model_name
    
    if best_model:
        reliability['best_model'] = best_model.replace('_', ' ').title()
        best_metrics = model_results[best_model].get('metrics', {})
        
        # Assess performance
        mae = best_metrics.get('MAE', 0)
        mape = best_metrics.get('MAPE', 0)
        
        if mae < 0.5 and mape < 10:
            assessment = "Excelente"
        elif mae < 1.0 and mape < 20:
            assessment = "Buena"
        elif mae < 2.0 and mape < 30:
            assessment = "Moderada"
        else:
            assessment = "Limitada"
        
        reliability['overall_assessment'] = assessment
        reliability['performance_summary'] = {
            'MAE': f"{mae:.3f}",
            'RMSE': f"{best_metrics.get('RMSE', 0):.3f}",
            'MAPE': f"{mape:.1f}%"
        }
    
    return reliability

def _generate_conclusions(self, historical: Dict[str, Any], 
                        predictions: Dict[str, Any],
                        interpretation: Dict[str, Any]) -> List[str]:
    """Generate conclusions about inflation trends."""
    conclusions = []
    
    # Historical conclusions
    hist_stats = historical.get('statistics', {})
    if hist_stats.get('mean', 0) > 0:
        conclusions.append(f"La inflación histórica promedio fue de {hist_stats['mean']:.2f}%, "
                            f"con una volatilidad de {hist_stats['std']:.2f}%")
    
    # Prediction conclusions
    pred_stats = predictions.get('statistics', {})
    trajectory = predictions.get('trajectory', {})
    
    conclusions.append(f"Las predicciones indican una inflación promedio de {pred_stats.get('mean', 0):.2f}% "
                        f"para los próximos 12 meses")
    
    if trajectory.get('overall_trend') == 'increasing':
        conclusions.append("Se espera una tendencia creciente en la inflación durante el período de predicción")
    elif trajectory.get('overall_trend') == 'decreasing':
        conclusions.append("Se espera una tendencia decreciente en la inflación durante el período de predicción")
    else:
        conclusions.append("Se espera una inflación relativamente estable durante el período de predicción")
    
    # Regime conclusions
    regime = interpretation.get('inflation_regime', {})
    if regime.get('regime_change') == 'Sí':
        conclusions.append(f"Se anticipa un cambio de régimen inflacionario: "
                            f"de {regime.get('historical_regime', 'N/A')} "
                            f"a {regime.get('predicted_regime', 'N/A')}")
    
    # Risk conclusions
    outlook = interpretation.get('economic_outlook', {})
    risk_level = outlook.get('risk_level', 'Desconocido')
    conclusions.append(f"El nivel de riesgo económico se evalúa como: {risk_level}")
    
    return conclusions

def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
    """Generate policy and economic recommendations."""
    recommendations = []
    
    interpretation = analysis.get('economic_interpretation', {})
    predictions = analysis.get('prediction_analysis', {})
    
    # Policy recommendations
    policy_implications = interpretation.get('policy_implications', [])
    if policy_implications:
        recommendations.extend(policy_implications)
    
    # Monitoring recommendations
    outlook = interpretation.get('economic_outlook', {})
    if outlook.get('volatility_assessment') == 'Alta':
        recommendations.append("Se recomienda monitoreo frecuente debido a la alta volatilidad esperada")
    
    confidence = predictions.get('confidence', {})
    if confidence.get('avg_confidence_width', 0) > 2:
        recommendations.append("Los amplios intervalos de confianza sugieren cautela en la interpretación")
    
    # Risk management recommendations
    risks = interpretation.get('risk_assessment', {})
    if risks.get('upside_risks'):
        recommendations.append("Preparar medidas preventivas contra presiones inflacionarias")
    if risks.get('downside_risks'):
        recommendations.append("Considerar estímulos económicos ante riesgos deflacionarios")
    
    return recommendations    

def create_technical_report(self, analysis: Dict[str, Any], 
                            visualizations: Dict[str, str],
                            model_results: Optional[Dict[str, Any]] = None,
                            output_filename: Optional[str] = None) -> str:
    """
    Create technical report for PDF generation using reportlab.
    
    Generates a comprehensive PDF report with methodology, results,
    and economic analysis.
    
    Args:
        analysis (Dict[str, Any]): Economic analysis results
        visualizations (Dict[str, str]): Dictionary of plot file paths
        model_results (Dict[str, Any], optional): Model evaluation results
        output_filename (str, optional): Output PDF filename
        
    Returns:
        str: Path to generated PDF report
    """
    if not _REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is required for PDF generation. "
                        "Install with: pip install reportlab")
    
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"informe_tecnico_inflacion_{timestamp}.pdf"
    
    output_path = self.reports_dir / output_filename
    
    self.logger.info(f"Creating technical PDF report: {output_path}")
    
    try:
        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=A4,
                                rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        # Build story (content)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1,  # Center
            textColor=colors.darkblue
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Title page
        story.append(Paragraph("INFORME TÉCNICO", title_style))
        story.append(Paragraph("Predicción de Inflación en España", title_style))
        story.append(Spacer(1, 20))
        
        # Date and summary
        story.append(Paragraph(f"Fecha: {datetime.now().strftime('%d/%m/%Y')}", styles['Normal']))
        story.append(Paragraph(f"Generado por: Sistema de Predicción de Inflación IA", styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Executive Summary
        story.append(Paragraph("RESUMEN EJECUTIVO", heading_style))
        
        pred_analysis = analysis.get('prediction_analysis', {})
        pred_stats = pred_analysis.get('statistics', {})
        
        summary_text = f"""
        Este informe presenta los resultados del análisis predictivo de la inflación en España 
        utilizando modelos de inteligencia artificial. Las predicciones indican una inflación 
        promedio de {pred_stats.get('mean', 0):.2f}% para los próximos 12 meses, con un rango 
        de {pred_stats.get('min', 0):.2f}% a {pred_stats.get('max', 0):.2f}%.
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Methodology section
        story.append(Paragraph("1. METODOLOGÍA", heading_style))
        
        methodology_text = """
        <b>Fuente de Datos:</b> Instituto Nacional de Estadística (INE) - Índice de Precios al Consumo (IPC)<br/>
        <b>Modelos Utilizados:</b> ARIMA, Random Forest, LSTM (Long Short-Term Memory)<br/>
        <b>Período de Análisis:</b> Datos históricos desde 2002<br/>
        <b>Horizonte de Predicción:</b> 12 meses<br/>
        <b>Métricas de Evaluación:</b> MAE (Error Absoluto Medio), RMSE (Raíz del Error Cuadrático Medio), MAPE (Error Porcentual Absoluto Medio)
        """
        story.append(Paragraph(methodology_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Model Performance section
        if model_results:
            story.append(Paragraph("2. RENDIMIENTO DE MODELOS", heading_style))
            
            # Create performance table
            model_data = [['Modelo', 'MAE', 'RMSE', 'MAPE (%)', 'Estado']]
            
            for model_name, results in model_results.items():
                if results.get('status') == 'success':
                    metrics = results.get('metrics', {})
                    model_data.append([
                        model_name.replace('_', ' ').title(),
                        f"{metrics.get('MAE', 0):.3f}",
                        f"{metrics.get('RMSE', 0):.3f}",
                        f"{metrics.get('MAPE', 0):.1f}",
                        "Exitoso"
                    ])
                else:
                    model_data.append([
                        model_name.replace('_', ' ').title(),
                        "N/A", "N/A", "N/A", "Error"
                    ])
            
            model_table = Table(model_data)
            model_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(model_table)
            story.append(Spacer(1, 20))
        
        # Economic Analysis section
        story.append(Paragraph("3. ANÁLISIS ECONÓMICO", heading_style))
        
        # Historical analysis
        hist_analysis = analysis.get('historical_analysis', {})
        hist_stats = hist_analysis.get('statistics', {})
        
        if hist_stats:
            hist_text = f"""
            <b>Análisis Histórico:</b><br/>
            • Inflación promedio histórica: {hist_stats.get('mean', 0):.2f}%<br/>
            • Desviación estándar: {hist_stats.get('std', 0):.2f}%<br/>
            • Rango: {hist_stats.get('min', 0):.2f}% - {hist_stats.get('max', 0):.2f}%<br/>
            • Número de observaciones: {hist_stats.get('count', 0)}
            """
            story.append(Paragraph(hist_text, styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Prediction analysis
        if pred_stats:
            pred_text = f"""
            <b>Análisis de Predicciones:</b><br/>
            • Inflación promedio predicha: {pred_stats.get('mean', 0):.2f}%<br/>
            • Desviación estándar: {pred_stats.get('std', 0):.2f}%<br/>
            • Rango predicho: {pred_stats.get('min', 0):.2f}% - {pred_stats.get('max', 0):.2f}%<br/>
            • Horizonte de predicción: {pred_stats.get('count', 0)} meses
            """
            story.append(Paragraph(pred_text, styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Economic interpretation
        interpretation = analysis.get('economic_interpretation', {})
        
        # Inflation regime
        regime = interpretation.get('inflation_regime', {})
        if regime:
            regime_text = f"""
            <b>Régimen Inflacionario:</b><br/>
            • Régimen histórico: {regime.get('historical_regime', 'N/A')}<br/>
            • Régimen predicho: {regime.get('predicted_regime', 'N/A')}<br/>
            • Cambio de régimen: {regime.get('regime_change', 'N/A')}
            """
            story.append(Paragraph(regime_text, styles['Normal']))
            story.append(Spacer(1, 10))
        
        # Economic outlook
        outlook = interpretation.get('economic_outlook', {})
        if outlook:
            outlook_text = f"""
            <b>Perspectiva Económica:</b><br/>
            • Evaluación: {outlook.get('outlook', 'N/A')}<br/>
            • Nivel de riesgo: {outlook.get('risk_level', 'N/A')}<br/>
            • Volatilidad: {outlook.get('volatility_assessment', 'N/A')}<br/>
            • Estabilidad: {outlook.get('stability', 'N/A')}
            """
            story.append(Paragraph(outlook_text, styles['Normal']))
            story.append(Spacer(1, 20))
        
        # Conclusions section
        story.append(Paragraph("4. CONCLUSIONES", heading_style))
        
        conclusions = analysis.get('conclusions', [])
        if conclusions:
            for i, conclusion in enumerate(conclusions, 1):
                story.append(Paragraph(f"{i}. {conclusion}", styles['Normal']))
                story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("No se generaron conclusiones específicas.", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Recommendations section
        story.append(Paragraph("5. RECOMENDACIONES", heading_style))
        
        recommendations = analysis.get('recommendations', [])
        if recommendations:
            for i, recommendation in enumerate(recommendations, 1):
                story.append(Paragraph(f"{i}. {recommendation}", styles['Normal']))
                story.append(Spacer(1, 8))
        else:
            story.append(Paragraph("No se generaron recomendaciones específicas.", styles['Normal']))
        
        # Risk Assessment section
        risks = interpretation.get('risk_assessment', {})
        if risks:
            story.append(Spacer(1, 20))
            story.append(Paragraph("6. EVALUACIÓN DE RIESGOS", heading_style))
            
            upside_risks = risks.get('upside_risks', [])
            if upside_risks:
                story.append(Paragraph("<b>Riesgos al Alza:</b>", styles['Normal']))
                for risk in upside_risks:
                    story.append(Paragraph(f"• {risk}", styles['Normal']))
                story.append(Spacer(1, 8))
            
            downside_risks = risks.get('downside_risks', [])
            if downside_risks:
                story.append(Paragraph("<b>Riesgos a la Baja:</b>", styles['Normal']))
                for risk in downside_risks:
                    story.append(Paragraph(f"• {risk}", styles['Normal']))
                story.append(Spacer(1, 8))
            
            structural_risks = risks.get('structural_risks', [])
            if structural_risks:
                story.append(Paragraph("<b>Riesgos Estructurales:</b>", styles['Normal']))
                for risk in structural_risks:
                    story.append(Paragraph(f"• {risk}", styles['Normal']))
        
        # Footer
        story.append(Spacer(1, 30))
        story.append(Paragraph("---", styles['Normal']))
        story.append(Paragraph(f"Informe generado automáticamente el {datetime.now().strftime('%d/%m/%Y a las %H:%M')}", 
                                styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        self.logger.info(f"Technical report created successfully: {output_path}")
        return str(output_path)
        
    except Exception as e:
        self.logger.error(f"Error creating technical report: {e}")
        raise

def export_code_screenshots(self, source_dir: str = "src", 
                            output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Export code screenshots for process documentation.
    
    Creates documentation of the source code by capturing
    code snippets and saving them as formatted text files.
    
    Args:
        source_dir (str): Directory containing source code
        output_dir (str, optional): Output directory for code documentation
        
    Returns:
        Dict[str, str]: Dictionary mapping file names to documentation paths
    """
    if output_dir is None:
        output_dir = self.reports_dir / "code_documentation"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    self.logger.info(f"Exporting code documentation from {source_dir} to {output_dir}")
    
    documented_files = {}
    source_path = Path(source_dir)
    
    if not source_path.exists():
        self.logger.warning(f"Source directory {source_dir} does not exist")
        return documented_files
    
    try:
        # Find all Python files
        python_files = list(source_path.glob("*.py"))
        
        for py_file in python_files:
            try:
                # Read source code
                with open(py_file, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                
                # Create formatted documentation
                doc_content = self._format_code_documentation(py_file.name, code_content)
                
                # Save documentation
                doc_filename = f"{py_file.stem}_documentation.txt"
                doc_path = output_dir / doc_filename
                
                with open(doc_path, 'w', encoding='utf-8') as f:
                    f.write(doc_content)
                
                documented_files[py_file.name] = str(doc_path)
                self.logger.info(f"Documented {py_file.name}")
                
            except Exception as e:
                self.logger.warning(f"Could not document {py_file}: {e}")
        
        # Create summary documentation
        summary_path = self._create_code_summary(documented_files, output_dir)
        if summary_path:
            documented_files['_summary'] = summary_path
        
        # Create process flow documentation
        process_path = self._create_process_documentation(output_dir)
        if process_path:
            documented_files['_process_flow'] = process_path
        
        self.logger.info(f"Code documentation completed. {len(documented_files)} files documented.")
        return documented_files
        
    except Exception as e:
        self.logger.error(f"Error exporting code documentation: {e}")
        raise

def _format_code_documentation(self, filename: str, code_content: str) -> str:
    """Format code content for documentation."""
    lines = code_content.split('\n')
    
    doc_lines = [
        "=" * 80,
        f"DOCUMENTACIÓN DE CÓDIGO: {filename}",
        "=" * 80,
        f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        f"Número de líneas: {len(lines)}",
        "",
        "CONTENIDO DEL ARCHIVO:",
        "-" * 40,
        ""
    ]
    
    # Add line numbers and content
    for i, line in enumerate(lines, 1):
        doc_lines.append(f"{i:4d}: {line}")
    
    doc_lines.extend([
        "",
        "-" * 40,
        "FIN DEL ARCHIVO",
        "=" * 80
    ])
    
    return '\n'.join(doc_lines)

def _create_code_summary(self, documented_files: Dict[str, str], output_dir: Path) -> Optional[str]:
    """Create a summary of all documented code files."""
    try:
        summary_content = [
            "=" * 80,
            "RESUMEN DE DOCUMENTACIÓN DE CÓDIGO",
            "=" * 80,
            f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            f"Total de archivos documentados: {len([f for f in documented_files.keys() if not f.startswith('_')])}",
            "",
            "ARCHIVOS DOCUMENTADOS:",
            "-" * 40
        ]
        
        for filename, doc_path in documented_files.items():
            if not filename.startswith('_'):
                summary_content.append(f"• {filename} -> {Path(doc_path).name}")
        
        summary_content.extend([
            "",
            "ESTRUCTURA DEL SISTEMA:",
            "-" * 40,
            "• ine_extractor.py: Descarga de datos del INE",
            "• data_cleaner.py: Limpieza y procesamiento de datos",
            "• feature_engineering.py: Ingeniería de características",
            "• model_trainer.py: Entrenamiento de modelos de IA",
            "• predictor.py: Generación de predicciones",
            "• report_generator.py: Generación de informes y visualizaciones",
            "",
            "FLUJO DE PROCESAMIENTO:",
            "-" * 40,
            "1. Extracción de datos (INE) -> datos brutos",
            "2. Limpieza de datos -> datos procesados",
            "3. Ingeniería de características -> características para ML",
            "4. Entrenamiento de modelos -> modelos entrenados",
            "5. Generación de predicciones -> predicciones futuras",
            "6. Generación de informes -> análisis y visualizaciones",
            "",
            "=" * 80
        ])
        
        summary_path = output_dir / "resumen_documentacion.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_content))
        
        return str(summary_path)
        
    except Exception as e:
        self.logger.warning(f"Could not create code summary: {e}")
        return None

def _create_process_documentation(self, output_dir: Path) -> Optional[str]:
    """Create process flow documentation."""
    try:
        process_content = [
            "=" * 80,
            "DOCUMENTACIÓN DEL PROCESO DE PREDICCIÓN DE INFLACIÓN",
            "=" * 80,
            f"Fecha de generación: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
            "",
            "DESCRIPCIÓN GENERAL:",
            "-" * 40,
            "Sistema automatizado para la predicción de la tasa de inflación en España",
            "utilizando datos del Instituto Nacional de Estadística (INE) y modelos de",
            "inteligencia artificial (ARIMA, Random Forest, LSTM).",
            "",
            "COMPONENTES PRINCIPALES:",
            "-" * 40,
            "",
            "1. EXTRACTOR DE DATOS INE (ine_extractor.py)",
            "   • Conecta con la API del INE",
            "   • Descarga datos de IPC general, por grupos e IPCA",
            "   • Maneja errores de conexión con reintentos",
            "   • Guarda datos en formato CSV",
            "",
            "2. PROCESADOR DE DATOS (data_cleaner.py)",
            "   • Carga datos brutos desde CSV",
            "   • Detecta y corrige valores faltantes",
            "   • Identifica y trata outliers estadísticos",
            "   • Normaliza fechas y calcula tasas de inflación",
            "",
            "3. INGENIERO DE CARACTERÍSTICAS (feature_engineering.py)",
            "   • Crea características lag (1, 3, 6, 12 meses)",
            "   • Genera medias móviles y componentes estacionales",
            "   • Prepara datos para modelos de machine learning",
            "",
            "4. ENTRENADOR DE MODELOS (model_trainer.py)",
            "   • Implementa modelos ARIMA, Random Forest y LSTM",
            "   • Divide datos en entrenamiento y validación",
            "   • Evalúa modelos con métricas MAE, RMSE, MAPE",
            "   • Selecciona el mejor modelo automáticamente",
            "",
            "5. PREDICTOR (predictor.py)",
            "   • Carga el mejor modelo entrenado",
            "   • Genera predicciones para 12 meses",
            "   • Calcula intervalos de confianza",
            "   • Valida y exporta predicciones",
            "",
            "6. GENERADOR DE INFORMES (report_generator.py)",
            "   • Crea visualizaciones con matplotlib/seaborn",
            "   • Genera análisis económico interpretativo",
            "   • Produce informes técnicos en PDF",
            "   • Documenta código fuente del proceso",
            "",
            "FLUJO DE EJECUCIÓN:",
            "-" * 40,
            "",
            "Paso 1: Descarga de Datos",
            "├── Conectar con API del INE",
            "├── Descargar IPC general (serie principal)",
            "├── Descargar IPC por grupos (sectorial)",
            "├── Descargar IPCA (armonizado europeo)",
            "└── Guardar en data/raw/",
            "",
            "Paso 2: Procesamiento de Datos",
            "├── Cargar datos brutos",
            "├── Limpiar valores faltantes (interpolación)",
            "├── Detectar outliers (IQR, Z-score)",
            "├── Calcular tasas de inflación",
            "└── Guardar en data/processed/",
            "",
            "Paso 3: Ingeniería de Características",
            "├── Crear características lag",
            "├── Generar medias móviles",
            "├── Añadir componentes estacionales",
            "└── Preparar datasets para ML",
            "",
            "Paso 4: Entrenamiento de Modelos",
            "├── Dividir datos (80% entrenamiento, 20% validación)",
            "├── Entrenar ARIMA (series temporales)",
            "├── Entrenar Random Forest (no lineal)",
            "├── Entrenar LSTM (redes neuronales)",
            "├── Evaluar con métricas de error",
            "└── Seleccionar mejor modelo",
            "",
            "Paso 5: Generación de Predicciones",
            "├── Cargar mejor modelo",
            "├── Generar predicciones 12 meses",
            "├── Calcular intervalos de confianza",
            "└── Exportar resultados",
            "",
            "Paso 6: Generación de Informes",
            "├── Crear visualizaciones (6 tipos de gráficos)",
            "├── Generar análisis económico interpretativo",
            "├── Producir informe técnico PDF",
            "└── Documentar proceso y código",
            "",
            "MÉTRICAS DE EVALUACIÓN:",
            "-" * 40,
            "• MAE (Mean Absolute Error): Error absoluto promedio",
            "• RMSE (Root Mean Square Error): Raíz del error cuadrático medio",
            "• MAPE (Mean Absolute Percentage Error): Error porcentual absoluto medio",
            "",
            "OUTPUTS GENERADOS:",
            "-" * 40,
            "• data/processed/: Datos limpios y procesados",
            "• models/: Modelos entrenados guardados",
            "• reports/: Visualizaciones, análisis y documentación",
            "• reports/informe_tecnico_*.pdf: Informe técnico completo",
            "",
            "CONFIGURACIÓN:",
            "-" * 40,
            "• config/config.yaml: Parámetros del sistema",
            "• requirements.txt: Dependencias de Python",
            "",
            "=" * 80
        ]
        
        process_path = output_dir / "documentacion_proceso.txt"
        with open(process_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(process_content))
        
        return str(process_path)
        
    except Exception as e:
        self.logger.warning(f"Could not create process documentation: {e}")
        return None