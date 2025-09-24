import platform
import matplotlib.font_manager as fm

# متغيرات الحالة
is_arabic = True
is_dark_mode = True
current_plot = None
user_notes = {}
current_analysis_history = []
ml_models = {}
data_snapshots = {}
analysis_results = {}

user_preferences = {
    "language": "arabic",
    "theme": "dark",
    "font_size": 12,
    "auto_save": False,
    "default_chart_style": "seaborn",
    "animation_effects": True,
    "tooltips": True,
    "auto_update": True,
    "backup_interval": 10,
    "export_quality": "high",
    "recent_files": [],
    "max_recent_files": 10,
    "default_analysis": "basic_stats",
    "data_preview_rows": 15,
    "chart_colors": "viridis",
    "number_format": "comma",
    "date_format": "YYYY-MM-DD",
    "timezone": "Asia/Riyadh"
}

COLORS = {
    "dark": {
        "bg": "#1e1e2f", "frame_bg": "#2a2a40", "text": "white",
        "button_bg": "#3b8ad8", "accent": "#6a6a9c", "success": "#4caf50",
        "warning": "#ff9800", "error": "#f44336", "highlight": "#ffd700",
        "header": "#343456", "border": "#4a4a6a", "card": "#2d2d44"
    },
    "light": {
        "bg": "#f5f5f5", "frame_bg": "#ffffff", "text": "#333333",
        "button_bg": "#2196F3", "accent": "#e0e0e0", "success": "#4CAF50",
        "warning": "#FFC107", "error": "#F44336", "highlight": "#FFEB3B",
        "header": "#e8e8e8", "border": "#d0d0d0", "card": "#f8f9fa"
    }
}

try:
    arabic_font = fm.FontProperties(fname='assets/arial.ttf')
except:
    arabic_font = None