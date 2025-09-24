import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

from config.settings import user_preferences, COLORS, is_arabic, is_dark_mode, arabic_font
from src.analyzer import AdvancedDataAnalyzer
from src.database import AdvancedDataAnalysisDB
from src.features import DataAnalysisFeatures
from ui.interface import create_advanced_interface
from utils.file_utils import load_file  # إضافة imports أخرى إذا لزم

# globals الرئيسية (يمكن تحويلها إلى singleton لاحقاً)
data_frame = None
current_session_id = None
root = None
notebook = None
status_var = None
preview_text = None
result_display = None
table_frame = None

def main():
    global root
    analyzer = AdvancedDataAnalyzer()
    advanced_db = AdvancedDataAnalysisDB()
    features = DataAnalysisFeatures()
    
    root = create_advanced_interface()
    
    # تحميل الإعدادات
    try:
        if os.path.exists("data_analysis_settings.json"):
            with open("data_analysis_settings.json", "r", encoding='utf-8') as f:
                loaded_settings = json.load(f)
                user_preferences.update(loaded_settings)
    except:
        pass
    
    root.mainloop()

if __name__ == "__main__":
    main()