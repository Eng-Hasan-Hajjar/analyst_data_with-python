import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
import io
import sys
import os
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.font_manager as fm
from datetime import datetime, timedelta
import json
import csv
import sqlite3
import threading
import time
import webbrowser
from PIL import Image, ImageTk
import requests
import hashlib
import zipfile
import tempfile
import platform
import calendar
from dateutil import parser
import re

warnings.filterwarnings('ignore')

# ------------------- إعدادات متقدمة -------------------
class AdvancedDataAnalyzer:
    def __init__(self):
        self.version = "3.0.0"
        self.author = "نظام تحليل البيانات المتقدم"
        self.support_email = "support@dataanalysis.com"
        
    def get_system_info(self):
        return {
            "platform": platform.system(),
            "version": platform.version(),
            "python_version": platform.python_version(),
            "processor": platform.processor()
        }

analyzer = AdvancedDataAnalyzer()

# ------------------- إعداد الخطوط العربية -------------------
try:
    arabic_font = fm.FontProperties(fname='arial.ttf')
except:
    arabic_font = None

# ------------------- الحالة والمتغيرات العالمية -------------------
is_arabic = True
is_dark_mode = True
data_frame = None
current_plot = None
user_notes = {}
current_analysis_history = []
ml_models = {}
data_snapshots = {}
analysis_results = {}
current_session_id = None

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

# ------------------- الألوان حسب السمة -------------------
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

# ------------------- قاعدة البيانات المتقدمة -------------------
class AdvancedDataAnalysisDB:
    def __init__(self):
        self.conn = sqlite3.connect('advanced_data_analysis.db', check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        
        tables = [
            '''CREATE TABLE IF NOT EXISTS analysis_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_name TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                data_shape TEXT, analysis_type TEXT, parameters TEXT)''',
            
            '''CREATE TABLE IF NOT EXISTS saved_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT, model_name TEXT,
                model_type TEXT, accuracy REAL, parameters TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''',
            
            '''CREATE TABLE IF NOT EXISTS data_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
                snapshot_name TEXT, data_hash TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''',
            
            '''CREATE TABLE IF NOT EXISTS user_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
                note_text TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''',
            
            '''CREATE TABLE IF NOT EXISTS analysis_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER,
                action_type TEXT, action_details TEXT, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)'''
        ]
        
        for table in tables:
            cursor.execute(table)
        self.conn.commit()
    
    def save_analysis_session(self, session_name, data_shape, analysis_type, parameters):
        cursor = self.conn.cursor()
        cursor.execute('''INSERT INTO analysis_sessions (session_name, data_shape, analysis_type, parameters)
                         VALUES (?, ?, ?, ?)''', 
                      (session_name, str(data_shape), analysis_type, json.dumps(parameters)))
        self.conn.commit()
        return cursor.lastrowid

advanced_db = AdvancedDataAnalysisDB()

# ------------------- الميزات الجديدة المتقدمة -------------------
class DataAnalysisFeatures:
    @staticmethod
    def detect_data_quality_issues(df):
        """كشف مشاكل جودة البيانات"""
        issues = []
        
        if df is None:
            return ["لا توجد بيانات للتحليل"]
        
        # القيم المفقودة
        missing = df.isnull().sum()
        if missing.sum() > 0:
            issues.append(f"القيم المفقودة: {missing.sum()} قيمة")
        
        # القيم المكررة
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            issues.append(f"الصفوف المكررة: {duplicates} صف")
        
        # القيم المتطرفة
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > 0:
                issues.append(f"قيم متطرفة في {col}: {outliers} قيمة")
        
        return issues

    @staticmethod
    def automated_data_cleaning(df):
        """تنظيف البيانات الآلي"""
        if df is None:
            return None
            
        df_clean = df.copy()
        
        # إزالة الصفوف المكررة
        df_clean = df_clean.drop_duplicates()
        
        # معالجة القيم المفقودة
        for col in df_clean.columns:
            if df_clean[col].dtype in ['int64', 'float64']:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown', inplace=True)
        
        return df_clean

    @staticmethod
    def create_advanced_chart(df, chart_type, x_col, y_col=None):
        """إنشاء رسوم بيانية متقدمة"""
        if df is None or df.empty:
            return None
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            if chart_type == "heatmap":
                corr = df.select_dtypes(include=[np.number]).corr()
                sns.heatmap(corr, annot=True, ax=ax)
            elif chart_type == "pairplot":
                sns.pairplot(df.select_dtypes(include=[np.number]))
            elif chart_type == "violin" and y_col:
                sns.violinplot(data=df, x=x_col, y=y_col, ax=ax)
            elif chart_type == "swarm" and y_col:
                sns.swarmplot(data=df, x=x_col, y=y_col, ax=ax)
            else:
                # رسم بياني افتراضي
                df.select_dtypes(include=[np.number]).iloc[:,0].plot(kind='hist', ax=ax)
        except Exception as e:
            print(f"Error creating chart: {e}")
            return None
        
        return fig

    @staticmethod
    def time_series_analysis(df, date_col, value_col):
        """تحليل السلاسل الزمنية"""
        if df is None or date_col not in df.columns or value_col not in df.columns:
            return {}
            
        try:
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
            
            analysis = {
                'monthly_mean': df[value_col].resample('M').mean(),
                'trend': df[value_col].rolling(window=30).mean(),
                'seasonality': df[value_col].groupby(df.index.month).mean()
            }
            
            return analysis
        except:
            return {}

    @staticmethod
    def feature_engineering(df):
        """هندسة المميزات المتقدمة"""
        if df is None:
            return None
            
        df_engineered = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                # إنشاء مميزات متقدمة
                df_engineered[f'{col}_squared'] = df[col] ** 2
                df_engineered[f'{col}_log'] = np.log1p(np.abs(df[col]) + 1)  # تجنب الأصفار والسالبة
                df_engineered[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
            except:
                continue
        
        return df_engineered

features = DataAnalysisFeatures()

# ------------------- الوظائف الأساسية المحسنة -------------------
def load_file():
    global data_frame, current_session_id
    file = filedialog.askopenfilename(
        title="اختر ملف البيانات" if is_arabic else "Select Data File",
        filetypes=[
            ("Excel Files", "*.xlsx *.xls"), 
            ("CSV Files", "*.csv"),
            ("JSON Files", "*.json"),
            ("Text Files", "*.txt"),
            ("All Files", ".")
        ]
    )
    
    if file:
        try:
            update_status("جاري تحميل الملف..." if is_arabic else "Loading file...")
            
            file_size = os.path.getsize(file) / (1024 * 1024)
            if file_size > 50:
                if not messagebox.askyesno("تحذير", f"الملف كبير ({file_size:.1f} MB). المتابعة؟"):
                    return
            
            # تحميل الملف مع معالجة الأخطاء
            if file.endswith('.csv'):
                data_frame = pd.read_csv(file, encoding='utf-8')
            elif file.endswith('.json'):
                data_frame = pd.read_json(file)
            elif file.endswith(('.xlsx', '.xls')):
                data_frame = pd.read_excel(file)
            else:
                data_frame = pd.read_csv(file, delimiter=None, engine='python')
            
            # تحديث الواجهة
            update_recent_files_menu(file)
            update_data_preview()
            show_dataframe_in_table()
            update_all_selectors()
            
            # حفظ الجلسة
            current_session_id = advanced_db.save_analysis_session(
                os.path.basename(file),
                data_frame.shape,
                "data_loading",
                {"file_type": file.split('.')[-1], "rows": data_frame.shape[0], "columns": data_frame.shape[1]}
            )
            
            # تحليل جودة البيانات
            quality_issues = features.detect_data_quality_issues(data_frame)
            if quality_issues:
                show_quality_report(quality_issues)
            
            update_status("تم تحميل البيانات بنجاح!" if is_arabic else "Data loaded successfully!")
            
        except Exception as e:
            update_status("فشل تحميل الملف" if is_arabic else "Failed to load file")
            messagebox.showerror("خطأ", f"فشل تحميل الملف: {str(e)}")

def update_recent_files_menu(file=None):
    if hasattr(root, 'recent_menu'):
        root.recent_menu.delete(0, tk.END)
        for file_path in user_preferences["recent_files"]:
            root.recent_menu.add_command(
                label=os.path.basename(file_path),
                command=lambda f=file_path: load_recent_file(f)
            )
        if file and file not in user_preferences["recent_files"]:
            user_preferences["recent_files"].insert(0, file)
            if len(user_preferences["recent_files"]) > user_preferences["max_recent_files"]:
                user_preferences["recent_files"].pop()

def load_recent_file(file):
    global data_frame
    try:
        if file.endswith('.csv'):
            data_frame = pd.read_csv(file, encoding='utf-8')
        elif file.endswith('.json'):
            data_frame = pd.read_json(file)
        elif file.endswith(('.xlsx', '.xls')):
            data_frame = pd.read_excel(file)
        else:
            data_frame = pd.read_csv(file, delimiter=None, engine='python')
        
        update_data_preview()
        show_dataframe_in_table()
        update_all_selectors()
        update_status("تم تحميل الملف الحديث!")
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل تحميل الملف: {str(e)}")

def show_dataframe_in_table():
    if data_frame is None or 'table_frame' not in globals():
        return
    
    for widget in table_frame.winfo_children():
        widget.destroy()
    
    # إنشاء Treeview مع شريط تمرير
    tree_frame = tk.Frame(table_frame)
    tree_frame.pack(fill="both", expand=True)
    
    tree_scroll = ttk.Scrollbar(tree_frame)
    tree_scroll.pack(side="right", fill="y")
    
    global data_table
    data_table = ttk.Treeview(tree_frame, yscrollcommand=tree_scroll.set, show="headings")
    data_table.pack(fill="both", expand=True)
    
    tree_scroll.config(command=data_table.yview)
    
    # إعداد الأعمدة
    data_table["columns"] = list(data_frame.columns)
    for col in data_frame.columns:
        data_table.heading(col, text=col)
        data_table.column(col, width=120, anchor="center")
    
    # إضافة البيانات
    for index, row in data_frame.head(1000).iterrows():
        data_table.insert("", "end", values=list(row))

def update_all_selectors():
    if data_frame is not None:
        columns = list(data_frame.columns)
        selectors = [
            'x_axis_selector', 'y_axis_selector', 'cluster_column_selector',
            'regression_x_selector', 'regression_y_selector',
            'classification_x_selector', 'classification_y_selector',
            'date_column_selector', 'value_column_selector'
        ]
        
        for selector_name in selectors:
            if selector_name in globals():
                try:
                    globals()[selector_name]['values'] = columns
                except:
                    pass

# ------------------- الميزات الجديدة الكاملة -------------------
def show_quality_report(issues):
    """عرض تقرير جودة البيانات"""
    report_window = tk.Toplevel(root)
    report_window.title("تقرير جودة البيانات")
    report_window.geometry("600x400")
    
    report_text = scrolledtext.ScrolledText(report_window, font=("Arial", 12))
    report_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    report_content = "📊 تقرير جودة البيانات\n" + "="*50 + "\n\n"
    for issue in issues:
        report_content += f"⚠️ {issue}\n"
    
    report_content += f"\n✅ إجمالي المشاكل: {len(issues)}"
    report_text.insert("1.0", report_content)
    report_text.config(state="disabled")

def automated_data_cleaning():
    """تنظيف البيانات الآلي"""
    global data_frame
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتنظيف!")
        return
    
    try:
        data_frame = features.automated_data_cleaning(data_frame)
        update_data_preview()
        show_dataframe_in_table()
        update_status("تم تنظيف البيانات بنجاح!")
        messagebox.showinfo("نجاح", "تم تنظيف البيانات بنجاح!")
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل تنظيف البيانات: {str(e)}")

def advanced_statistical_analysis():
    """تحليل إحصائي متقدم"""
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتحليل!")
        return
    
    try:
        numeric_cols = data_frame.select_dtypes(include=[np.number]).columns
        results = {}
        
        for col in numeric_cols:
            results[col] = {
                'mean': data_frame[col].mean(),
                'median': data_frame[col].median(),
                'std': data_frame[col].std(),
                'skewness': data_frame[col].skew(),
                'kurtosis': data_frame[col].kurtosis(),
                'variance': data_frame[col].var()
            }
        
        # عرض النتائج في نافذة جديدة
        show_statistical_results(results)
        
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل التحليل الإحصائي: {str(e)}")

def show_statistical_results(results):
    """عرض النتائج الإحصائية"""
    results_window = tk.Toplevel(root)
    results_window.title("النتائج الإحصائية المتقدمة")
    results_window.geometry("800x600")
    
    notebook = ttk.Notebook(results_window)
    notebook.pack(fill="both", expand=True, padx=10, pady=10)
    
    # تبويب النتائج
    results_frame = ttk.Frame(notebook)
    notebook.add(results_frame, text="النتائج")
    
    text_widget = scrolledtext.ScrolledText(results_frame, font=("Courier", 10))
    text_widget.pack(fill="both", expand=True)
    
    results_text = "النتائج الإحصائية المتقدمة\n" + "="*50 + "\n\n"
    for col, stats in results.items():
        results_text += f"📊 العمود: {col}\n"
        for stat_name, value in stats.items():
            results_text += f"   {stat_name}: {value:.4f}\n"
        results_text += "\n"
    
    text_widget.insert("1.0", results_text)
    text_widget.config(state="disabled")

def correlation_analysis():
    """تحليل الارتباط المتقدم"""
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتحليل!")
        return
    
    try:
        numeric_df = data_frame.select_dtypes(include=[np.number])
        if numeric_df.empty:
            messagebox.showwarning("تحذير", "لا توجد أعمدة رقمية للتحليل!")
            return
        
        # حساب مصفوفة الارتباط
        correlation_matrix = numeric_df.corr()
        
        # إنشاء رسم بياني
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # خريطة حرارية
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax1)
        ax1.set_title('مصفوفة الارتباط')
        
        # رسم الارتباطات
        if len(numeric_df.columns) >= 2:
            sns.scatterplot(data=numeric_df, x=numeric_df.columns[0], y=numeric_df.columns[1], ax=ax2)
            ax2.set_title('مخطط الانتشار')
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل تحليل الارتباط: {str(e)}")

def time_series_analysis():
    """تحليل السلاسل الزمنية"""
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتحليل!")
        return
    
    try:
        # نافذة اختيار الأعمدة
        ts_window = tk.Toplevel(root)
        ts_window.title("تحليل السلاسل الزمنية")
        ts_window.geometry("400x200")
        
        tk.Label(ts_window, text="اختر عمود التاريخ:").pack(pady=5)
        date_selector = ttk.Combobox(ts_window)
        date_selector.pack(pady=5)
        
        tk.Label(ts_window, text="اختر عمود القيم:").pack(pady=5)
        value_selector = ttk.Combobox(ts_window)
        value_selector.pack(pady=5)
        
        date_selector['values'] = list(data_frame.columns)
        value_selector['values'] = list(data_frame.select_dtypes(include=[np.number]).columns)
        
        def run_analysis():
            date_col = date_selector.get()
            value_col = value_selector.get()
            
            if not date_col or not value_col:
                messagebox.showwarning("تحذير", "يجب اختيار العمودين!")
                return
            
            analysis = features.time_series_analysis(data_frame, date_col, value_col)
            show_time_series_results(analysis, value_col)
            ts_window.destroy()
        
        tk.Button(ts_window, text="تشغيل التحليل", command=run_analysis).pack(pady=10)
        
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل التحليل الزمني: {str(e)}")

def show_time_series_results(analysis, value_col):
    """عرض نتائج التحليل الزمني"""
    results_window = tk.Toplevel(root)
    results_window.title("نتائج التحليل الزمني")
    results_window.geometry("600x400")
    
    text_widget = scrolledtext.ScrolledText(results_window, font=("Arial", 11))
    text_widget.pack(fill="both", expand=True, padx=10, pady=10)
    
    results_text = f"نتائج التحليل الزمني للعمود: {value_col}\n" + "="*50 + "\n\n"
    
    for key, value in analysis.items():
        results_text += f"{key}:\n{value}\n\n"
    
    text_widget.insert("1.0", results_text)
    text_widget.config(state="disabled")

def machine_learning_predictions():
    """التنبؤ باستخدام التعلم الآلي"""
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتحليل!")
        return
    
    try:
        ml_window = tk.Toplevel(root)
        ml_window.title("التنبؤ بالتعلم الآلي")
        ml_window.geometry("500x300")
        
        # إعداد واجهة المستخدم للتعلم الآلي
        tk.Label(ml_window, text="اختر خوارزمية التعلم الآلي:", font=("Arial", 12)).pack(pady=10)
        
        ml_algorithms = [
            ("انحدار خطي (Linear Regression)", "linear_regression"),
            ("غابة عشوائية (Random Forest)", "random_forest"),
            ("التجميع (K-Means)", "kmeans"),
            ("الانحدار اللوجستي (Logistic Regression)", "logistic_regression")
        ]
        
        algorithm_var = tk.StringVar(value="linear_regression")
        
        for text, value in ml_algorithms:
            ttk.Radiobutton(ml_window, text=text, variable=algorithm_var, value=value).pack(anchor="w", padx=20)
        
        def run_ml_analysis():
            algorithm = algorithm_var.get()
            result = run_machine_learning(algorithm)
            show_ml_results(result, algorithm)
            ml_window.destroy()
        
        tk.Button(ml_window, text="تشغيل النموذج", command=run_ml_analysis).pack(pady=20)
        
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل التحليل بالتعلم الآلي: {str(e)}")

def run_machine_learning(algorithm):
    """تشغيل خوارزميات التعلم الآلي"""
    if data_frame is None:
        return "لا توجد بيانات للتحليل!"
        
    numeric_df = data_frame.select_dtypes(include=[np.number])
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return "لا توجد بيانات كافية للتحليل!"
    
    try:
        X = numeric_df.iloc[:, :-1]
        y = numeric_df.iloc[:, -1]
        
        if algorithm == "linear_regression":
            model = LinearRegression()
            model.fit(X, y)
            predictions = model.predict(X)
            r2 = r2_score(y, predictions)
            return f"نموذج الانحدار الخطي - R²: {r2:.4f}"
        
        elif algorithm == "random_forest":
            model = RandomForestRegressor()
            model.fit(X, y)
            predictions = model.predict(X)
            r2 = r2_score(y, predictions)
            return f"النموذج الغابة العشوائية - R²: {r2:.4f}"
        
        elif algorithm == "kmeans":
            model = KMeans(n_clusters=3)
            clusters = model.fit_predict(X)
            return f"نموذج التجميع - عدد المجموعات: 3"
        
        elif algorithm == "logistic_regression":
            model = LogisticRegression()
            model.fit(X, y)
            accuracy = model.score(X, y)
            return f"الانحدار اللوجستي - الدقة: {accuracy:.4f}"
        else:
            return "الخوارزمية غير معروفة"
    
    except Exception as e:
        return f"خطأ في النموذج: {str(e)}"

def show_ml_results(result, algorithm):
    """عرض نتائج التعلم الآلي"""
    results_window = tk.Toplevel(root)
    results_window.title("نتائج التعلم الآلي")
    results_window.geometry("500x200")
    
    tk.Label(results_window, text=f"نتائج {algorithm}", font=("Arial", 14, "bold")).pack(pady=10)
    tk.Label(results_window, text=result, font=("Arial", 12)).pack(pady=10)
    tk.Button(results_window, text="حفظ النموذج", command=lambda: save_ml_model(algorithm, result)).pack(pady=10)

def save_ml_model(algorithm, result):
    """حفظ نموذج التعلم الآلي"""
    try:
        model_data = {
            'algorithm': algorithm,
            'result': result,
            'timestamp': datetime.now().isoformat(),
            'data_shape': data_frame.shape if data_frame is not None else 'No data'
        }
        
        with open(f"model_{algorithm}{datetime.now().strftime('%Y%m%d%H%M%S')}.json", 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        messagebox.showinfo("نجاح", "تم حفظ النموذج بنجاح!")
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل حفظ النموذج: {str(e)}")

def export_dashboard():
    """تصدير لوحة التحكم"""
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتصدير!")
        return
    
    try:
        file_path = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", ".html"), ("All files", ".*")]
        )
        
        if file_path:
            # إنشاء تقرير HTML بسيط
            report = create_html_report()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            messagebox.showinfo("نجاح", f"تم تصدير التقرير إلى: {file_path}")
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل التصدير: {str(e)}")

def create_html_report():
    """إنشاء تقرير HTML"""
    if data_frame is None:
        return "<html><body>لا توجد بيانات</body></html>"
    
    html_content = f"""
    <html>
    <head>
        <title>تقرير تحليل البيانات</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background: #2c3e50; color: white; padding: 20px; text-align: center; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            table {{ width: 100%; border-collapse: collapse; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>تقرير تحليل البيانات</h1>
            <p>تم إنشاؤه في: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>معلومات البيانات</h2>
            <p>عدد الصفوف: {data_frame.shape[0]}</p>
            <p>عدد الأعمدة: {data_frame.shape[1]}</p>
        </div>
        
        <div class="section">
            <h2>عينة من البيانات</h2>
            {data_frame.head().to_html() if hasattr(data_frame, 'to_html') else 'لا توجد بيانات'}
        </div>
    </body>
    </html>
    """
    
    return html_content

# ------------------- إصلاح الدوال التي تحتوي على أخطاء -------------------
def advanced_feature_engineering():
    """هندسة المميزات المتقدمة - الإصدار المصحح"""
    global data_frame
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتحويل!")
        return
    
    try:
        data_frame = features.feature_engineering(data_frame)
        update_data_preview()
        show_dataframe_in_table()
        messagebox.showinfo("نجاح", "تمت هندسة المميزات بنجاح!")
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل هندسة المميزات: {str(e)}")

def normalize_data():
    """تطبيع البيانات - الإصدار المصحح"""
    global data_frame
    if data_frame is not None:
        numeric_df = data_frame.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            scaler = MinMaxScaler()
            data_frame[numeric_df.columns] = scaler.fit_transform(numeric_df)
            update_data_preview()
            show_dataframe_in_table()
            messagebox.showinfo("نجاح", "تم تطبيع البيانات بنجاح!")

def standardize_data():
    """توحيد البيانات - الإصدار المصحح"""
    global data_frame
    if data_frame is not None:
        numeric_df = data_frame.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            scaler = StandardScaler()
            data_frame[numeric_df.columns] = scaler.fit_transform(numeric_df)
            update_data_preview()
            show_dataframe_in_table()
            messagebox.showinfo("نجاح", "تم توحيد البيانات بنجاح!")

def log_transform():
    """تحويل لوغاريتمي - الإصدار المصحح"""
    global data_frame
    if data_frame is not None:
        numeric_df = data_frame.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            # تجنب الأعداد السالبة والصفر
            numeric_df = numeric_df.apply(lambda x: x + abs(x.min()) + 1 if x.min() <= 0 else x)
            data_frame[numeric_df.columns] = np.log1p(numeric_df)
            update_data_preview()
            show_dataframe_in_table()
            messagebox.showinfo("نجاح", "تم التحويل اللوغاريتمي بنجاح!")

def pattern_detection():
    """اكتشاف الأنماط في البيانات - الإصدار المصحح"""
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتحليل!")
        return
    
    try:
        # استخدام خوارزميات لاكتشاف الأنماط
        numeric_df = data_frame.select_dtypes(include=[np.number])
        
        if not numeric_df.empty:
            # تحليل الاتجاهات
            trends = {}
            for col in numeric_df.columns:
                if len(numeric_df[col]) > 1:
                    try:
                        correlation = numeric_df[col].corr(pd.Series(range(len(numeric_df[col]))))
                        trends[col] = "تصاعدي" if correlation > 0.5 else "تنازلي" if correlation < -0.5 else "ثابت"
                    except:
                        trends[col] = "غير محدد"
            
            # عرض النتائج
            pattern_window = tk.Toplevel(root)
            pattern_window.title("اكتشاف الأنماط")
            pattern_window.geometry("400x300")
            
            text_widget = scrolledtext.ScrolledText(pattern_window)
            text_widget.pack(fill="both", expand=True, padx=10, pady=10)
            
            results_text = "📊 نتائج اكتشاف الأنماط\n" + "="*40 + "\n\n"
            for col, trend in trends.items():
                results_text += f"📈 {col}: اتجاه {trend}\n"
            
            text_widget.insert("1.0", results_text)
            text_widget.config(state="disabled")
        else:
            messagebox.showwarning("تحذير", "لا توجد أعمدة رقمية للتحليل!")
    
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل اكتشاف الأنماط: {str(e)}")

# ------------------- واجهة المستخدم المحسنة -------------------
def create_advanced_menu():
    """إنشاء قائمة متقدمة للواجهة"""
    menubar = tk.Menu(root)
    
    # قائمة ملف
    file_menu = tk.Menu(menubar, tearoff=0)
    file_menu.add_command(label="تحميل بيانات", command=load_file)
    file_menu.add_command(label="تصدير لوحة التحكم", command=export_dashboard)
    file_menu.add_separator()
    file_menu.add_command(label="خروج", command=root.quit)
    menubar.add_cascade(label="ملف", menu=file_menu)
    
    # قائمة حديثة
    recent_menu = tk.Menu(file_menu, tearoff=0)
    file_menu.add_cascade(label="ملفات حديثة", menu=recent_menu)
    root.recent_menu = recent_menu
    
    # قائمة أدوات
    tools_menu = tk.Menu(menubar, tearoff=0)
    tools_menu.add_command(label="تحليل سريع", command=quick_analysis)
    tools_menu.add_command(label="تنظيف آلي", command=automated_data_cleaning)
    tools_menu.add_command(label="تحليل إحصائي", command=advanced_statistical_analysis)
    menubar.add_cascade(label="أدوات", menu=tools_menu)
    
    # قائمة مساعدة
    help_menu = tk.Menu(menubar, tearoff=0)
    help_menu.add_command(label="حول", command=lambda: messagebox.showinfo("حول", "نظام تحليل البيانات المتقدم v3.0"))
    menubar.add_cascade(label="مساعدة", menu=help_menu)
    
    root.config(menu=menubar)

def create_advanced_interface():
    global root, notebook, status_var, preview_text, result_display, table_frame
    
    root = tk.Tk()
    root.title("نظام تحليل البيانات المتقدم v3.0")
    root.geometry("1400x900")
    root.configure(bg=COLORS["dark" if is_dark_mode else "light"]["bg"])
    
    create_advanced_menu()
    create_enhanced_header()
    
    main_frame = tk.Frame(root, bg=COLORS["dark" if is_dark_mode else "light"]["bg"])
    main_frame.pack(fill="both", expand=True, padx=10, pady=10)
    
    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill="both", expand=True)
    
    create_enhanced_analysis_tab()
    create_advanced_visualization_tab()
    create_machine_learning_tab()
    create_statistical_analysis_tab()
    create_data_management_tab()
    
    create_enhanced_status_bar()
    
    return root

def create_enhanced_header():
    header_frame = tk.Frame(root, bg=COLORS["dark" if is_dark_mode else "light"]["header"], height=100)
    header_frame.pack(fill="x", padx=10, pady=5)
    header_frame.pack_propagate(False)
    
    # العنوان الرئيسي مع أيقونات
    title_frame = tk.Frame(header_frame, bg=COLORS["dark" if is_dark_mode else "light"]["header"])
    title_frame.pack(fill="x", padx=20, pady=10)
    
    tk.Label(title_frame, 
             text="🧠 نظام تحليل البيانات المتقدم v3.0",
             font=("Arial", 24, "bold"), 
             bg=COLORS["dark" if is_dark_mode else "light"]["header"],
             fg=COLORS["dark" if is_dark_mode else "light"]["highlight"]).pack(side="left")
    
    # أزرار التحكم السريع المتقدمة
    control_frame = tk.Frame(header_frame, bg=COLORS["dark" if is_dark_mode else "light"]["header"])
    control_frame.pack(side="right", padx=20, pady=10)
    
    quick_actions = [
        ("🚀 تحليل سريع", quick_analysis),
        ("📊 لوحة التحكم", show_dashboard),
        ("🤖 ذكاء اصطناعي", machine_learning_predictions),
        ("📈 تصدير تقرير", export_dashboard)
    ]
    
    for text, command in quick_actions:
        btn = tk.Button(control_frame, text=text, font=("Arial", 10), 
                       bg=COLORS["dark" if is_dark_mode else "light"]["button_bg"],
                       fg="white", command=command)
        btn.pack(side="left", padx=5)

def create_enhanced_analysis_tab():
    analysis_tab = ttk.Frame(notebook)
    notebook.add(analysis_tab, text="📊 التحليل المتقدم")
    
    # استخدام PanedWindow للتقسيم المتقدم
    main_paned = ttk.PanedWindow(analysis_tab, orient=tk.HORIZONTAL)
    main_paned.pack(fill="both", expand=True, padx=10, pady=10)
    
    # اللوحة اليسرى - أدوات التحكم
    left_panel = ttk.Frame(main_paned)
    main_paned.add(left_panel, weight=1)
    
    # اللوحة اليمنى - النتائج
    right_panel = ttk.Frame(main_paned)
    main_paned.add(right_panel, weight=2)
    
    create_analysis_control_panel(left_panel)
    create_analysis_results_panel(right_panel)

def create_analysis_control_panel(parent):
    """لوحة تحكم التحليل المتقدمة"""
    
    # بطاقة تحميل البيانات
    load_card = create_card(parent, "تحميل البيانات")
    
    tk.Button(load_card, text="📂 تحميل ملف بيانات", command=load_file,
              bg=COLORS["dark" if is_dark_mode else "light"]["button_bg"],
              fg="white", font=("Arial", 11)).pack(fill="x", pady=5)
    
    # بطاقة التحليل السريع
    quick_card = create_card(parent, "التحليل السريع")
    
    quick_actions = [
        ("🔍 فحص جودة البيانات", lambda: show_quality_report(features.detect_data_quality_issues(data_frame)) if data_frame is not None else messagebox.showwarning("تحذير", "لا توجد بيانات!")),
        ("✨ تنظيف آلي", automated_data_cleaning),
        ("📈 إحصائيات متقدمة", advanced_statistical_analysis),
        ("🔗 تحليل الارتباط", correlation_analysis)
    ]
    
    for text, command in quick_actions:
        tk.Button(quick_card, text=text, command=command,
                 bg=COLORS["dark" if is_dark_mode else "light"]["accent"],
                 fg=COLORS["dark" if is_dark_mode else "light"]["text"]).pack(fill="x", pady=2)
    
    # بطاقة التحليل المتقدم
    advanced_card = create_card(parent, "التحليل المتقدم")
    
    advanced_actions = [
        ("⏰ تحليل زمني", time_series_analysis),
        ("🎯 تنبؤ بالذكاء الاصطناعي", machine_learning_predictions),
        ("📊 مصفوفة الارتباط", correlation_analysis),
        ("🔍 اكتشاف الأنماط", pattern_detection)
    ]
    
    for text, command in advanced_actions:
        tk.Button(advanced_card, text=text, command=command,
                 bg=COLORS["dark" if is_dark_mode else "light"]["success"]).pack(fill="x", pady=2)

def create_card(parent, title):
    """إنشاء بطاقة واجهة مستخدم"""
    card = tk.Frame(parent, bg=COLORS["dark" if is_dark_mode else "light"]["card"],
                   relief="raised", bd=1)
    card.pack(fill="x", pady=5, padx=5)
    
    tk.Label(card, text=title, font=("Arial", 12, "bold"),
            bg=COLORS["dark" if is_dark_mode else "light"]["card"]).pack(pady=5)
    
    content_frame = tk.Frame(card, bg=COLORS["dark" if is_dark_mode else "light"]["card"])
    content_frame.pack(fill="x", padx=10, pady=5)
    
    return content_frame

def create_analysis_results_panel(parent):
    """لوحة نتائج التحليل المتقدمة"""
    notebook_results = ttk.Notebook(parent)
    notebook_results.pack(fill="both", expand=True)
    
    # تبويب معاينة البيانات
    preview_frame = ttk.Frame(notebook_results)
    notebook_results.add(preview_frame, text="👀 معاينة البيانات")
    
    global preview_text
    preview_text = scrolledtext.ScrolledText(preview_frame, font=("Consolas", 10))
    preview_text.pack(fill="both", expand=True, padx=10, pady=10)
    
    # تبويب النتائج
    results_frame = ttk.Frame(notebook_results)
    notebook_results.add(results_frame, text="📈 النتائج")
    
    global result_display
    result_display = scrolledtext.ScrolledText(results_frame, font=("Arial", 11))
    result_display.pack(fill="both", expand=True, padx=10, pady=10)
    
    # تبويب الجدول
    global table_frame
    table_frame = ttk.Frame(notebook_results)
    notebook_results.add(table_frame, text="📋 عرض الجدول")

def create_advanced_visualization_tab():
    vis_tab = ttk.Frame(notebook)
    notebook.add(vis_tab, text="📊 التصور المتقدم")
    
    tk.Label(vis_tab, text="🎨 واجهة التصور البياني المتقدمة", 
             font=("Arial", 16, "bold")).pack(pady=20)
    
    # شبكة أزرار التصور
    viz_frame = tk.Frame(vis_tab)
    viz_frame.pack(fill="both", expand=True, padx=20, pady=10)
    
    visualization_types = [
        ("📈 مخططات خطية", "line"),
        ("📊 أعمدة", "bar"),
        ("🔴 مبعثر", "scatter"),
        ("📦 صندوقي", "box"),
        ("🎻 كماني", "violin"),
        ("🔥 خريطة حرارية", "heatmap"),
    ]
    
    for i, (text, viz_type) in enumerate(visualization_types):
        row, col = i // 3, i % 3
        btn = tk.Button(viz_frame, text=text, font=("Arial", 10),
                       command=lambda vt=viz_type: create_visualization(vt))
        btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
    
    viz_frame.grid_columnconfigure((0,1,2), weight=1)

def create_visualization(viz_type):
    """إنشاء التصورات البيانية"""
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتصور!")
        return
    
    try:
        fig = features.create_advanced_chart(data_frame, viz_type, 
                                           data_frame.columns[0] if len(data_frame.columns) > 0 else None,
                                           data_frame.columns[1] if len(data_frame.columns) > 1 else None)
        if fig:
            plt.show()
        else:
            messagebox.showwarning("تحذير", "تعذر إنشاء الرسم البياني!")
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل إنشاء الرسم: {str(e)}")

def create_machine_learning_tab():
    ml_tab = ttk.Frame(notebook)
    notebook.add(ml_tab, text="🤖 الذكاء الاصطناعي")
    
    tk.Label(ml_tab, text="🧠 نماذج التعلم الآلي المتقدمة", 
             font=("Arial", 16, "bold")).pack(pady=20)
    
    ml_frame = tk.Frame(ml_tab)
    ml_frame.pack(fill="both", expand=True, padx=20, pady=10)
    
    ml_algorithms = [
        ("📈 الانحدار الخطي", "linear_regression"),
        ("🌳 الغابة العشوائية", "random_forest"),
        ("🔍 التجميع", "kmeans"),
        ("📊 الانحدار اللوجستي", "logistic_regression"),
    ]
    
    for i, (text, algo) in enumerate(ml_algorithms):
        row, col = i // 2, i % 2
        btn = tk.Button(ml_frame, text=text, font=("Arial", 10),
                       command=lambda a=algo: run_advanced_ml(a))
        btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
    
    ml_frame.grid_columnconfigure((0,1), weight=1)

def run_advanced_ml(algorithm):
    """تشغيل خوارزميات متقدمة"""
    machine_learning_predictions()

def create_statistical_analysis_tab():
    stats_tab = ttk.Frame(notebook)
    notebook.add(stats_tab, text="📈 الإحصاء المتقدم")
    
    tk.Label(stats_tab, text="📊 التحليل الإحصائي المتقدم", 
             font=("Arial", 16, "bold")).pack(pady=20)
    
    stats_frame = tk.Frame(stats_tab)
    stats_frame.pack(fill="both", expand=True, padx=20, pady=10)
    
    statistical_tests = [
        ("📏 اختبار T", "t_test"),
        ("📐 ANOVA", "anova"),
        ("📊 اختبار كاي squared", "chi_square"),
        ("📈 الانحدار", "regression"),
        ("📉 الارتباط", "correlation"),
        ("📋 الوصف الإحصائي", "descriptive")
    ]
    
    for i, (text, test) in enumerate(statistical_tests):
        row, col = i // 3, i % 3
        btn = tk.Button(stats_frame, text=text, font=("Arial", 10),
                       command=lambda t=test: run_statistical_test(t))
        btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
    
    stats_frame.grid_columnconfigure((0,1,2), weight=1)

def run_statistical_test(test_type):
    """تشغيل الاختبارات الإحصائية"""
    if test_type == "descriptive":
        advanced_statistical_analysis()
    elif test_type == "correlation":
        correlation_analysis()
    else:
        messagebox.showinfo("معلومات", f"اختبار {test_type} قيد التطوير")

def create_data_management_tab():
    data_tab = ttk.Frame(notebook)
    notebook.add(data_tab, text="💾 إدارة البيانات")
    
    tk.Label(data_tab, text="🛠️ أدوات إدارة البيانات المتقدمة", 
             font=("Arial", 16, "bold")).pack(pady=20)
    
    management_frame = tk.Frame(data_tab)
    management_frame.pack(fill="both", expand=True, padx=20, pady=10)
    
    management_tools = [
        ("🧹 تنظيف البيانات", automated_data_cleaning),
        ("🔧 هندسة المميزات", advanced_feature_engineering),
        ("💾 نسخ احتياطي", backup_data),
        ("📊 تحويل البيانات", transform_data),
        ("🔍 فحص الجودة", lambda: show_quality_report(features.detect_data_quality_issues(data_frame)) if data_frame is not None else messagebox.showwarning("تحذير", "لا توجد بيانات!"))
    ]
    
    for i, (text, command) in enumerate(management_tools):
        row, col = i // 3, i % 3
        btn = tk.Button(management_frame, text=text, font=("Arial", 10), command=command)
        btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
    
    management_frame.grid_columnconfigure((0,1,2), weight=1)

def import_export_data():
    """استيراد وتصدير البيانات"""
    messagebox.showinfo("معلومات", "أداة الاستيراد/التصدير قيد التطوير")

def backup_data():
    """نسخ احتياطي للبيانات"""
    try:
        if data_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = f"backup_data_{timestamp}.csv"
            data_frame.to_csv(backup_file, index=False)
            messagebox.showinfo("نجاح", f"تم النسخ الاحتياطي إلى: {backup_file}")
        else:
            messagebox.showwarning("تحذير", "لا توجد بيانات للنسخ الاحتياطي")
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل النسخ الاحتياطي: {str(e)}")

def transform_data():
    """تحويل البيانات"""
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتحويل!")
        return
    
    transform_window = tk.Toplevel(root)
    transform_window.title("تحويل البيانات")
    transform_window.geometry("300x200")
    
    transforms = [
        ("Normalization", normalize_data),
        ("Standardization", standardize_data),
        ("Log Transform", log_transform)
    ]
    
    for text, command in transforms:
        tk.Button(transform_window, text=text, command=command).pack(pady=5)

def create_enhanced_status_bar():
    global status_var
    
    status_frame = tk.Frame(root, bg=COLORS["dark" if is_dark_mode else "light"]["accent"])
    status_frame.pack(fill="x", side="bottom")
    
    status_var = tk.StringVar()
    status_var.set("✅ جاهز - نظام تحليل البيانات المتقدم v3.0")
    
    status_label = tk.Label(status_frame, textvariable=status_var, 
                           font=("Arial", 10),
                           bg=COLORS["dark" if is_dark_mode else "light"]["accent"],
                           fg=COLORS["dark" if is_dark_mode else "light"]["text"])
    status_label.pack(side="left", padx=10, pady=2)
    
    # معلومات النظام
    system_info = f"Python {platform.python_version()} | {platform.system()}"
    system_label = tk.Label(status_frame, text=system_info, font=("Arial", 9))
    system_label.pack(side="right", padx=10, pady=2)

def show_dashboard():
    """عرض لوحة التحكم الرئيسية"""
    dashboard_window = tk.Toplevel(root)
    dashboard_window.title("لوحة تحليل البيانات - Dashboard")
    dashboard_window.geometry("1000x700")
    
    tk.Label(dashboard_window, text="📊 لوحة تحليل البيانات الشاملة", 
             font=("Arial", 20, "bold")).pack(pady=20)
    
    # إحصائيات سريعة
    if data_frame is not None:
        stats_text = f"""
        📈 إحصائيات البيانات:
        • عدد الصفوف: {data_frame.shape[0]:,}
        • عدد الأعمدة: {data_frame.shape[1]}
        • القيم المفقودة: {data_frame.isnull().sum().sum():,}
        • الذاكرة: {data_frame.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB
        """
        tk.Label(dashboard_window, text=stats_text, font=("Arial", 12), justify="left").pack()
    else:
        tk.Label(dashboard_window, text="⚠️ لا توجد بيانات محملة", font=("Arial", 14)).pack()

def update_data_preview():
    if data_frame is not None and 'preview_text' in globals():
        preview_text.delete(1.0, tk.END)
        preview_rows = user_preferences.get("data_preview_rows", 15)
        preview_text.insert(tk.END, data_frame.head(preview_rows).to_string())

def quick_analysis():
    if data_frame is not None:
        advanced_statistical_analysis()
    else:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتحليل!")

def update_status(msg):
    """تحديث شريط الحالة"""
    global status_var
    if 'status_var' in globals():
        status_var.set(msg)

# ------------------- التشغيل الرئيسي -------------------
def main():
    global root
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