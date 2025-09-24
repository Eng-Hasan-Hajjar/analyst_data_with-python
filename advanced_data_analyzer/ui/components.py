import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import platform
from datetime import datetime
from config.settings import COLORS, is_dark_mode, user_preferences
# imports أخرى إذا لزم

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

def show_quality_report(issues: list):
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

# ... (أضف show_statistical_results, show_time_series_results, show_ml_results هنا بنفس الطريقة)
def show_statistical_results(results: dict):
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

def show_time_series_results(analysis: dict, value_col: str):
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

def show_ml_results(result: str, algorithm: str):
    """عرض نتائج التعلم الآلي"""
    results_window = tk.Toplevel(root)
    results_window.title("نتائج التعلم الآلي")
    results_window.geometry("500x200")
    
    tk.Label(results_window, text=f"نتائج {algorithm}", font=("Arial", 14, "bold")).pack(pady=10)
    tk.Label(results_window, text=result, font=("Arial", 12)).pack(pady=10)
    tk.Button(results_window, text="حفظ النموذج", command=lambda: save_ml_model(algorithm, result)).pack(pady=10)