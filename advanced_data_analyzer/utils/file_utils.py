import pandas as pd
import os
import json
from datetime import datetime
from tkinter import filedialog, messagebox
from config.settings import is_arabic, user_preferences
from src.database import AdvancedDataAnalysisDB
from src.features import DataAnalysisFeatures
from ui.components import update_status, update_data_preview, show_quality_report
# globals: data_frame, current_session_id, root, table_frame

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
            advanced_db = AdvancedDataAnalysisDB()
            current_session_id = advanced_db.save_analysis_session(
                os.path.basename(file),
                data_frame.shape,
                "data_loading",
                {"file_type": file.split('.')[-1], "rows": data_frame.shape[0], "columns": data_frame.shape[1]}
            )
            
            # تحليل جودة البيانات
            quality_issues = DataAnalysisFeatures.detect_data_quality_issues(data_frame)
            if quality_issues:
                show_quality_report(quality_issues)
            
            update_status("تم تحميل البيانات بنجاح!" if is_arabic else "Data loaded successfully!")
            
        except Exception as e:
            update_status("فشل تحميل الملف" if is_arabic else "Failed to load file")
            messagebox.showerror("خطأ", f"فشل تحميل الملف: {str(e)}")

def update_recent_files_menu(file: str = None):
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

def load_recent_file(file: str):
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
    global data_table
    if data_frame is None or table_frame is None:
        return
    
    for widget in table_frame.winfo_children():
        widget.destroy()
    
    # إنشاء Treeview مع شريط تمرير
    tree_frame = tk.Frame(table_frame)
    tree_frame.pack(fill="both", expand=True)
    
    tree_scroll = ttk.Scrollbar(tree_frame)
    tree_scroll.pack(side="right", fill="y")
    
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

def create_html_report() -> str:
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