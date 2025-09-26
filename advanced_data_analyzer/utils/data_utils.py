import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tkinter import messagebox
import tkinter as tk
from src.features import DataAnalysisFeatures
from ui.components import update_data_preview, update_status, show_statistical_results, show_time_series_results
from utils.file_utils import show_dataframe_in_table

# globals: data_frame, root, table_frame

def automated_data_cleaning():
    """تنظيف البيانات الآلي"""
    global data_frame
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتنظيف!")
        return
    
    try:
        data_frame = DataAnalysisFeatures.automated_data_cleaning(data_frame)
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
        import matplotlib.pyplot as plt
        import seaborn as sns
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
            
            analysis = DataAnalysisFeatures.time_series_analysis(data_frame, date_col, value_col)
            show_time_series_results(analysis, value_col)
            ts_window.destroy()
        
        tk.Button(ts_window, text="تشغيل التحليل", command=run_analysis).pack(pady=10)
        
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل التحليل الزمني: {str(e)}")

def advanced_feature_engineering():
    """هندسة المميزات المتقدمة"""
    global data_frame
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتحويل!")
        return
    
    try:
        data_frame = DataAnalysisFeatures.feature_engineering(data_frame)
        update_data_preview()
        show_dataframe_in_table()
        messagebox.showinfo("نجاح", "تمت هندسة المميزات بنجاح!")
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل هندسة المميزات: {str(e)}")

def normalize_data():
    """تطبيع البيانات"""
    global data_frame
    if data_frame is not None:
        numeric_df = data_frame.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            scaler = MinMaxScaler()
            data_frame[numeric_df.columns] = scaler.fit_transform(numeric_df)
            update_data_preview()
            show_dataframe_in_table()
            messagebox.showinfo("نجاح", "تم تطبيع البيانات بنجاح!")
        else:
            messagebox.showwarning("تحذير", "لا توجد أعمدة رقمية للتطبيع!")
    else:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتطبيع!")

def standardize_data():
    """توحيد البيانات"""
    global data_frame
    if data_frame is not None:
        numeric_df = data_frame.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            scaler = StandardScaler()
            data_frame[numeric_df.columns] = scaler.fit_transform(numeric_df)
            update_data_preview()
            show_dataframe_in_table()
            messagebox.showinfo("نجاح", "تم توحيد البيانات بنجاح!")
        else:
            messagebox.showwarning("تحذير", "لا توجد أعمدة رقمية للتوحيد!")
    else:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتوحيد!")

def log_transform():
    """تحويل لوغاريتمي"""
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
        else:
            messagebox.showwarning("تحذير", "لا توجد أعمدة رقمية للتحويل!")
    else:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتحويل!")

def pattern_detection():
    """اكتشاف الأنماط في البيانات"""
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

def quick_analysis():
    """تحليل سريع"""
    if data_frame is not None:
        advanced_statistical_analysis()
    else:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتحليل!")