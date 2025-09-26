from tkinter import messagebox, ttk
import tkinter as tk
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from ui.components import show_ml_results
# globals: data_frame, root

def save_ml_model(algorithm: str, result: str):
    """حفظ نموذج التعلم الآلي"""
    messagebox.showinfo("معلومات", f"حفظ النموذج {algorithm} قيد التطوير: \n{result}")

def machine_learning_predictions():
    """تنبؤات التعلم الآلي"""
    global data_frame
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتحليل!")
        return
    
    try:
        # نافذة اختيار الأعمدة
        ml_window = tk.Toplevel(root)
        ml_window.title("تنبؤات التعلم الآلي")
        ml_window.geometry("400x300")
        
        tk.Label(ml_window, text="اختر عمود الهدف:").pack(pady=5)
        target_selector = ttk.Combobox(ml_window)
        target_selector.pack(pady=5)
        
        tk.Label(ml_window, text="اختر الأعمدة المميزة:").pack(pady=5)
        features_selector = ttk.Combobox(ml_window)
        features_selector.pack(pady=5)
        
        tk.Label(ml_window, text="اختر الخوارزمية:").pack(pady=5)
        algo_selector = ttk.Combobox(ml_window, values=["Logistic Regression", "Random Forest"])
        algo_selector.pack(pady=5)
        
        target_selector['values'] = list(data_frame.columns)
        features_selector['values'] = list(data_frame.columns)
        
        def run_ml_analysis():
            target_col = target_selector.get()
            feature_cols = features_selector.get()
            algorithm = algo_selector.get()
            
            if not target_col or not feature_cols or not algorithm:
                messagebox.showwarning("تحذير", "يجب اختيار جميع الحقول!")
                return
            
            try:
                # تحضير البيانات
                X = data_frame[[feature_cols]]
                y = data_frame[target_col]
                
                # تقسيم البيانات
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # اختيار النموذج
                if algorithm == "Logistic Regression":
                    model = LogisticRegression()
                else:
                    model = RandomForestClassifier()
                
                # تدريب النموذج
                model.fit(X_train, y_train)
                
                # التنبؤ
                y_pred = model.predict(X_test)
                
                # إنشاء تقرير الأداء
                report = classification_report(y_test, y_pred)
                
                # عرض النتائج
                show_ml_results(report, algorithm)
                ml_window.destroy()
                
            except Exception as e:
                messagebox.showerror("خطأ", f"فشل تحليل التعلم الآلي: {str(e)}")
        
        tk.Button(ml_window, text="تشغيل التحليل", command=run_ml_analysis).pack(pady=10)
        
    except Exception as e:
        messagebox.showerror("خطأ", f"فشل إعداد التعلم الآلي: {str(e)}")