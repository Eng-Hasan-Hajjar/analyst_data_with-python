import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from tkinter import messagebox
from datetime import datetime
import json
from ui.components import show_ml_results, save_ml_model

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

def run_machine_learning(algorithm: str) -> str:
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

def run_advanced_ml(algorithm: str):
    """تشغيل خوارزميات متقدمة"""
    machine_learning_predictions()

def save_ml_model(algorithm: str, result: str):
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