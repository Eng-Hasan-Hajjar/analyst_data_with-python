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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# ------------------- إعداد الخطوط العربية -------------------
try:
    arabic_font = fm.FontProperties(fname='arial.ttf')  # يمكن تغييرها لخط عربي
except:
    arabic_font = None

# ------------------- الحالة والمتغيرات العالمية -------------------
is_arabic = True
is_dark_mode = True
data_frame = None
current_plot = None
user_notes = {}
user_settings = {
    "language": "arabic",
    "theme": "dark",
    "font_size": 12,
    "auto_save": False
}

# ------------------- الألوان حسب السمة -------------------
COLORS = {
    "dark": {
        "bg": "#1e1e2f",
        "frame_bg": "#2a2a40",
        "text": "white",
        "button_bg": "#3b8ad8",
        "accent": "#6a6a9c",
        "success": "#4caf50",
        "warning": "#ff9800",
        "error": "#f44336"
    },
    "light": {
        "bg": "#f0f0f0",
        "frame_bg": "#ffffff",
        "text": "black",
        "button_bg": "#2196F3",
        "accent": "#e0e0e0",
        "success": "#4CAF50",
        "warning": "#FFC107",
        "error": "#F44336"
    }
}

# ------------------- البيانات المرجعية للمكتبات -------------------
LIBRARY_DATA = {
    "Pandas": {
        "Basic Info": ["df.head()", "df.tail()", "df.shape", "df.columns", "df.dtypes", "df.info()"],
        "Statistics": ["df.describe()", "df.mean()", "df.median()", "df.mode()", "df.std()", "df.sum()"],
        "Data Cleaning": ["df.dropna()", "df.fillna(value)", "df.drop_duplicates()"],
        "Filtering": ["df[df['column'] > value]", "df[df['column'] == value]"],
        "Sorting": ["df.sort_values(by='column')", "df.sort_index()"],
        "Date & Time": ["pd.to_datetime(df['column'])"],
        "Grouping": ["df.groupby('column')", "df.groupby('column').mean()"],
        "Merging": ["pd.concat([df1, df2])", "pd.merge(df1, df2, on='column')"]
    },
    "Matplotlib": {
        "Line Plots": ["plt.plot(x, y)", "plt.title('Title')", "plt.xlabel('X')", "plt.ylabel('Y')", 
                       "plt.grid(True)", "plt.legend()", "plt.show()"],
        "Scatter Plots": ["plt.scatter(x, y)", "plt.scatter(x, y, marker='o', color='r')"],
        "Bar Plots": ["plt.bar(x, y)", "plt.barh(x, y)"],
        "Customization": ["plt.plot(x,y, linestyle='--', linewidth=2, color='blue', marker='o')", 
                          "plt.xlim()", "plt.ylim()"],
        "Subplots": ["fig, ax = plt.subplots()", "ax.plot(x, y)"],
        "Histograms": ["plt.hist(data, bins=10)"]
    },
    "Seaborn": {
        "Line Plot": ["sns.lineplot(x='x', y='y', data=df)"],
        "Scatter Plot": ["sns.scatterplot(x='x', y='y', data=df)"],
        "Bar Plot": ["sns.barplot(x='x', y='y', data=df)"],
        "Box Plot": ["sns.boxplot(x='x', y='y', data=df)"],
        "Heatmap": ["sns.heatmap(df.corr(), annot=True)"],
        "Histogram": ["sns.histplot(df['column'], bins=10)"],
        "Customization": ["sns.set_style('darkgrid')", "sns.set_palette('pastel')"],
        "Regression": ["sns.regplot(x='x', y='y', data=df)"]
    },
    "Numpy": {
        "Array Creation": ["np.array([1,2,3])", "np.zeros(5)", "np.ones(5)", "np.arange(10)"],
        "Array Operations": ["np.add(a, b)", "np.subtract(a, b)", "np.multiply(a, b)", "np.divide(a, b)"],
        "Statistics": ["np.mean(arr)", "np.median(arr)", "np.std(arr)", "np.var(arr)"],
        "Random": ["np.random.rand(5)", "np.random.randint(1, 10, 5)"],
        "Linear Algebra": ["np.dot(a, b)", "np.linalg.inv(matrix)"]
    },
    "Scipy": {
        "Statistics": ["scipy.stats.ttest_ind(a, b)", "scipy.stats.pearsonr(x, y)"],
        "Optimization": ["scipy.optimize.minimize(func, x0)"],
        "Integration": ["scipy.integrate.quad(func, a, b)"]
    },
    "Scikit-learn": {
        "Preprocessing": ["StandardScaler()", "MinMaxScaler()"],
        "Clustering": ["KMeans(n_clusters=3)"],
        "Regression": ["LinearRegression()"],
        "Model Evaluation": ["metrics.accuracy_score(y_true, y_pred)"]
    }
}

# ------------------- الوظائف الأساسية -------------------
def load_file():
    global data_frame
    file = filedialog.askopenfilename(filetypes=[
        ("Excel Files", "*.xlsx"), 
        ("CSV Files", "*.csv"),
        ("JSON Files", "*.json"),
        ("Text Files", "*.txt")
    ])
    if file:
        try:
            if file.endswith('.csv'):
                data_frame = pd.read_csv(file)
            elif file.endswith('.json'):
                data_frame = pd.read_json(file)
            elif file.endswith('.xlsx'):
                data_frame = pd.read_excel(file)
            else:
                data_frame = pd.read_csv(file, delimiter='\t')
            
            update_data_preview()
            populate_table(data_frame)
            update_column_selector()
            messagebox.showinfo(
                title="تم التحميل" if is_arabic else "Loaded",
                message="تم تحميل البيانات بنجاح!" if is_arabic else "Data loaded successfully!"
            )
        except Exception as e:
            messagebox.showerror(
                title="خطأ" if is_arabic else "Error",
                message=f"فشل تحميل الملف: {str(e)}" if is_arabic else f"Failed to load file: {str(e)}"
            )

def update_data_preview():
    if data_frame is not None:
        preview_text.delete(1.0, tk.END)
        preview_text.insert(tk.END, data_frame.head(10).to_string())
        
        # تحديث معلومات البيانات
        info_text = f"""
        شكل البيانات: {data_frame.shape}
        الأعمدة: {list(data_frame.columns)}
        القيم المفقودة: {data_frame.isnull().sum().sum()}
        الأنواع: 
        {data_frame.dtypes}
        """
        data_info_label.config(text=info_text)

def update_column_selector():
    if data_frame is not None:
        columns = list(data_frame.columns)
        x_axis_selector['values'] = columns
        y_axis_selector['values'] = columns
        cluster_column_selector['values'] = columns
        regression_x_selector['values'] = columns
        regression_y_selector['values'] = columns

def run_analysis():
    global data_frame
    try:
        # قراءة من إدخال المستخدم أو DataFrame
        input_text = input_entry.get()
        if input_text:
            arr = np.fromstring(input_text, sep=',')
            df = pd.DataFrame({'Values': arr})
        elif data_frame is not None:
            df = data_frame.copy()
        else:
            messagebox.showwarning(
                title="تحذير" if is_arabic else "Warning",
                message="لا توجد بيانات!" if is_arabic else "No data provided!"
            )
            return

        option = selected_option.get()
        result_text = ""

        # تحليلات أساسية
        if option in ["أماكن الرقم 4", "Where value is 4"]:
            res = np.where(df['Values'] == 4)[0]
            result_text = str(res)
        elif option in ["الأرقام الزوجية", "Even numbers"]:
            res = df[df['Values'] % 2 == 0]['Values'].values
            result_text = str(res)
        elif option in ["الأرقام الفردية", "Odd numbers"]:
            res = df[df['Values'] % 2 == 1]['Values'].values
            result_text = str(res)
        elif option in ["فرز المصفوفة", "Sort array"]:
            res = np.sort(df['Values'].values)
            result_text = str(res)
        elif option in ["فلترة > 42", "Filter > 42"]:
            res = df[df['Values'] > 42]['Values'].values
            result_text = str(res)
        elif option in ["فلترة زوجية", "Filter even"]:
            res = df[df['Values'] % 2 == 0]['Values'].values
            result_text = str(res)
        elif option in ["عدد عشوائي 0-100", "Random int 0-100"]:
            res = random.randint(100)
            result_text = str(res)
        elif option in ["5 أعداد صحيحة", "5 random integers"]:
            res = random.randint(100, size=5)
            result_text = str(res)
        elif option in ["5 أعداد عشرية", "5 random floats"]:
            res = random.rand(5)
            result_text = str(res)
        elif option in ["عنصر عشوائي من المصفوفة", "Random choice from array"]:
            res = random.choice(df['Values'].values)
            result_text = str(res)
        elif option in ["إحصائيات", "Statistics"]:
            stats = df['Values'].describe()
            result_text = str(stats)
        elif option in ["التوزيع الطبيعي", "Normal distribution"]:
            res = random.normal(loc=0, scale=1, size=100)
            result_text = str(res[:10]) + " ..."
        elif option in ["توزيع بواسون", "Poisson distribution"]:
            res = random.poisson(lam=5, size=100)
            result_text = str(res[:10]) + " ..."

        result_display.config(text=result_text)
        populate_table(df)  # عرض البيانات في الجدول
    except Exception as e:
        result_display.config(text=f"خطأ: {e}" if is_arabic else f"Error: {e}")

def detect_outliers():
    global data_frame
    try:
        if data_frame is None:
            messagebox.showwarning(
                title="تحذير" if is_arabic else "Warning",
                message="لا توجد بيانات!" if is_arabic else "No data loaded!"
            )
            return
        df = data_frame.copy()
        # Z-score
        mean = df['Values'].mean()
        std = df['Values'].std()
        z_scores = (df['Values'] - mean) / std
        outliers_z = df[np.abs(z_scores) > 2]['Values'].values

        # IQR
        Q1 = df['Values'].quantile(0.25)
        Q3 = df['Values'].quantile(0.75)
        IQR = Q3 - Q1
        outliers_iqr = df[(df['Values'] < Q1 - 1.5*IQR) | (df['Values'] > Q3 + 1.5*IQR)]['Values'].values

        result_display.config(text=f"{'القيم الشاذة Z-score: ' if is_arabic else 'Outliers Z-score: '}{outliers_z}\n"
                                   f"{'القيم الشاذة IQR: ' if is_arabic else 'Outliers IQR: '}{outliers_iqr}")
    except Exception as e:
        result_display.config(text=f"خطأ: {e}" if is_arabic else f"Error: {e}")

def draw_plot(plot_type):
    global data_frame, current_plot
    try:
        if data_frame is None:
            arr = np.fromstring(input_entry.get(), sep=',')
            df = pd.DataFrame({'Values': arr})
        else:
            df = data_frame.copy()

        # الحصول على الأعمدة المحددة
        x_col = x_axis_selector.get() if x_axis_selector.get() else 'Values'
        y_col = y_axis_selector.get() if y_axis_selector.get() else 'Values'

        fig, ax = plt.subplots(figsize=(6, 4))
        
        if plot_type == "hist":
            ax.hist(df[x_col], bins=10, color="#3b8ad8", edgecolor="black")
            ax.set_title("Histogram" if not is_arabic else "مخطط التوزيع")
            ax.set_xlabel(x_col if not is_arabic else x_col)
            ax.set_ylabel("Frequency" if not is_arabic else "التكرار")
        elif plot_type == "box":
            ax.boxplot(df[x_col], patch_artist=True, boxprops=dict(facecolor="#3b8ad8"))
            ax.set_title("Boxplot" if not is_arabic else "مخطط الصندوق")
        elif plot_type == "scatter" and y_col:
            ax.scatter(df[x_col], df[y_col], color="#3b8ad8")
            ax.set_title("Scatter Plot" if not is_arabic else "مخطط الانتشار")
            ax.set_xlabel(x_col if not is_arabic else x_col)
            ax.set_ylabel(y_col if not is_arabic else y_col)
        elif plot_type == "line" and y_col:
            ax.plot(df[x_col], df[y_col], color="#3b8ad8")
            ax.set_title("Line Plot" if not is_arabic else "مخطط الخط")
            ax.set_xlabel(x_col if not is_arabic else x_col)
            ax.set_ylabel(y_col if not is_arabic else y_col)
        elif plot_type == "bar" and y_col:
            ax.bar(df[x_col], df[y_col], color="#3b8ad8")
            ax.set_title("Bar Plot" if not is_arabic else "مخطط الأعمدة")
            ax.set_xlabel(x_col if not is_arabic else x_col)
            ax.set_ylabel(y_col if not is_arabic else y_col)

        plt.tight_layout()
        
        # عرض الرسم في واجهة التطبيق
        if current_plot:
            current_plot.get_tk_widget().destroy()
            
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        toolbar = NavigationToolbar2Tk(canvas, plot_frame)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        current_plot = canvas
        
    except Exception as e:
        messagebox.showerror(
            title="خطأ" if is_arabic else "Error",
            message=f"خطأ في الرسم: {e}" if is_arabic else f"Plot Error: {e}"
        )

def save_to_excel():
    global data_frame
    if data_frame is None:
        messagebox.showwarning(
            title="تحذير" if is_arabic else "Warning",
            message="لا توجد بيانات للحفظ!" if is_arabic else "No data to save!"
        )
        return
    file = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel Files", "*.xlsx")])
    if file:
        df = data_frame.copy()
        # إضافة التحليل الأساسي
        df['Mean'] = df['Values'].mean()
        df['Median'] = df['Values'].median()
        df['Std'] = df['Values'].std()
        df['Max'] = df['Values'].max()
        df['Min'] = df['Values'].min()
        df.to_excel(file, index=False)
        messagebox.showinfo(
            title="تم الحفظ" if is_arabic else "Saved",
            message="تم حفظ الملف بنجاح!" if is_arabic else "Excel file saved!"
        )

def populate_table(df):
    for row in result_table.get_children():
        result_table.delete(row)
    for index, row in df.iterrows():
        result_table.insert("", "end", values=list(row))

def toggle_language():
    global is_arabic
    is_arabic = not is_arabic
    update_language()
    apply_theme()

def update_language():
    if is_arabic:
        notebook.tab(0, text="تحليل البيانات")
        notebook.tab(1, text="المرجع التعليمي")
        notebook.tab(2, text="التصور البياني")
        notebook.tab(3, text="التعلم الآلي")
        notebook.tab(4, text="الإحصاءات")
        notebook.tab(5, text="الإعدادات")
        
        title.config(text="نظام تحليل البيانات المتقدم")
        input_label.config(text="🧮 أدخل مصفوفة (مثال: 1,2,3,4) أو ارفع ملف:")
        option_label.config(text="📊 اختر نوع التحليل:")
        analyze_btn.config(text="🔍 تحليل الآن")
        save_btn.config(text="💾 حفظ Excel")
        hist_btn.config(text="📊 رسم Histogram")
        box_btn.config(text="📦 Boxplot")
        scatter_btn.config(text="📈 Scatter Plot")
        line_btn.config(text="📈 Line Plot")
        bar_btn.config(text="📊 Bar Plot")
        outlier_btn.config(text="🧠 القيم الشاذة")
        load_btn.config(text="📂 رفع ملف")
        lang_btn.config(text="🔄 English")
        theme_btn.config(text="🎨 تغيير السمة")
        x_axis_label.config(text="المحور X:")
        y_axis_label.config(text="المحور Y:")
        cluster_btn.config(text="تجميع البيانات")
        regression_btn.config(text="انحدار خطي")
        stats_btn.config(text="إحصائيات متقدمة")
        corr_btn.config(text="مصفوفة الارتباط")
        prob_btn.config(text="التوزيعات الاحتمالية")
        settings_label.config(text="الإعدادات")
        font_label.config(text="حجم الخط:")
        auto_save_label.config(text="الحفظ التلقائي:")
        save_settings_btn.config(text="حفظ الإعدادات")
        reset_settings_btn.config(text="إعادة التعيين")
        
        selected_option['values'] = [
            "أماكن الرقم 4", "الأرقام الزوجية", "الأرقام الفردية",
            "فرز المصفوفة", "فلترة > 42", "فلترة زوجية",
            "عدد عشوائي 0-100", "5 أعداد صحيحة", "5 أعداد عشرية",
            "عنصر عشوائي من المصفوفة", "إحصائيات", "التوزيع الطبيعي", "توزيع بواسون"
        ]
    else:
        notebook.tab(0, text="Data Analysis")
        notebook.tab(1, text="Learning Reference")
        notebook.tab(2, text="Visualization")
        notebook.tab(3, text="Machine Learning")
        notebook.tab(4, text="Statistics")
        notebook.tab(5, text="Settings")
        
        title.config(text="Advanced Data Analysis System")
        input_label.config(text="🧮 Enter array (e.g., 1,2,3,4) or load file:")
        option_label.config(text="📊 Select analysis type:")
        analyze_btn.config(text="🔍 Analyze")
        save_btn.config(text="💾 Save Excel")
        hist_btn.config(text="📊 Plot Histogram")
        box_btn.config(text="📦 Boxplot")
        scatter_btn.config(text="📈 Scatter Plot")
        line_btn.config(text="📈 Line Plot")
        bar_btn.config(text="📊 Bar Plot")
        outlier_btn.config(text="🧠 Outliers")
        load_btn.config(text="📂 Load File")
        lang_btn.config(text="🔄 العربية")
        theme_btn.config(text="🎨 Change Theme")
        x_axis_label.config(text="X Axis:")
        y_axis_label.config(text="Y Axis:")
        cluster_btn.config(text="Cluster Data")
        regression_btn.config(text="Linear Regression")
        stats_btn.config(text="Advanced Stats")
        corr_btn.config(text="Correlation Matrix")
        prob_btn.config(text="Probability Distributions")
        settings_label.config(text="Settings")
        font_label.config(text="Font Size:")
        auto_save_label.config(text="Auto Save:")
        save_settings_btn.config(text="Save Settings")
        reset_settings_btn.config(text="Reset Settings")
        
        selected_option['values'] = [
            "Where value is 4", "Even numbers", "Odd numbers",
            "Sort array", "Filter > 42", "Filter even",
            "Random int 0-100", "5 random integers", "5 random floats",
            "Random choice from array", "Statistics", "Normal distribution", "Poisson distribution"
        ]

def apply_theme():
    theme = "dark" if is_dark_mode else "light"
    colors = COLORS[theme]
    
    root.configure(bg=colors['bg'])
    title.config(bg=colors['bg'], fg=colors['text'])
    input_frame.config(bg=colors['bg'])
    input_label.config(bg=colors['bg'], fg=colors['text'])
    option_label.config(bg=colors['bg'], fg=colors['text'])
    result_frame.config(bg=colors['frame_bg'], fg=colors['text'])
    result_display.config(bg=colors['frame_bg'], fg=colors['text'])
    button_frame.config(bg=colors['bg'])
    lang_btn.config(bg=colors['accent'], fg=colors['text'])
    theme_btn.config(bg=colors['accent'], fg=colors['text'])
    
    # تطبيق المظهر على جميع العناصر
    for widget in root.winfo_children():
        if isinstance(widget, (tk.Frame, ttk.Frame)):
            try:
                widget.config(bg=colors['bg'])
            except:
                pass
                
        try:
            if hasattr(widget, 'config'):
                widget.config(bg=colors['bg'], fg=colors['text'])
        except:
            pass

    # تحديث ألوان الأزرار
    analyze_btn.config(bg=colors['button_bg'], fg='white')
    save_btn.config(bg=colors['success'], fg='white')
    hist_btn.config(bg=colors['button_bg'], fg='white')
    box_btn.config(bg=colors['button_bg'], fg='white')
    scatter_btn.config(bg=colors['button_bg'], fg='white')
    line_btn.config(bg=colors['button_bg'], fg='white')
    bar_btn.config(bg=colors['button_bg'], fg='white')
    outlier_btn.config(bg=colors['warning'], fg='white')
    load_btn.config(bg=colors['button_bg'], fg='white')
    cluster_btn.config(bg=colors['button_bg'], fg='white')
    regression_btn.config(bg=colors['button_bg'], fg='white')
    stats_btn.config(bg=colors['button_bg'], fg='white')
    corr_btn.config(bg=colors['button_bg'], fg='white')
    prob_btn.config(bg=colors['button_bg'], fg='white')
    save_settings_btn.config(bg=colors['success'], fg='white')
    reset_settings_btn.config(bg=colors['error'], fg='white')

    # تحديث ألوان الجدول
    style = ttk.Style()
    style.theme_use('default')
    style.configure("Treeview",
                    background=colors['frame_bg'],
                    foreground=colors['text'],
                    fieldbackground=colors['frame_bg'])
    style.map('Treeview', background=[('selected', colors['accent'])], 
              foreground=[('selected', 'white')])

def toggle_theme():
    global is_dark_mode
    is_dark_mode = not is_dark_mode
    apply_theme()

def run_clustering():
    global data_frame
    try:
        if data_frame is None:
            messagebox.showwarning(
                title="تحذير" if is_arabic else "Warning",
                message="لا توجد بيانات!" if is_arabic else "No data loaded!"
            )
            return
        
        column = cluster_column_selector.get()
        if not column:
            messagebox.showwarning(
                title="تحذير" if is_arabic else "Warning",
                message="يرجى اختيار عمود!" if is_arabic else "Please select a column!"
            )
            return
        
        # تطبيق K-means clustering
        kmeans = KMeans(n_clusters=3)
        data_frame['Cluster'] = kmeans.fit_predict(data_frame[[column]])
        
        # عرض النتائج
        result_display.config(text=f"{'تم التجميع إلى 3 clusters في العمود ' if is_arabic else 'Clustered into 3 clusters in column '}{column}")
        populate_table(data_frame)
        
    except Exception as e:
        messagebox.showerror(
            title="خطأ" if is_arabic else "Error",
            message=f"خطأ في التجميع: {e}" if is_arabic else f"Clustering Error: {e}"
        )

def run_regression():
    global data_frame
    try:
        if data_frame is None:
            messagebox.showwarning(
                title="تحذير" if is_arabic else "Warning",
                message="لا توجد بيانات!" if is_arabic else "No data loaded!"
            )
            return
        
        x_col = regression_x_selector.get()
        y_col = regression_y_selector.get()
        
        if not x_col or not y_col:
            messagebox.showwarning(
                title="تحذير" if is_arabic else "Warning",
                message="يرجى اختيار الأعمدة!" if is_arabic else "Please select columns!"
            )
            return
        
        # تطبيق الانحدار الخطي
        X = data_frame[x_col].values.reshape(-1, 1)
        y = data_frame[y_col].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # عرض النتائج
        r_squared = model.score(X, y)
        equation = f"y = {model.coef_[0]:.2f}x + {model.intercept_:.2f}"
        
        result_display.config(text=f"{'معادلة الانحدار: ' if is_arabic else 'Regression equation: '}{equation}\n"
                                  f"{'R-squared: ' if not is_arabic else 'معامل التحديد: '}{r_squared:.4f}")
        
    except Exception as e:
        messagebox.showerror(
            title="خطأ" if is_arabic else "Error",
            message=f"خطأ في الانحدار: {e}" if is_arabic else f"Regression Error: {e}"
        )

def show_correlation():
    global data_frame
    try:
        if data_frame is None:
            messagebox.showwarning(
                title="تحذير" if is_arabic else "Warning",
                message="لا توجد بيانات!" if is_arabic else "No data loaded!"
            )
            return
        
        # حساب مصفوفة الارتباط
        corr = data_frame.corr()
        
        # عرض المصفوفة
        result_display.config(text=f"{'مصفوفة الارتباط:' if is_arabic else 'Correlation Matrix:'}\n{corr}")
        
        # رسم heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title("Correlation Matrix" if not is_arabic else "مصفوفة الارتباط")
        
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        messagebox.showerror(
            title="خطأ" if is_arabic else "Error",
            message=f"خطأ في حساب الارتباط: {e}" if is_arabic else f"Correlation Error: {e}"
        )

def show_advanced_stats():
    global data_frame
    try:
        if data_frame is None:
            messagebox.showwarning(
                title="تحذير" if is_arabic else "Warning",
                message="لا توجد بيانات!" if is_arabic else "No data loaded!"
            )
            return
        
        # إحصائيات متقدمة
        stats_text = ""
        for col in data_frame.select_dtypes(include=[np.number]).columns:
            stats_text += f"{col}:\n"
            stats_text += f"  - التباين: {data_frame[col].var():.2f}\n" if is_arabic else f"  - Variance: {data_frame[col].var():.2f}\n"
            stats_text += f"  - الانحراف المعياري: {data_frame[col].std():.2f}\n" if is_arabic else f"  - Std Deviation: {data_frame[col].std():.2f}\n"
            stats_text += f"  - المدى: {data_frame[col].max() - data_frame[col].min():.2f}\n" if is_arabic else f"  - Range: {data_frame[col].max() - data_frame[col].min():.2f}\n"
            stats_text += f"  - الالتواء: {data_frame[col].skew():.2f}\n" if is_arabic else f"  - Skewness: {data_frame[col].skew():.2f}\n"
            stats_text += f"  - التفرطح: {data_frame[col].kurtosis():.2f}\n\n" if is_arabic else f"  - Kurtosis: {data_frame[col].kurtosis():.2f}\n\n"
        
        result_display.config(text=stats_text)
        
    except Exception as e:
        messagebox.showerror(
            title="خطأ" if is_arabic else "Error",
            message=f"خطأ في الإحصائيات: {e}" if is_arabic else f"Stats Error: {e}"
        )

def show_probability_distributions():
    try:
        # إنشاء توزيعات احتمالية
        normal_dist = np.random.normal(0, 1, 1000)
        uniform_dist = np.random.uniform(0, 1, 1000)
        binomial_dist = np.random.binomial(10, 0.5, 1000)
        
        # رسم التوزيعات
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].hist(normal_dist, bins=30, alpha=0.7, color='blue')
        axes[0].set_title("Normal Distribution" if not is_arabic else "التوزيع الطبيعي")
        
        axes[1].hist(uniform_dist, bins=30, alpha=0.7, color='green')
        axes[1].set_title("Uniform Distribution" if not is_arabic else "التوزيع المنتظم")
        
        axes[2].hist(binomial_dist, bins=30, alpha=0.7, color='red')
        axes[2].set_title("Binomial Distribution" if not is_arabic else "توزيع ثنائي")
        
        plt.tight_layout()
        plt.show()
        
        result_display.config(text="تم عرض التوزيعات الاحتمالية" if is_arabic else "Probability distributions displayed")
        
    except Exception as e:
        messagebox.showerror(
            title="خطأ" if is_arabic else "Error",
            message=f"خطأ في التوزيعات: {e}" if is_arabic else f"Distribution Error: {e}"
        )

def display_library_commands(lib):
    frame = tabs[lib]
    for widget in frame.winfo_children():
        widget.destroy()
    
    title_text = f"📌 {lib} Commands" if not is_arabic else f"📌 أوامر {lib}"
    tk.Label(frame, text=title_text, font=("Arial", 18, "bold"), bg="white", fg="#2c3e50").pack(anchor="w", pady=5)
    
    search_var = tk.StringVar()
    search_entry = tk.Entry(frame, textvariable=search_var, font=("Arial", 14))
    search_entry.pack(fill="x", padx=10, pady=5)
    search_entry.insert(0, "")

    text_widget = tk.Text(frame, font=("Consolas", 14), bg="#f9f9f9", height=22)
    text_widget.pack(fill="both", expand=True, padx=10, pady=10)

    notes_widget = tk.Text(frame, font=("Arial", 12), bg="#e8f0fe", height=5)
    notes_widget.pack(fill="x", padx=10, pady=5)
    
    # تحميل الملاحظات المحفوظة
    if lib in user_notes:
        notes_widget.insert("1.0", user_notes[lib])
    else:
        notes_widget.insert("1.0", "💡 Notes: Add your personal notes here..." if not is_arabic else "💡 ملاحظات: أضف ملاحظاتك هنا...")

    def update_display(*args):
        search = search_var.get().lower()
        text_widget.delete("1.0", tk.END)
        for category, commands in LIBRARY_DATA[lib].items():
            for cmd in commands:
                if search in cmd.lower() or search in category.lower():
                    text_widget.insert(tk.END, f"{category} → {cmd}\n")

    def save_notes():
        user_notes[lib] = notes_widget.get("1.0", tk.END)
        messagebox.showinfo(
            title="تم الحفظ" if is_arabic else "Saved",
            message="تم حفظ الملاحظات!" if is_arabic else "Notes saved!"
        )

    search_var.trace_add("write", update_display)
    update_display()

    # زر لتشغيل الكود
    def run_code():
        code = text_widget.get("1.0", tk.END)
        try:
            buffer = io.StringIO()
            sys.stdout = buffer
            # تعريف بعض المتغيرات الافتراضية لتشغيل الأمثلة
            df = pd.DataFrame({"x":[1,2,3,4,5],"y":[10,20,25,30,40]})
            x = df["x"]
            y = df["y"]
            exec(code)
            sys.stdout = sys._stdout_
            messagebox.showinfo(
                title="الناتج" if is_arabic else "Output",
                message=buffer.getvalue()
            )
        except Exception as e:
            messagebox.showerror(
                title="خطأ" if is_arabic else "Error",
                message=str(e)
            )
    
    # زر لحفظ الملاحظات
    save_notes_btn = tk.Button(frame, text="💾 Save Notes" if not is_arabic else "💾 حفظ الملاحظات", 
                              font=("Arial", 12), bg="#4caf50", fg="white", command=save_notes)
    save_notes_btn.pack(side=tk.LEFT, padx=10, pady=5)
    
    run_btn = tk.Button(frame, text="▶ Run Selected Code" if not is_arabic else "▶ تشغيل الكود", 
                       font=("Arial", 12, "bold"), bg="#1abc9c", fg="white", command=run_code)
    run_btn.pack(side=tk.RIGHT, padx=10, pady=5)

def on_tab_change(event):
    selected = event.widget.tab(event.widget.index("current"), "text")
    if selected in ["المرجع التعليمي", "Learning Reference"]:
        display_library_commands("Pandas")
    elif selected in ["التعلم الآلي", "Machine Learning"]:
        update_column_selector()

def save_settings():
    try:
        user_settings["font_size"] = int(font_size_var.get())
        user_settings["auto_save"] = auto_save_var.get()
        user_settings["language"] = "arabic" if is_arabic else "english"
        user_settings["theme"] = "dark" if is_dark_mode else "light"
        
        with open("data_analysis_settings.json", "w") as f:
            json.dump(user_settings, f)
        
        messagebox.showinfo(
            title="تم الحفظ" if is_arabic else "Saved",
            message="تم حفظ الإعدادات!" if is_arabic else "Settings saved!"
        )
    except Exception as e:
        messagebox.showerror(
            title="خطأ" if is_arabic else "Error",
            message=f"خطأ في حفظ الإعدادات: {e}" if is_arabic else f"Error saving settings: {e}"
        )

def load_settings():
    global is_arabic, is_dark_mode
    try:
        if os.path.exists("data_analysis_settings.json"):
            with open("data_analysis_settings.json", "r") as f:
                loaded_settings = json.load(f)
                
            user_settings.update(loaded_settings)
            is_arabic = user_settings["language"] == "arabic"
            is_dark_mode = user_settings["theme"] == "dark"
            font_size_var.set(str(user_settings["font_size"]))
            auto_save_var.set(user_settings["auto_save"])
            
            update_language()
            apply_theme()
    except:
        pass

def reset_settings():
    global is_arabic, is_dark_mode
    is_arabic = True
    is_dark_mode = True
    font_size_var.set("12")
    auto_save_var.set(False)
    update_language()
    apply_theme()

# ------------------- واجهة المستخدم الرئيسية -------------------
root = tk.Tk()
root.title("نظام تحليل البيانات المتقدم")
root.geometry("1200x800")
root.configure(bg="#1e1e2f")

# إنشاء Notebook للتبويب
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=10, pady=10)
notebook.bind("<<NotebookTabChanged>>", on_tab_change)

# ------------------- تبويب تحليل البيانات -------------------
analysis_tab = ttk.Frame(notebook)
notebook.add(analysis_tab, text="تحليل البيانات")

title = tk.Label(analysis_tab, font=("Arial", 24, "bold"), bg="#1e1e2f", fg="white")
title.pack(pady=10)

lang_btn = tk.Button(analysis_tab, text="🔄 English", command=toggle_language, font=("Arial", 12, "bold"))
lang_btn.place(x=1050, y=20)

theme_btn = tk.Button(analysis_tab, text="🎨 تغيير السمة", command=toggle_theme, font=("Arial", 12, "bold"))
theme_btn.place(x=920, y=20)

input_frame = tk.Frame(analysis_tab, bg="#1e1e2f")
input_frame.pack(pady=10)

input_label = tk.Label(input_frame, font=("Arial", 14), bg="#1e1e2f", fg="white")
input_label.pack(anchor="w")

input_entry = tk.Entry(input_frame, width=70, font=("Arial", 14))
input_entry.pack(pady=5)

load_btn = tk.Button(input_frame, font=("Arial", 12), bg="#009688", fg="white", text="📂 رفع ملف", command=load_file)
load_btn.pack(pady=5)

option_label = tk.Label(analysis_tab, font=("Arial", 14), bg="#1e1e2f", fg="white")
option_label.pack()

selected_option = ttk.Combobox(analysis_tab, font=("Arial", 13), width=35, state="readonly")
selected_option.pack(pady=10)

analyze_btn = tk.Button(analysis_tab, font=("Arial", 13), bg="#3b8ad8", fg="white",
                        width=20, height=1, command=run_analysis)
analyze_btn.pack(pady=5)

button_frame = tk.Frame(analysis_tab, bg="#1e1e2f")
button_frame.pack(pady=5)

save_btn = tk.Button(button_frame, font=("Arial", 12), bg="#4caf50", fg="white", command=save_to_excel)
save_btn.grid(row=0, column=0, padx=10)

hist_btn = tk.Button(button_frame, font=("Arial", 12), bg="#9c27b0", fg="white", command=lambda: draw_plot("hist"))
hist_btn.grid(row=0, column=1, padx=10)

box_btn = tk.Button(button_frame, font=("Arial", 12), bg="#ff9800", fg="white", command=lambda: draw_plot("box"))
box_btn.grid(row=0, column=2, padx=10)

scatter_btn = tk.Button(button_frame, font=("Arial", 12), bg="#607d8b", fg="white", command=lambda: draw_plot("scatter"))
scatter_btn.grid(row=0, column=3, padx=10)

line_btn = tk.Button(button_frame, font=("Arial", 12), bg="#795548", fg="white", command=lambda: draw_plot("line"))
line_btn.grid(row=0, column=4, padx=10)

bar_btn = tk.Button(button_frame, font=("Arial", 12), bg="#009688", fg="white", command=lambda: draw_plot("bar"))
bar_btn.grid(row=0, column=5, padx=10)

outlier_btn = tk.Button(button_frame, font=("Arial", 12), bg="#f44336", fg="white", command=detect_outliers)
outlier_btn.grid(row=0, column=6, padx=10)

# إطار لعرض البيانات والنتائج
result_frame = tk.LabelFrame(analysis_tab, font=("Arial", 16, "bold"), bg="#2a2a40", fg="white", labelanchor="n")
result_frame.pack(fill="both", expand=True, padx=20, pady=10)

result_display = tk.Label(result_frame, font=("Courier", 14), wraplength=950,
                          justify="left", bg="#2a2a40", fg="white")
result_display.pack(padx=10, pady=10)

# جدول لعرض النتائج
result_table = ttk.Treeview(result_frame, columns=("Values", "Mean", "Median", "Std", "Max", "Min"), show="headings")
for col in result_table["columns"]:
    result_table.heading(col, text=col)
result_table.pack(fill="both", expand=True, padx=10, pady=10)

# ------------------- تبويب المرجع التعليمي -------------------
reference_tab = ttk.Frame(notebook)
notebook.add(reference_tab, text="المرجع التعليمي")

# إنشاء Notebook داخل التبويب للمكتبات
lib_notebook = ttk.Notebook(reference_tab)
lib_notebook.pack(fill="both", expand=True, padx=10, pady=10)

# إنشاء تبويبات لكل مكتبة
tabs = {}
for lib in LIBRARY_DATA.keys():
    frame = ttk.Frame(lib_notebook)
    lib_notebook.add(frame, text=lib)
    tabs[lib] = frame

# ------------------- تبويب التصور البياني -------------------
visualization_tab = ttk.Frame(notebook)
notebook.add(visualization_tab, text="التصور البياني")

# إطار لاختيار المحاور
axis_frame = tk.Frame(visualization_tab, bg="#1e1e2f")
axis_frame.pack(pady=10)

x_axis_label = tk.Label(axis_frame, text="المحور X:", font=("Arial", 12), bg="#1e1e2f", fg="white")
x_axis_label.grid(row=0, column=0, padx=5)

x_axis_selector = ttk.Combobox(axis_frame, width=20, state="readonly")
x_axis_selector.grid(row=0, column=1, padx=5)

y_axis_label = tk.Label(axis_frame, text="المحور Y:", font=("Arial", 12), bg="#1e1e2f", fg="white")
y_axis_label.grid(row=0, column=2, padx=5)

y_axis_selector = ttk.Combobox(axis_frame, width=20, state="readonly")
y_axis_selector.grid(row=0, column=3, padx=5)

# إطار للرسم البياني
plot_frame = tk.Frame(visualization_tab, bg="#1e1e2f")
plot_frame.pack(fill="both", expand=True, padx=20, pady=10)

# ------------------- تبويب التعلم الآلي -------------------
ml_tab = ttk.Frame(notebook)
notebook.add(ml_tab, text="التعلم الآلي")

ml_title = tk.Label(ml_tab, text="تقنيات التعلم الآلي", font=("Arial", 18, "bold"), bg="#1e1e2f", fg="white")
ml_title.pack(pady=10)

# التجميع
cluster_frame = tk.Frame(ml_tab, bg="#1e1e2f")
cluster_frame.pack(pady=10)

cluster_label = tk.Label(cluster_frame, text="تجميع K-means:", font=("Arial", 12), bg="#1e1e2f", fg="white")
cluster_label.grid(row=0, column=0, padx=5)

cluster_column_selector = ttk.Combobox(cluster_frame, width=20, state="readonly")
cluster_column_selector.grid(row=0, column=1, padx=5)

cluster_btn = tk.Button(cluster_frame, text="تجميع البيانات", font=("Arial", 12), bg="#3b8ad8", fg="white", command=run_clustering)
cluster_btn.grid(row=0, column=2, padx=5)

# الانحدار الخطي
regression_frame = tk.Frame(ml_tab, bg="#1e1e2f")
regression_frame.pack(pady=10)

regression_x_label = tk.Label(regression_frame, text="المتغير المستقل (X):", font=("Arial", 12), bg="#1e1e2f", fg="white")
regression_x_label.grid(row=0, column=0, padx=5)

regression_x_selector = ttk.Combobox(regression_frame, width=20, state="readonly")
regression_x_selector.grid(row=0, column=1, padx=5)

regression_y_label = tk.Label(regression_frame, text="المتغير التابع (Y):", font=("Arial", 12), bg="#1e1e2f", fg="white")
regression_y_label.grid(row=0, column=2, padx=5)

regression_y_selector = ttk.Combobox(regression_frame, width=20, state="readonly")
regression_y_selector.grid(row=0, column=3, padx=5)

regression_btn = tk.Button(regression_frame, text="انحدار خطي", font=("Arial", 12), bg="#3b8ad8", fg="white", command=run_regression)
regression_btn.grid(row=0, column=4, padx=5)

# ------------------- تبويب الإحصاءات -------------------
stats_tab = ttk.Frame(notebook)
notebook.add(stats_tab, text="الإحصاءات")

stats_title = tk.Label(stats_tab, text="الإحصائيات المتقدمة", font=("Arial", 18, "bold"), bg="#1e1e2f", fg="white")
stats_title.pack(pady=10)

stats_btn = tk.Button(stats_tab, text="إحصائيات متقدمة", font=("Arial", 12), bg="#3b8ad8", fg="white", command=show_advanced_stats)
stats_btn.pack(pady=5)

corr_btn = tk.Button(stats_tab, text="مصفوفة الارتباط", font=("Arial", 12), bg="#3b8ad8", fg="white", command=show_correlation)
corr_btn.pack(pady=5)

prob_btn = tk.Button(stats_tab, text="التوزيعات الاحتمالية", font=("Arial", 12), bg="#3b8ad8", fg="white", command=show_probability_distributions)
prob_btn.pack(pady=5)

# ------------------- تبويب الإعدادات -------------------
settings_tab = ttk.Frame(notebook)
notebook.add(settings_tab, text="الإعدادات")

settings_label = tk.Label(settings_tab, text="الإعدادات", font=("Arial", 18, "bold"), bg="#1e1e2f", fg="white")
settings_label.pack(pady=10)

# إعدادات الخط
font_frame = tk.Frame(settings_tab, bg="#1e1e2f")
font_frame.pack(pady=10)

font_label = tk.Label(font_frame, text="حجم الخط:", font=("Arial", 12), bg="#1e1e2f", fg="white")
font_label.grid(row=0, column=0, padx=5)

font_size_var = tk.StringVar(value="12")
font_size = ttk.Combobox(font_frame, textvariable=font_size_var, width=10, state="readonly")
font_size['values'] = ("10", "12", "14", "16", "18")
font_size.grid(row=0, column=1, padx=5)

# إعدادات أخرى
auto_save_frame = tk.Frame(settings_tab, bg="#1e1e2f")
auto_save_frame.pack(pady=10)

auto_save_label = tk.Label(auto_save_frame, text="الحفظ التلقائي:", font=("Arial", 12), bg="#1e1e2f", fg="white")
auto_save_label.grid(row=0, column=0, padx=5)

auto_save_var = tk.BooleanVar()
auto_save_check = tk.Checkbutton(auto_save_frame, variable=auto_save_var, bg="#1e1e2f")
auto_save_check.grid(row=0, column=1, padx=5)

# أزرار حفظ وإعادة التعيين
save_settings_btn = tk.Button(settings_tab, text="حفظ الإعدادات", font=("Arial", 12), bg="#4caf50", fg="white", command=save_settings)
save_settings_btn.pack(pady=10)

reset_settings_btn = tk.Button(settings_tab, text="إعادة التعيين", font=("Arial", 12), bg="#f44336", fg="white", command=reset_settings)
reset_settings_btn.pack(pady=5)

# ------------------- معاينة البيانات -------------------
preview_frame = tk.LabelFrame(analysis_tab, text="معاينة البيانات", font=("Arial", 14, "bold"), bg="#2a2a40", fg="white")
preview_frame.pack(fill="x", padx=20, pady=10)

preview_text = scrolledtext.ScrolledText(preview_frame, height=5, font=("Consolas", 10))
preview_text.pack(fill="x", padx=10, pady=10)

data_info_label = tk.Label(preview_frame, text="", font=("Arial", 10), bg="#2a2a40", fg="white", justify=tk.LEFT)
data_info_label.pack(padx=10, pady=5)

# ------------------- التهيئة النهائية -------------------
update_language()
apply_theme()
load_settings()
display_library_commands("Pandas")

root.mainloop()