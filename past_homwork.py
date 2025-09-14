
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

# ------------------- الحالة -------------------
is_arabic = True
is_dark_mode = True
data_frame = None  # DataFrame loaded

# ------------------- الوظائف -------------------
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
            messagebox.showwarning("Warning", "No data provided!" if not is_arabic else "لا توجد بيانات!")
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

        result_display.config(text=result_text)
        populate_table(df)  # عرض البيانات في الجدول
    except Exception as e:
        result_display.config(text=f"خطأ: {e}" if is_arabic else f"Error: {e}")

def detect_outliers():
    global data_frame
    try:
        if data_frame is None:
            messagebox.showwarning("Warning", "No data loaded!" if not is_arabic else "لا توجد بيانات!")
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

def draw_hist():
    global data_frame
    try:
        if data_frame is None:
            arr = np.fromstring(input_entry.get(), sep=',')
        else:
            arr = data_frame['Values'].values
        fig, ax = plt.subplots(figsize=(6,4))
        ax.hist(arr, bins=10, color="#3b8ad8", edgecolor="black")
        ax.set_title("Histogram" if not is_arabic else "مخطط التوزيع")
        ax.set_xlabel("Values" if not is_arabic else "القيم")
        ax.set_ylabel("Frequency" if not is_arabic else "التكرار")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        result_display.config(text=f"خطأ في الرسم: {e}" if is_arabic else f"Plot Error: {e}")

def draw_boxplot():
    global data_frame
    try:
        if data_frame is None:
            arr = np.fromstring(input_entry.get(), sep=',')
        else:
            arr = data_frame['Values'].values
        fig, ax = plt.subplots(figsize=(6,4))
        ax.boxplot(arr, patch_artist=True, boxprops=dict(facecolor="#3b8ad8"))
        ax.set_title("Boxplot" if not is_arabic else "مخطط الصندوق")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        result_display.config(text=f"خطأ في الرسم: {e}" if is_arabic else f"Plot Error: {e}")

def save_to_excel():
    global data_frame
    if data_frame is None:
        messagebox.showwarning("Warning", "No data to save!" if not is_arabic else "لا توجد بيانات للحفظ!")
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
        messagebox.showinfo("Saved", "Excel file saved!" if not is_arabic else "تم حفظ الملف بنجاح!")

def load_file():
    global data_frame
    file = filedialog.askopenfilename(filetypes=[("Excel Files", ".xlsx"), ("CSV Files", ".csv")])
    if file:
        if file.endswith('.csv'):
            data_frame = pd.read_csv(file)
        else:
            data_frame = pd.read_excel(file)
        if 'Values' not in data_frame.columns:
            messagebox.showwarning("Warning", "File must have 'Values' column!" if not is_arabic else "الملف يجب أن يحتوي على عمود 'Values'")
            data_frame = None
            return
        populate_table(data_frame)
        messagebox.showinfo("Loaded", "Data loaded successfully!" if not is_arabic else "تم تحميل البيانات بنجاح!")

def populate_table(df):
    for row in result_table.get_children():
        result_table.delete(row)
    for index, row in df.iterrows():
        result_table.insert("", "end", values=list(row))

def toggle_language():
    global is_arabic
    is_arabic = not is_arabic
    update_language()

def update_language():
    if is_arabic:
        title.config(text="تحليل بيانات متقدم")
        input_label.config(text="🧮 أدخل مصفوفة (مثال: 1,2,3,4) أو ارفع ملف Excel/CSV:")
        option_label.config(text="📊 اختر نوع التحليل:")
        analyze_btn.config(text="🔍 تحليل الآن")
        save_btn.config(text="💾 حفظ Excel")
        graph_btn.config(text="📊 رسم Histogram")
        box_btn.config(text="📦 Boxplot")
        outlier_btn.config(text="🧠 القيم الشاذة")
        load_btn.config(text="📂 رفع ملف")
        lang_btn.config(text="🔄 English")
        selected_option['values'] = [
            "أماكن الرقم 4", "الأرقام الزوجية", "الأرقام الفردية",
            "فرز المصفوفة", "فلترة > 42", "فلترة زوجية",
            "عدد عشوائي 0-100", "5 أعداد صحيحة", "5 أعداد عشرية",
            "عنصر عشوائي من المصفوفة", "إحصائيات"
        ]
    else:
        title.config(text="Advanced Data Analysis")
        input_label.config(text="🧮 Enter array (e.g., 1,2,3,4) or load Excel/CSV:")
        option_label.config(text="📊 Select analysis type:")
        analyze_btn.config(text="🔍 Analyze")
        save_btn.config(text="💾 Save Excel")
        graph_btn.config(text="📊 Plot Histogram")
        box_btn.config(text="📦 Boxplot")
        outlier_btn.config(text="🧠 Outliers")
        load_btn.config(text="📂 Load File")
        lang_btn.config(text="🔄 العربية")
        selected_option['values'] = [
            "Where value is 4", "Even numbers", "Odd numbers",
            "Sort array", "Filter > 42", "Filter even",
            "Random int 0-100", "5 random integers", "5 random floats",
            "Random choice from array", "Statistics"
        ]

def toggle_theme():
    global is_dark_mode
    is_dark_mode = not is_dark_mode
    bg = "#1e1e2f" if is_dark_mode else "#f0f0f0"
    fg = "white" if is_dark_mode else "black"
    frame_bg = "#2a2a40" if is_dark_mode else "#ffffff"

    root.configure(bg=bg)
    title.config(bg=bg, fg=fg)
    input_frame.config(bg=bg)
    input_label.config(bg=bg, fg=fg)
    option_label.config(bg=bg, fg=fg)
    result_frame.config(bg=frame_bg, fg=fg)
    result_display.config(bg=frame_bg, fg=fg)
    button_frame.config(bg=bg)
    lang_btn.config(bg="#555" if is_dark_mode else "#ddd", fg=fg)
    theme_btn.config(bg="#444" if is_dark_mode else "#ccc", fg=fg)

    # تعديل ألوان Treeview باستخدام Style
    style = ttk.Style()
    style.theme_use('default')
    style.configure("Treeview",
                    background=frame_bg,
                    foreground=fg,
                    fieldbackground=frame_bg)
    style.map('Treeview', background=[('selected', '#6a6a9c')], foreground=[('selected', 'white')])

# ------------------- واجهة المستخدم -------------------
root = tk.Tk()
root.title("تحليل البيانات المتقدم")
root.geometry("1000x700")
root.configure(bg="#1e1e2f")

title = tk.Label(root, font=("Arial", 24, "bold"), bg="#1e1e2f", fg="white")
title.pack(pady=10)

lang_btn = tk.Button(root, text="🔄 English", command=toggle_language, font=("Arial", 12, "bold"))
lang_btn.place(x=880, y=20)

theme_btn = tk.Button(root, text="🎨 النمط", command=toggle_theme, font=("Arial", 12, "bold"))
theme_btn.place(x=780, y=20)

input_frame = tk.Frame(root, bg="#1e1e2f")
input_frame.pack(pady=10)

input_label = tk.Label(input_frame, font=("Arial", 14), bg="#1e1e2f", fg="white")
input_label.pack(anchor="w")

input_entry = tk.Entry(input_frame, width=70, font=("Arial", 14))
input_entry.pack(pady=5)

load_btn = tk.Button(input_frame, font=("Arial", 12), bg="#009688", fg="white", text="📂 رفع ملف", command=load_file)
load_btn.pack(pady=5)

option_label = tk.Label(root, font=("Arial", 14), bg="#1e1e2f", fg="white")
option_label.pack()

selected_option = ttk.Combobox(root, font=("Arial", 13), width=35, state="readonly")
selected_option.pack(pady=10)

analyze_btn = tk.Button(root, font=("Arial", 13), bg="#3b8ad8", fg="white",
                        width=20, height=1, command=run_analysis)
analyze_btn.pack(pady=5)

button_frame = tk.Frame(root, bg="#1e1e2f")
button_frame.pack(pady=5)

save_btn = tk.Button(button_frame, font=("Arial", 12), bg="#4caf50", fg="white", command=save_to_excel)
save_btn.grid(row=0, column=0, padx=10)

graph_btn = tk.Button(button_frame, font=("Arial", 12), bg="#9c27b0", fg="white", command=draw_hist)
graph_btn.grid(row=0, column=1, padx=10)

box_btn = tk.Button(button_frame, font=("Arial", 12), bg="#ff9800", fg="white", command=draw_boxplot)
box_btn.grid(row=0, column=2, padx=10)

outlier_btn = tk.Button(button_frame, font=("Arial", 12), bg="#f44336", fg="white", command=detect_outliers)
outlier_btn.grid(row=0, column=3, padx=10)

result_frame = tk.LabelFrame(root, font=("Arial", 16, "bold"), bg="#2a2a40", fg="white", labelanchor="n")
result_frame.pack(fill="both", expand=True, padx=20, pady=10)

result_display = tk.Label(result_frame, font=("Courier", 14), wraplength=950,
                          justify="left", bg="#2a2a40", fg="white")
result_display.pack(padx=10, pady=10)

# جدول لعرض النتائج
result_table = ttk.Treeview(result_frame, columns=("Values", "Mean", "Median", "Std", "Max", "Min"), show="headings")
for col in result_table["columns"]:
    result_table.heading(col, text=col)
result_table.pack(fill="both", expand=True, padx=10, pady=10)

update_language()
toggle_theme()
root.mainloop()