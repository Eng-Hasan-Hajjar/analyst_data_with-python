import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys

# -------------------------
# إعداد النافذة الرئيسية
# -------------------------
root = tk.Tk()
root.title("📊 دليل كامل لمكتبات تحليل البيانات")
root.geometry("1000x650")
root.config(bg="#f4f6f7")

title = tk.Label(root, text="📘 مرجع مكتبات تحليل البيانات", 
                 font=("Arial", 24, "bold"), bg="#2c3e50", fg="white", pady=15)
title.pack(fill="x")

# -------------------------
# Notebook Tabs
# -------------------------
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True, padx=10, pady=10)

# -------------------------
# إنشاء Tabs لكل مكتبة
# -------------------------
tabs = {}
libraries = ["Pandas", "Matplotlib", "Seaborn"]
for lib in libraries:
    frame = tk.Frame(notebook, bg="white")
    notebook.add(frame, text=lib)
    tabs[lib] = frame

# -------------------------
# بيانات لكل مكتبة
# -------------------------
pandas_commands = {
    "Basic Info": ["df.head()", "df.tail()", "df.shape", "df.columns", "df.dtypes", "df.info()"],
    "Statistics": ["df.describe()", "df.mean()", "df.median()", "df.mode()", "df.std()", "df.sum()"],
    "Data Cleaning": ["df.dropna()", "df.fillna(value)", "df.drop_duplicates()"],
    "Filtering": ["df[df['column'] > value]", "df[df['column'] == value]"],
    "Sorting": ["df.sort_values(by='column')", "df.sort_index()"],
    "Date & Time": ["pd.to_datetime(df['column'])"],
}

matplotlib_commands = {
    "Line Plots": ["plt.plot(x, y)", "plt.title('Title')", "plt.xlabel('X')", "plt.ylabel('Y')", 
                   "plt.grid(True)", "plt.legend()", "plt.show()"],
    "Scatter Plots": ["plt.scatter(x, y)", "plt.scatter(x, y, marker='o', color='r')"],
    "Bar Plots": ["plt.bar(x, y)", "plt.barh(x, y)"],
    "Customization": ["plt.plot(x,y, linestyle='--', linewidth=2, color='blue', marker='o')", 
                      "plt.xlim()", "plt.ylim()"]
}

seaborn_commands = {
    "Line Plot": ["sns.lineplot(x='x', y='y', data=df)"],
    "Scatter Plot": ["sns.scatterplot(x='x', y='y', data=df)"],
    "Bar Plot": ["sns.barplot(x='x', y='y', data=df)"],
    "Box Plot": ["sns.boxplot(x='x', y='y', data=df)"],
    "Heatmap": ["sns.heatmap(df.corr(), annot=True)"],
    "Histogram": ["sns.histplot(df['column'], bins=10)"],
    "Customization": ["sns.set_style('darkgrid')", "sns.set_palette('pastel')"]
}

library_data = {
    "Pandas": pandas_commands,
    "Matplotlib": matplotlib_commands,
    "Seaborn": seaborn_commands
}

# -------------------------
# دالة عرض الأوامر داخل Tab
# -------------------------
def display_commands(lib):
    frame = tabs[lib]
    for widget in frame.winfo_children():
        widget.destroy()
    
    tk.Label(frame, text=f"📌 {lib} Commands", font=("Arial", 18, "bold"), bg="white", fg="#2c3e50").pack(anchor="w", pady=5)
    
    search_var = tk.StringVar()
    search_entry = tk.Entry(frame, textvariable=search_var, font=("Arial", 14))
    search_entry.pack(fill="x", padx=10, pady=5)
    search_entry.insert(0, "")

    text_widget = tk.Text(frame, font=("Consolas", 14), bg="#f9f9f9", height=22)
    text_widget.pack(fill="both", expand=True, padx=10, pady=10)

    notes_widget = tk.Text(frame, font=("Arial", 12), bg="#e8f0fe", height=5)
    notes_widget.pack(fill="x", padx=10, pady=5)
    notes_widget.insert("1.0", "💡 Notes: Add your personal notes here...")

    def update_display(*args):
        search = search_var.get().lower()
        text_widget.delete("1.0", tk.END)
        for category, commands in library_data[lib].items():
            for cmd in commands:
                if search in cmd.lower() or search in category.lower():
                    text_widget.insert(tk.END, f"{category} → {cmd}\n")

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
            messagebox.showinfo("Output", buffer.getvalue())
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    run_btn = tk.Button(frame, text="▶ Run Selected Code", font=("Arial", 14, "bold"), bg="#1abc9c", fg="white", command=run_code)
    run_btn.pack(pady=5)

# -------------------------
# عند الضغط على Tab
# -------------------------
def on_tab_change(event):
    selected = event.widget.tab(event.widget.index("current"), "text")
    display_commands(selected)

notebook.bind("<<NotebookTabChanged>>", on_tab_change)
display_commands("Pandas")  # عرض افتراضي عند البداية

root.mainloop()