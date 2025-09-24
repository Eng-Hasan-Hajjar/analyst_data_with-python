import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from config.settings import COLORS, is_dark_mode, is_arabic
from utils.file_utils import load_file, update_recent_files_menu, export_dashboard
from utils.data_utils import automated_data_cleaning, advanced_statistical_analysis, correlation_analysis, time_series_analysis, machine_learning_predictions, pattern_detection, quick_analysis
from utils.ml_utils import machine_learning_predictions, run_advanced_ml
from src.features import DataAnalysisFeatures
from .components import create_enhanced_status_bar, show_dashboard, update_status, update_data_preview, show_quality_report
# ... (أي imports أخرى ضرورية)

def create_advanced_interface():
    global root, notebook, preview_text, result_display, table_frame
    
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
        ("🔍 فحص جودة البيانات", lambda: show_quality_report(DataAnalysisFeatures.detect_data_quality_issues(data_frame)) if data_frame is not None else messagebox.showwarning("تحذير", "لا توجد بيانات!")),
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

def create_card(parent, title: str) -> tk.Frame:
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

def create_visualization(viz_type: str):
    """إنشاء التصورات البيانية"""
    if data_frame is None:
        messagebox.showwarning("تحذير", "لا توجد بيانات للتصور!")
        return
    
    try:
        fig = DataAnalysisFeatures.create_advanced_chart(data_frame, viz_type, 
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

def run_statistical_test(test_type: str):
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
        ("🔍 فحص الجودة", lambda: show_quality_report(DataAnalysisFeatures.detect_data_quality_issues(data_frame)) if data_frame is not None else messagebox.showwarning("تحذير", "لا توجد بيانات!"))
    ]
    
    for i, (text, command) in enumerate(management_tools):
        row, col = i // 3, i % 3
        btn = tk.Button(management_frame, text=text, font=("Arial", 10), command=command)
        btn.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
    
    management_frame.grid_columnconfigure((0,1,2), weight=1)