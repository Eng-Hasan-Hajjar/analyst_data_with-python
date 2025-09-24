import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalysisFeatures:
    @staticmethod
    def detect_data_quality_issues(df: pd.DataFrame) -> list:
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
    def automated_data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
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
    def create_advanced_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str = None) -> plt.Figure:
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
    def time_series_analysis(df: pd.DataFrame, date_col: str, value_col: str) -> dict:
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
    def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
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