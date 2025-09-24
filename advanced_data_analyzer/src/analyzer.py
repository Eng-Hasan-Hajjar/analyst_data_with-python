import platform

class AdvancedDataAnalyzer:
    def __init__(self):
        self.version = "3.0.0"
        self.author = "نظام تحليل البيانات المتقدم"
        self.support_email = "support@dataanalysis.com"
        
    def get_system_info(self) -> dict:
        return {
            "platform": platform.system(),
            "version": platform.version(),
            "python_version": platform.python_version(),
            "processor": platform.processor()
        }