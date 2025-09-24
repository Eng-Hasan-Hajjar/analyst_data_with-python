import pytest
import pandas as pd
from src.features import DataAnalysisFeatures

def test_detect_data_quality_issues():
    df = pd.DataFrame({'a': [1, np.nan, 1], 'b': [2, 2, 3]})
    issues = DataAnalysisFeatures.detect_data_quality_issues(df)
    assert len(issues) > 0  # يجب أن يكتشف مفقودات ومكررات

# أضف اختبارات أخرى