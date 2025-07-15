import numpy as np

print(np.__version__)


arr = np.array([1, 2, 3, 4])  #  أحادية البعد مصفوفة

print(arr)  # الناتج: [1 2 3 4]




arr_2d = np.array([[1, 2], [3, 4], [5, 6]])
print(arr_2d)  # الناتج:
# [[1 2]
# [3 4]
# [5 6]]



# إنشاء مصفوفة
a = np.array([1, 2, 3, 4])
print(a)

# مصفوفة ثنائية الأبعاد
b = np.array([[1, 2], [3, 4]])
print(b)

# مصفوفة أصفار وأحاد وعشوائية
zeros = np.zeros((2, 3))
ones = np.ones((2, 3))
random = np.random.rand(3, 3)
print(zeros,"\n",ones,"\n",random)
# العمليات الحسابية
c = np.array([10, 20, 30])
d = np.array([1, 2, 3])
print(c + d)     # جمع
print(c * d)     # ضرب
print(c / d)     # قسمة

# الإحصائيات
arr = np.array([10, 20, 30, 40, 50])
print(arr.mean())     # المتوسط
print(arr.std())      # الانحراف المعياري
print(arr.max())      # القيمة الكبرى



print(np.eye(5))
















