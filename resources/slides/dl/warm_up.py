''' 1.导入numpy库 '''
import numpy as np

''' 
2.建立一个一维数组a 初始化为[4, 5, 6], (1) 输出a 的类型（type）
(2) 输出a的各维度的大小（shape）(3) 输出a的第一个元素（值为4）
'''
a = np.array([4, 5, 6])
type(a)
a.shape
a[0]

'''
3.建立一个二维数组b, 初始化为[[4, 5, 6], [1, 2, 3]](1)输出各维度的大小（shape）
(2)输出b(0, 0)，b(0, 1), b(1, 1)这三个元素（对应值分别为4, 5, 2）
'''
b = np.array([[4, 5, 6], [1, 2, 3]])
b.shape
b[0, 0]
b[0, 1]
b[1, 1]

'''
4.(1)建立一个全0矩阵a, 大小为3x3;类型为整型（提示: dtype = int）
(2)建立一个全1矩阵b, 大小为4x5;
(3)建立一个单位矩阵c, 大小为4x4;
(4)生成一个随机数矩阵d, 大小为3x2.
'''
a = np.zeros((3, 3), dtype=int)
b = np.ones((4, 5))
c = np.identity(4)
d = np.random.rand(3, 2)

'''
5.建立一个数组a, (值为[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]), (1)打印a;(2)输出下标为(2, 3), (0, 0)这两个数组元素的值
'''
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a)
a[2, 3]
a[0, 0]

'''
6.把上一题的a数组的0到1行2到3列，放到b里面去，（此处不需要从新建立a, 直接调用即可）(1), 输出b;(2)输出b的（0, 0）这个元素的值
'''
b = a[0:2, 2:4]
b[0, 0]

'''
7.把第5题中数组a的最后两行所有元素放到c中，(1)输出c;(2)输出c中第一行的最后一个元素（提示，使用 - 1表示最后一个元素）
'''
c = a[-2:]
c
c[0, -1]

'''
8.建立数组a, 初始化a为[[1, 2], [3, 4], [5, 6]]，输出 （0, 0）（1, 1）（2, 0）这三个元素（提示： 使用
print(a[[0, 1, 2], [0, 1, 0]]) ）
'''
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a[[0, 1, 2], [0, 1, 0]])

'''
9.建立矩阵a, 初始化为[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]，输出(0, 0), (1, 2), (2, 0), (3, 1)(提示使用
b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b]))
'''
a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
b = np.array([0, 2, 0, 1])
print(a[np.arange(4), b])

'''
10.对9中输出的那四个元素，每个都加上10，然后重新输出矩阵a.(提示： a[np.arange(4), b] += 10 ）
'''
a[np.arange(4), b] += 10
print(a)

'''
11.执行x = np.array([1, 2])，然后输出x的数据类型
'''
x = np.array([1, 2])
x.dtype

'''
12.执行x = np.array([1.0, 2.0]) ，然后输出x的数据类类型
'''
x = np.array([1.0, 2.0])
x.dtype

'''
13.执行x = np.array([[1, 2], [3, 4]], dtype=np.float64) ，y = np.array([[5, 6], [7, 8]], dtype=np.float64)，然后输出
x + y, 和np.add(x, y)
'''
x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)
print(x+y)
print(np.add(x, y))

'''
14.利用13题目中的x, y输出x - y和np.subtract(x, y)
'''
print(x-y)
print(np.subtract(x, y))

'''
15.利用13题目中的x，y输出x * y, 和np.multiply(x, y)还有np.dot(x, y), 比较差异。
'''
print(x*y)
print(np.multiply(x, y))
print(np.dot(x, y))

'''
16.利用13题目中的x, y, 输出x / y.(提示 ： 使用函数np.divide())
'''
print(np.divide(x, y))

'''
17.利用13题目中的x, 输出x的开方。(提示： 使用函数 np.sqrt() )
'''
print(np.sqrt(x))

'''
18.利用13题目中的x, y, 执行print(x.dot(y))和print(np.dot(x, y))
'''
print(x.dot(y))
print(np.dot(x, y))

'''
19.利用13题目中的x, 进行求和。提示：输出三种求和(1)print(np.sum(x)): 
(2)print(np.sum(x，axis = 0 )); (3)print(np.sum(x, axis=1))
'''
print(np.sum(x))
print(np.sum(x, axis=0))
print(np.sum(x, axis=1))

'''
20.利用13题目中的x, 进行求平均数（提示：输出三种平均数(1)
print(np.mean(x))(2)
print(np.mean(x, axis=0))(3)
print(np.mean(x, axis=1))）
'''
print(np.mean(x))(2)
print(np.mean(x, axis=0))(3)
print(np.mean(x, axis=1))

'''
21.利用13题目中的x，对x进行矩阵转置，然后输出转置后的结果，（提示： x.T表示对x的转置）
'''
print(x.T)

'''
22.利用13题目中的x, 求e的指数（提示： 函数np.exp()）
'''
print(np.exp(x))

'''
23.利用13题目中的x, 求值最大的下标（提示(1)print(np.argmax(x)), (2)print(np.argmax(x, axis=0))(3)print(np.argmax(x), axis=1))
'''
print(np.argmax(x))
print(np.argmax(x, axis=0))
print(np.argmax(x), axis=1)

'''
24, 画图，y = x * x其中x = np.arange(0, 100, 0.1) （提示这里用到matplotlib.pyplot库）
'''
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 100, 0.1)
y = x * x
plt.plot(x, y)
plt.xlabel('x label')
plt.ylabel('y label')
plt.show()

'''
25.画图。画正弦函数和余弦函数， x = np.arange(0, 3 * np.pi, 0.1)(提示：这里用到
np.sin()np.cos()函数和matplotlib.pyplot库)
'''
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 100, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, label = 'linear')
plt.plot(x, y2, label = 'cubic')
plt.xlabel('x label')
plt.ylabel('y label')
plt.legend(loc=1)
plt.show()