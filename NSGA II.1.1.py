# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari

#Importing required modules
import math
import random
import matplotlib.pyplot as plt

#First function to optimize  第一个优化函数
def function1(x):
    value = -x**2
    return value

#Second function to optimize  第二个优化函数
def function2(x):
    value = -(x-2)**2
    return value


# ---------------------------------------------------------
# 工具函数
#Function to find index of list  函数查找列表的索引
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values   函数按值排序
def sort_by_values(list1, values):
    # sort_by_values(front, values1[:])
    # list1 是索引值 values是function1_values[:] 也就是函数值
    sorted_list = []
    while(len(sorted_list)!=len(list1)):
        # 如果这个函数值的最小值的索引是在索引数组里 那我们就把这个索引加上
        if index_of(min(values),values) in list1:
            sorted_list.append(index_of(min(values),values))
        # 让当前最小值变为变为最大值
        values[index_of(min(values),values)] = math.inf # 浮点正无穷大
    return sorted_list
# ---------------------------------------------------------



#Function to carry out NSGA-II's fast non dominated sort  函数来执行NSGA-II的快速非支配排序
def fast_non_dominated_sort(values1, values2):
# fast_non_dominated_sort(function1_values[:],function2_values[:])
    S=[[] for i in range(0,len(values1))]
    front = [[]]
    n=[0 for i in range(0,len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0,len(values1)):
        S[p]=[]
        n[p]=0
        for q in range(0, len(values1)):
            if (values1[p] > values1[q] and values2[p] > values2[q]) or (values1[p] >= values1[q] and values2[p] > values2[q]) or (values1[p] > values1[q] and values2[p] >= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] > values1[p] and values2[q] > values2[p]) or (values1[q] >= values1[p] and values2[q] > values2[p]) or (values1[q] > values1[p] and values2[q] >= values2[p]):
                n[p] = n[p] + 1
        if n[p]==0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while(front[i] != []):
        # 当我们循环得到的Q仍然是空集合时，我们就退出
        Q=[]
        for p in front[i]:
            for q in S[p]:
                n[q] =n[q] - 1
                if( n[q]==0):
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i = i+1
        front.append(Q)
    # del可以删除列表中指定位置的元素
    # 删除最后一个空集合 返回的是
    del front[len(front)-1]
    # 返回的到底是什么 这个我依旧不太清楚
    # 单纯从输出看得出 每一层的数据是父代种群中的值的索引
    return front

#Function to calculate crowding distance 函数来计算拥挤距离
def crowding_distance(values1, values2, front):
    # 第一个函数 第二个函数 每一层的快速非支配
    # crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]
    distance = [0 for i in range(0,len(front))]

    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])

    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max(values1)-min(values1))
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max(values2)-min(values2))
    return distance

#Function to carry out the crossover  函数进行交叉
def crossover(a,b):
    r=random.random()
    if r>0.5:
        return mutation((a+b)/2)
    else:
        return mutation((a-b)/2)

#Function to carry out the mutation operator  函数执行变异算子
def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob <1:
        solution = min_x+(max_x-min_x)*random.random()
    return solution

#Main program starts here 主程序从这里开始
pop_size = 20
max_gen = 921   # 最大迭代次数

#Initialization 初始化
min_x=-55
max_x=55
solution=[min_x+(max_x-min_x)*random.random() for i in range(0,pop_size)]

# 上面是初始种群

gen_no=0    # 迭代次数
while(gen_no<max_gen):
    # x的取值是唯一的 在后面我们放入了两个函数中 这就是多目标？ 一个变量 两个函数。。。。
    function1_values = [function1(solution[i])for i in range(0,pop_size)]
    function2_values = [function2(solution[i])for i in range(0,pop_size)]
    # 得到对应的函数值
    # 第一次快速非支配排序 得到的是list  对应参数是
    non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])


    print("The best front for Generation number ",gen_no, " is")
    for valuez in non_dominated_sorted_solution[0]:
        print(round(solution[valuez],3),end=" ")
    print("\n")

    # 第一次拥挤距离值
    crowding_distance_values=[]
    for i in range(0,len(non_dominated_sorted_solution)):
        crowding_distance_values.append(crowding_distance(function1_values[:],function2_values[:],non_dominated_sorted_solution[i][:]))
    # 得到拥挤度可以干嘛？
    # 说实话 我没看出来这次的拥挤度计算有什么用

    # 子代
    solution2 = solution[:]
    #Generating offsprings  产生后代
    # 也就是产生了子代
    while(len(solution2)!=2*pop_size):
        a1 = random.randint(0,pop_size-1)
        b1 = random.randint(0,pop_size-1)
        # 它是从父亲那里继承过来的 不是随机数
        solution2.append(crossover(solution[a1],solution[b1]))

    # 利用子代获得总种群
    function1_values2 = [function1(solution2[i])for i in range(0,2*pop_size)]
    function2_values2 = [function2(solution2[i])for i in range(0,2*pop_size)]

    # 父代子代种群合并进行快速非支配排序
    # 函数来执行NSGA-II的快速非支配排序
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])

    # 对每个非支配层 进行拥挤度计算
    crowding_distance_values2=[]
    for i in range(0,len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:],function2_values2[:],non_dominated_sorted_solution2[i][:]))

    # 新的种群 更具非支配关系和个体拥挤度 选取合适个体组成新的父代种群
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        # 遍历第二次快速非支配排序 也就是对快速非支配排序得到的每一层进行遍历
        # 第二次快速非支配排序得到的是二维矩阵 这个是
        non_dominated_sorted_solution2_1 = [index_of(non_dominated_sorted_solution2[i][j],non_dominated_sorted_solution2[i] ) for j in range(0,len(non_dominated_sorted_solution2[i]))]

        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])

        front = [non_dominated_sorted_solution2[i][front22[j]] for j in range(0,len(non_dominated_sorted_solution2[i]))]

        front.reverse()

        for value in front:
            new_solution.append(value)
            if(len(new_solution)==pop_size):
                break
        if (len(new_solution) == pop_size):
            break

    solution = [solution2[i] for i in new_solution]
    # 把新的种群当作父代 进行循环迭代
    gen_no = gen_no + 1



#Lets plot the final front now  现在让我们画出最后的前线
function1 = [i * -1 for i in function1_values]
function2 = [j * -1 for j in function2_values]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()
