# Program Name: NSGA-II.py
# Description: This is a python implementation of Prof. Kalyanmoy Deb's popular NSGA-II algorithm
# Author: Haris Ali Khan 
# Supervisor: Prof. Manoj Kumar Tiwari

#Importing required modules
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 函数 利用图片的x y坐标来操作
def function1(template,target,x,y):
    funzi = 0
    funmu1 = 0
    funmu2 = 0

    theight, twidth = template.shape[:2]
    h,w = target.shape[:2]
    # print(theight)
    # print(twidth)
    for i in range(0,theight-1): # 高度
        if y + i >= h:
            if funmu1 == 0: # 除0异常
                return 1
            return funzi / ((funmu1 * funmu2) ** 0.5)
        for j in range(0,twidth-1):
            a = template[i,j]
            if x+j >= w :
                break
            b = target[y+i,x+j]
            funzi = funzi + a * b
            funmu1 = funmu1 + a*a
            funmu2 = funmu2 + b*b
    if funmu1 == 0:
        return 1
    res = funzi/((funmu1*funmu2) ** 0.5)
    return res

def function2(template,target,x,y):
    funzi = 0
    funmu1 = 0
    funmu2 = 0
    theight, twidth = template.shape[:2]
    h,w = target.shape[:2]
    for i in range(0,theight-1): # 高度
        if y + i >= h:
            if funmu1 == 0:
                return 1
            return funzi / ((funmu1 * funmu2) ** 0.5)
        for j in range(0,twidth-1):
            a = template[i,j]
            if x+j >= w :
                break
            b = target[y+i,x+j]
            funzi = funzi + a * b
            funmu1 = funmu1 + a*a
            funmu2 = funmu2 + b*b
    if funmu1 == 0:
        return 1
    res = (funzi-2)/((funmu1*funmu2) ** 0.5)
    return res

# ---------------------------------------------------------
# 工具函数
#Function to find index of list  函数查找列表的索引
def index_of(a,list):
    for i in range(0,len(list)):
        if list[i] == a:
            return i
    return -1

#Function to sort by values   函数按值排序
# 快速排序
def quick_sort(res ,nums, left, right):
    if left < right:
        i = left
        j = right
        privot2 = res[left]
        privot = nums[left]
        while i!=j:
            while j>i and nums[j] > privot:
                j -= 1
            if j>i:
                nums[i] = nums[j]
                res[i] = res[j]
                i += 1
            while i < j and nums[i] < privot:
                    i = i + 1
            if i < j:
                    nums[j] = nums[i]
                    res[j] = res[i]
                    j = j - 1
        nums[i] = privot
        res[i] = privot2
        # 以i为枢轴进行子序列元素交换
        quick_sort(res, nums, left, i - 1)
        quick_sort(res, nums, i + 1, right)
def sort_by_values(list1, values):
    # sort_by_values(front, values1[:])
    # list1 是索引值 values是function1_values[:] 也就是函数值
    # sorted_list = []
    front_values = [values[i] for i in list1]
    # 遍历所有数组元素
    # 冒泡排序 根据front_values的大小对list1j进行排序
    quick_sort(list1,front_values,0,len(list1)-1)
    # for i in range(len(front_values)):
    #     # Last i elements are already in place
    #     for j in range(0, len(front_values) - i - 1):
    #         if front_values[j] < front_values[j + 1]:
    #             front_values[j], front_values[j + 1] = front_values[j + 1], front_values[j]
    #             list1[j], list1[j + 1] = list1[j + 1], list1[j]
    # while(len(sorted_list)!=len(list1)):
    #     # 如果这个函数值的最小值的索引是在索引数组里 那我们就把这个索引加上
    #     if index_of(min(values),values) in list1:
    #         sorted_list.append(index_of(min(values),values))
    #     # 让当前最小值变为变为最大值
    #     values[index_of(min(values),values)] = math.inf # 浮点正无穷大
    return list1
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

    sorted1 = sort_by_values(front, values1[:])  # 这里卡住了 为什么？
    sorted2 = sort_by_values(front, values2[:])

    # 下面的代码我看的懂
    distance[0] = 444444444
    distance[len(front) - 1] = 444444444
    max_1 = max(values1)
    min_1 = min(values1)
    max_2 = max(values2)
    min_2 = min(values2)
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted1[k+1]] - values2[sorted1[k-1]])/(max_1-min_1)
    for k in range(1,len(front)-1):
        distance[k] = distance[k]+ (values1[sorted2[k+1]] - values2[sorted2[k-1]])/(max_2-min_2)
    return distance

def crowding_distance_liduoan(values1, values2, front):
    #     nLen = len(I)  # I中的个体数量
    #
    # for i in I:
    #     i.distance = 0  # 初始化所有个体的拥挤距离
    # for objFun in M:  # M为所有目标函数的列表
    #     I = sort(I, objFun)  # 按照目标函数objFun进行升序排序
    #     I[0] = I[len[I] - 1] = ∞  # 对第一个和最后一个个体的距离设为无穷大
    #     for i in xrange(1, len(I) - 2):
    #         I[i].distance = I[i].distance + (objFun(I[i + 1]) - objFun(I[i - 1])) / (Max(objFun()) - Min(objFun()))
    # front_values1 = [values1[front[i]] for i in range(0, len(front))]
    # front_values2 = [values2[front[i]] for i in range(0, len(front))]
    # sorted1 = inplace_quick_sort(front_values1,front,0,len(front_values1)) # 这里卡住了 为什么？
    # sorted2 = inplace_quick_sort(front_values2,front,0,len(front_values1))
    # distance = [0 for i in range(0, len(front))]
    # # 下面的代码我看的懂
    # # 下面的代码我看的懂
    # distance[0] = 4444444
    # distance[len(front) - 1] = 444444
    # max_1 = max(values1)
    # min_1 = min(values1)
    # for k in range(1, len(front) - 1):
    #     distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (max_1 - min_1)
    # max_2 = max(values2)
    # min_2 = min(values2)
    # for k in range(1, len(front) - 1):
    #     distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max_2 - min_2)
    # return distance

    # values1.sort()
    # values2.sort()
    # front_1_so = [0 for i in range(0,len(front))]
    # front_2_so = [0 for i in range(0, len(front))]
    # 找到 front_values1 在values的索引
    # fv1 = 0
    # fv2 = 0
    # for i in range(0,values1):
    #     if front_values1[fv1] == values1[i]:
    #         front_1_so.append(i)
    #         fv1 = fv1 + 1
    #     if front_values2[fv2] == values2[i]:
    #         front_2_so.append(i)
    #         fv2 = fv2 + 1
    pass
# ------------------------------------------------------------------------------------------------------------
# 编码 形式为 xxxyyy 共六位
# 返回值  有两个字符串 分别是 a坐标的 编码  b坐标的编码
def coding(a,b):
    str1 = ''
    str2 = ''
    for i in range(0,2):
        if a[i] < 10:
            str1 = str1 + "00" + str(a[i])
        elif a[i] < 100:
            str1 = str1 + "0" + str(a[i])
        else :
            str1 = str1 + str(a[i])

        if b[i] < 10:
            str2 = str2 + "00" + str(b[i])
        elif b[i] < 100:
            str2 = str2 + "0" + str(b[i])
        else :
            str2 = str2 + str(b[i])
    return str1 , str2
# 交叉变异操作 返回的是数组
def crossover(var1,var2):
    # 编码
    str1,str2 = coding(var1,var2)
    str1 = list(map(int, str1))
    str2 = list(map(int, str2))
    son = cross(str1,str2)
    # 交叉
    x = 0
    y = 0
    # print("son:" ,son)
    # if len(son) < 6:
    #     return str2
    for i in range(0,3):
        x = x + (10**(2-i))*son[i]
        y = y + (10 ** (2 - i)) * son[i+3]
    # for i in range(3,6):
    #     y = y + (10**(2-i))*son[i]
    return [x,y]
# 变异操作 返回的是字符串
def cross(parent1, parent2):
    # parent1 = list(map(int, parent1))
    # parent2 = list(map(int, parent2))
    # all = len(parent2)
    index1 = np.random.randint(0,high=2)
    index2 = np.random.randint(index1,high=6)
    tempGene = parent2[index1:index2]  # 交叉的基因片段
    newGene = [parent1[i] for i in range(0,index1)]
    for i in range(0,len(tempGene)):
        newGene.append(tempGene[i])
    for i in range(index2,len(parent2)):
        newGene.append(parent1[i])
    # 变异
    r = random.random()
    if r > 0.5:
        temp = np.random.randint(1,high=6)
        for i in range(0,temp):
            key = np.random.randint(0,high=6)
            value = np.random.randint(0,high=10)
            # print("key:",key)
            newGene[key] = value
    # newGene = [parent1[0,index1],tempGene,parent1[index2,len(parent2)]]
    # for g in parent1:
    #     if len == index1:
    #         newGene.extend(tempGene)  # 插入基因片段
    #         len += 1
    #     if g not in tempGene:
    #         newGene.append(g)
    #         len += 1
    return newGene

#Function to carry out the crossover  函数进行交叉
def crossoverH(a,b):
    r=random.random()
    if r>0.5:
        return mutationH((a+b)/2)
    else:
        return mutationH((a-b)/2)

#Function to carry out the mutation operator  函数执行变异算子
def mutationH(solution):
    mutation_prob = random.random()
    if mutation_prob <1:
        solution = int(min_x + (solution) * random.random())
        # solution = int(min_x+(theight-min_x)*random.random())
    return solution

#Function to carry out the crossover  函数进行交叉
def crossoverW(a,b):
    r=random.random()
    if r>0.5:
        return mutationW((a+b)/2)
    else:
        return mutationW((a-b)/2)

#Function to carry out the mutation operator  函数执行变异算子
def mutationW(solution):
    mutation_prob = random.random()
    if mutation_prob <1:
        solution = int(min_x + (solution) * random.random())
        # solution = int(min_x+(twidth-min_x)*random.random())
    return solution
# ------------------------------------------------------------------------------------------------------------

#Main program starts here 主程序从这里开始
pop_size = 20
max_gen = 921   # 最大迭代次数

#读取目标图片
# target = cv2.imread("target.jpg")
target = cv2.imread("stack.jpg")
target_copy = target.copy()
# cv2.imshow("target",target)
#读取模板图片
# template = cv2.imread("templete.jpg")
template = cv2.imread("templete2.jpg")
# 灰度化
target = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)


#Initialization 初始化
min_x=1
# 控制我们检索的范围
theight1, twidth1 = target.shape[:2]
theight2, twidth2 = template.shape[:2]

# theight = theight1 - theight2 - 1
# twidth = twidth1 - twidth2 - 1
theight = int(theight1*4/5)
twidth = int(twidth1*4/5)

#solution=[[min_x+(twidth-min_x)*random.randint(),min_x+(theight-min_x)*random.randint()] for i in range(0,pop_size)]
solution=[[random.randint(min_x,twidth),random.randint(min_x,theight)] for i in range(0,pop_size)]
# print(solution[1][1])
# 上面是初始种群

gen_no=0    # 迭代次数
while(gen_no<max_gen):
    # x的取值是唯一的 在后面我们放入了两个函数中 这就是多目标？ 一个变量 两个函数。。。。
    # function1_values = [function1(template,target,solution[i][0],solution[i][1])for i in range(0,pop_size)]
    # function2_values = [function2(template,target,solution[i][1],solution[i][0])for i in range(0,pop_size)]
    # function1_values = np.array(function1_values)
    # function2_values = np.array(function2_values)
    # # # 这里会出现nan值 我们强制把他赋予0
    # fun1TF = np.isnan(function1_values)
    # fun2TF = np.isnan(function2_values)
    # for i in range(0,pop_size):
    #     if fun1TF[i] == True:
    #         function1_values[i] = 1
    #     if fun2TF[i] == True:
    #         function2_values[i] = 1
    # # 得到对应的函数值
    # # 第一次快速非支配排序 得到的是list  对应参数是
    # non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:],function2_values[:])
    # 子代
    solution2 = solution[:]
    #Generating offsprings  产生后代
    # 也就是产生了子代
    while(len(solution2)!=2*pop_size):
        # 选择个体适应度高的
        # size_non_1 = len(non_dominated_sorted_solution[0])
        # # 在第一序列中随机抽取
        # a1 = random.randint(0,size_non_1-1)
        # b1 = random.randint(0,size_non_1-1)
        # # 得到抽取的对应坐标
        # a1 = non_dominated_sorted_solution[0][a1]
        # b1 = non_dominated_sorted_solution[0][b1]
        # # 它是从父亲那里继承过来的 不是随机数
        # solution2.append(crossover(solution[a1],solution[b1]))
        a1 = random.randint(0,pop_size-1)
        b1 = random.randint(0,pop_size-1)
        solution2.append(crossover(solution[a1], solution[b1]))
        # solution2.append([crossoverW(solution[a1][0],solution[b1][0]),crossoverH(solution[a1][1],solution[b1][1])])

    # 利用子代获得总种群
    function1_values2 = [function1(template,target,solution2[i][0],solution2[i][1])for i in range(0,2*pop_size)]
    function2_values2 = [function2(template,target,solution2[i][1],solution2[i][0])for i in range(0,2*pop_size)]
    function1_values2 = np.array(function1_values2)
    function2_values2 = np.array(function2_values2)
    # # 这里会出现nan值 我们强制把他赋予0
    fun1TF = np.isnan(function1_values2)
    fun2TF = np.isnan(function2_values2)
    for i in range(0,pop_size):
        if fun1TF[i] == True:
            function1_values2[i] = 1
        if fun2TF[i] == True:
            function2_values2[i] = 1



    # 父代子代种群合并进行快速非支配排序
    # 函数来执行NSGA-II的快速非支配排序
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:],function2_values2[:])


    crowding_distance_values2 = []
    for i in range(0, len(non_dominated_sorted_solution2)):
        crowding_distance_values2.append(crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))
        if i > 2:
            break
    # 新的种群 更具非支配关系和个体拥挤度 选取合适个体组成新的父代种群
    new_solution= []
    for i in range(0,len(non_dominated_sorted_solution2)):
        # 遍历第二次快速非支配排序 也就是对快速非支配排序得到的每一层进行遍历
        # 第二次快速非支配排序得到的是二维矩阵 这个是
        non_dominated_sorted_solution2_1 = [
            index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
            range(0, len(non_dominated_sorted_solution2[i]))]

        front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])

        front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                 range(0, len(non_dominated_sorted_solution2[i]))]

        front.reverse()

        for value in front:
            new_solution.append(value)
            if (len(new_solution) == pop_size):
                break
        if (len(new_solution) == pop_size):
            break

    solution = [solution2[i] for i in new_solution]
    # 把新的种群当作父代 进行循环迭代

    # ------------------------------------------------------------------------------------------------------
    print("The best front for Generation number ",gen_no, " is")
    for valuez in solution:
        print(valuez)
        # 有一个很基础的知识点 x坐标是横向的 也就是宽
        # cv2.rectangle(target_copy, (solution[valuez][0],solution[valuez][1]), (solution[valuez][0]+twidth2, solution[valuez][1]+theight2), (0, 0, 255), 2)
        # print(round(solution[valuez],3),end=" ")
    print("\n")


    if gen_no == 10:
        valuez = non_dominated_sorted_solution2[0][0]
        for value in non_dominated_sorted_solution2[0]:
            # 有一个很基础的知识点 x坐标是横向的 也就是宽
            if function1_values2[value] > function1_values2[valuez]:
                # 这里做个优化 选择最大的值
                valuez = value
                # cv2.rectangle(target_copy, (solution2[valuez][0], solution2[valuez][1]),(solution2[valuez][0] + twidth2, solution2[valuez][1] + theight2), (0, 0, 255), 2)
                # break
            # print(round(solution[valuez],3),end=" ")
        cv2.rectangle(target_copy, (solution2[valuez][0], solution2[valuez][1]),(solution2[valuez][0] + twidth2, solution2[valuez][1] + theight2), (0, 0, 255), 2)
        print(solution2[valuez])
        print("\n")
        cv2.imshow("Gray Image", target_copy)
        cv2.waitKey(0)
        print(10)
    # --------------------------------------------------------------------------------------------------------
    gen_no = gen_no + 1

