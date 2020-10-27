#opencv模板匹配----单目标匹配
import cv2
import math
def function1(template,target,x,y):
    funzi = 0
    funmu1 = 0
    funmu2 = 0

    theight, twidth = template.shape[:2]
    print(theight)
    print(twidth)
    for i in range(0,theight-1): # 高度
        for j in range(0,twidth-1):
            a = template[i,j]
            b = target[x+i,y+j]
            # funzi = funzi + template[j,i] * target[y+j,x+i]
            # funmu1 = funmu1 + template[j,i]*template[j,i]
            # funmu2 = funmu2 + target[y+j,x+i]*target[y+j,x+i]
            funzi = funzi + a * b
            funmu1 = funmu1 + a*a
            funmu2 = funmu2 + b*b
    res = 0
    res = funzi/math.sqrt(funmu1*funmu2)
    return res
#读取目标图片
target = cv2.imread("test2.jpg")
cv2.imshow("target",target)
#读取模板图片
template = cv2.imread("target2.jpg")
target = cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)

fff = function1(template,target,3,2)
print(fff)
#获得模板图片的高宽尺寸
theight, twidth = template.shape[:2]
#执行模板匹配，采用的匹配方式cv2.TM_SQDIFF_NORMED
result = cv2.matchTemplate(target,template,cv2.TM_SQDIFF_NORMED)
#归一化处理
cv2.normalize( result, result, 0, 1, cv2.NORM_MINMAX, -1 )
print(result)
#寻找矩阵（一维数组当做向量，用Mat定义）中的最大值和最小值的匹配结果及其位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
print(min_val,max_val,min_loc,max_loc)
#匹配值转换为字符串
#对于cv2.TM_SQDIFF及cv2.TM_SQDIFF_NORMED方法min_val越趋近与0匹配度越好，匹配位置取min_loc
#对于其他方法max_val越趋近于1匹配度越好，匹配位置取max_loc
strmin_val = str(min_val)
#绘制矩形边框，将匹配区域标注出来
#min_loc：矩形定点
#(min_loc[0]+twidth,min_loc[1]+theight)：矩形的宽高
#(0,0,225)：矩形的边框颜色；2：矩形边框宽度
cv2.rectangle(target,min_loc,(min_loc[0]+twidth,min_loc[1]+theight),(0,0,225),2)
#显示结果,并将匹配值显示在标题栏上
cv2.imshow("template",template)
cv2.imshow("MatchResult----MatchingValue="+strmin_val,target)
cv2.waitKey()
cv2.destroyAllWindows()