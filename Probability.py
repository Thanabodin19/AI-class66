"""
Subject : Artificial Intelligence
Assignment Code: Classwork #2
StudID: 6410301026
StudName: Thanabodin Keawmaha
Deptment: CPE
Due Date: 2023-12-18
"""

#Probability Theory

def Bayes_rule(Red, Blue):

    py = Red[0]/(Red[0]+Blue[0])                                #P(Y)
    pxy = Red[2]/(Red[1] + Red[2])                              #P(X|Y)
    px = ((pxy*py) + (Blue[2]/(Blue[1]+Blue[2])*(1-py)))        #P(X)
    pyx = pxy*py/px                                             #Blueayes' Theorem

    # print(f"\nP(X|Y)*P(X) = {pxy}*{py}")
    # print(" "*13,"-"*20)
    # print(f"P(X)        = {pxy}*{py} + {Blue[2]/(Blue[1]+Blue[2])}*{1-py}")
    # print(f"\nP(X) = {px}   P(Y) = {py}   P(X|Y) = {pxy}")
    # print(f"P(Y|X) = {pyx}")
    # print(f"Red box = {Red}\nBlue box = {Blue}")
    return pyx


if __name__ == "__main__":
    # index 0 = ความน่าจะเป็นของการสุ่มกล่อง
    # index 1 = Apple
    # index 2 = Orange
    Redbox = []
    Bluebox = []

    Redbox.append(int(input("Redbox = ")))
    Bluebox.append(int(input("Bluebox = ")))

    Redbox.append(int(input("Red Box Apple = ")))
    Redbox.append(int(input("Red Box Orange = ")))

    Bluebox.append(int(input("Blue Box Apple = ")))
    Bluebox.append(int(input("Blue Box Orange = ")))

    sum_R = Redbox[1] + Redbox[2] 
    sum_B = Bluebox[1] + Bluebox[2]
    Rate_Apple = ((Redbox[1]/sum_R*Redbox[0])+(Bluebox[1]/sum_B*Bluebox[0]))
    Rate_Orange = ((Redbox[2]/sum_R*Redbox[0])+(Bluebox[2]/sum_B*Bluebox[0]))
    Rate_O_R = Bayes_rule(Redbox, Bluebox)

    print(f"ความน่าจะเป็นที่จะหยิบได้ Apple = {Rate_Apple}")
    print(f"ความน่าจะเป็นที่จะหยิบได้ Orange = {Rate_Orange}")
    print(f"เมื่อหยิบส้มมาได้ อยากทราบว่ามีความน่าจะเป็นที่จะมาจากกล่องสีแดง = {Rate_O_R}")

