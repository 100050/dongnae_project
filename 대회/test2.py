time = open("아이디어.txt", "r", encoding='UTF8')

a = time.readlines()

print(a[-1])

time.close()

with open("아이디어.txt", "r", encoding='UTF8') as time:
    a = time.readlines()

    print(a[-1])


