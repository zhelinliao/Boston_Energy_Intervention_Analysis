def readFromFile(fileName):
    f = open(fileName, "r")
    f.readline() #burn header
    lines = f.readlines()
    data = []
    ethList = []
    ethicity = ['am.ind.', 'asian', 'black', 'hispanic', 'white']
    cnt = 0
    for l in lines:
        cnt += 1
       
        split = l.split(",")
        #print(split)
        name = split[0].strip().lower()
        if split[1] != 'rank':
            stat = int(split[2])

            for j in range(3,8):
                if split[j].find('(S)') != -1:
                    split[j] = '0.00'
                ethList.append(float(split[j].strip())/100)
            for i in range(5):
                ethList[i] = int(stat * ethList[i])
            data.append((name, stat, ethList))
            ethList = []
    f.close()
    return data
