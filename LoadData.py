def readFromFile(fileName):
    f = open(fileName, "r")
    f.readline() #burn header
    lines = f.readlines()
    data = []
    ethList = []
    #statList = []
    ethicity = ['am.ind.', 'asian', 'black', 'hispanic', 'white']
    cnt = 0
    for l in lines:
        cnt += 1
       
        split = l.split(",")
        #print(split)
        name = split[0].strip().lower()
        if split[1] != 'rank':
            #for k in range(1,3):
                #statList.append(split[k])
            stat = int(split[2])

            #if (stat < 10000):
            #    break

            for j in range(3,8):
                if split[j].find('(S)') != -1:
                    split[j] = '0.00'
                ethList.append(float(split[j].strip())/100)
            #print(name, stat, ethList)
            #ethList.insert(4, 0.00)
            #nameRecord = surname(name, statList, ethList)
            #nameDict[nameRecord.name] = nameRecord

            for i in range(5):
                ethList[i] = int(stat * ethList[i])
                '''
                num //= 100
                for j in range(num):
                    data += [(name, ethicity[i])]
                '''
            data.append((name, stat, ethList))

            ethList = []
            #statList = []
            
            #print(statList[:2])
            #print nameRecord.toString()
        #if cnt > 15:
        #    break
    f.close()

    return data



#data = readFromFile("surname_ethnicity_data.csv")
#print(data[2])