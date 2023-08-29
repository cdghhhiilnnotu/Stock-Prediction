import csv
from datetime import datetime
import os
import matplotlib.pyplot as plt


def load_data(filename):
    myListDate = []
    myListOpen = []
    myListHigh = []
    myListLow = []
    myListClose = []
    myListTotal = []

    with open(filename) as stocks:
        dataFile = csv.reader(stocks)
        next(dataFile)
        for row in dataFile:
            if len(myListDate) > 0:
                if row[1].split(' ')[0] != myListDate[len(myListDate) - 1]:
                    tempList = [myListDate[len(myListDate) - 1], myListOpen[len(myListOpen) - 1], myListHigh[len(myListHigh) - 1], myListLow[len(myListLow) - 1], myListClose[len(myListClose) - 1]]
                    myListTotal.append(tempList)
                    # Get List Date
                    myListDate.append(row[1].split(' ')[0])
                    # Get List Open
                    myListOpen.append(row[2])
                    myListHigh.append(row[3])
                    myListLow.append(row[2])
                    myListClose.append(row[5])
                else:
                    # Get List Close
                    myListClose[len(myListClose) - 1] = row[5]
                    
                    # Get List High
                    if(myListHigh[len(myListHigh) - 1] < row[3]):
                        myListHigh[len(myListHigh) - 1] = row[3]

                    # Get List Low
                    if(myListLow[len(myListLow) - 1] > row[4]):
                        myListLow[len(myListLow) - 1] = row[4]
            else:
                myListDate.append(row[1].split(' ')[0])
                myListOpen.append(row[2])
                myListHigh.append(row[3])
                myListLow.append(row[2])
                myListClose.append(row[5])
    return myListTotal

def write_data(listData, filename):
    if os.path.exists(filename):
        os.remove(filename)
        print(f"Existed {filename}. Deleted File!")
    with open(filename, 'w', newline='') as myFile:
        wr = csv.writer(myFile)
        listHead = ['Date', 'Open', 'High', 'Low', 'Close']
        wr.writerow(listHead)
        for word in listData:
            wr.writerow(word)


def Plot_Data(filename, idxFig):
    listTotal = load_data(filename)
    listDate = [datetime.strptime(item[0], '%m/%d/%Y') for item in listTotal]
    listOpen = [float(item[1]) for item in listTotal]
    listHigh = [float(item[2]) for item in listTotal]
    listLow = [float(item[3]) for item in listTotal]
    listClose = [float(item[4]) for item in listTotal]

    plt.subplot(2,2,idxFig)
    plt.plot(listDate, listHigh, label="High", color="blue", linewidth=1)
    plt.plot(listDate, listLow, label="Low", color="red", linewidth=1)
    plt.title(filename)
    plt.xlabel("Date")
    plt.ylabel("Value")

if __name__ == "__main__":
    for file in os.listdir('Data'):
        write_data(load_data(f'Data/{file}'), f'Clean/Clean-{file}')

    plt.show()








