'''
Code from Kazi Saidul Hasan

Implement the SingleRank appoach for keyphrase extraction in Python

By kai
'''
import sys

fileList=""
goldKeyList=""
fileDir=""
goldKeyDir=""
outputDir=""

keyCount=0
iteration=20

#document names
files=list()
#gold keyphrase file names
keylist=list()

docCount=0
docLen=0
nodeCount=0
windowSize=0

#actual document
document=list()
#actual document's POS tags
posTags=list()

totalKey=0
matched=0
predicated=0

# Map<string, string>
goldKeyMap=dict()
# string list
goldKey=list()

# Map<String, Map<String, double>>
srGraph=dict()
# Map<String, double>
srScore=dict()
# Map<String, List<Long>>
positon=dict()


def readParams(file):
    obj = open(file, 'r')
    lines = obj.readlines()
    paras=[]
    for x in lines:
        paras.append(x.split('\t\r\n'))
    print paras

def main(arg):
    '''
    :param arg:
    :return:
    '''
    print "Reading params..."
    #reading params
    readParams(arg[1])




if __name__ == '__main__':
   main(sys.argv)