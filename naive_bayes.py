import unicodecsv
import numpy
import nltk
import csv
from collections import defaultdict


def load_dataset_csv(filename):
    info = open(filename, "rb")
    has_header = unicodecsv.Sniffer().has_header(info.read(1024))
    info.seek(0)
    incsv = csv.reader(info)
    if has_header:
        next(incsv)  #Skip header
    dataset = list(incsv)
    return dataset

filename1 = 'train_set_x.csv'
dataset_without_classes = load_dataset_csv(filename1)
print('Loaded data file {0} with {1} rows').format(filename1, len(dataset_without_classes))
print(dataset_without_classes[18])

def loadClassCsv(filename):
    info = open(filename, "rb")
    has_header = csv.Sniffer().has_header(info.read(1024))
    info.seek(0)
    incsv = csv.reader(info)
    if has_header:
        next(incsv)  #Skip header
    dataset = list(incsv)
    return dataset

filename = 'train_set_y.csv'
classds = loadClassCsv(filename)
print('Loaded data file {0} with {1} rows').format(filename, len(classds))
print(classds[18])

#get Id from list object
def getId(line):
    info = unicodecsv.reader(line, delimiter=',')
    new_line = list(info)
    return new_line[0]

#get Utterance from list object
def getUtt(line):
    info = unicodecsv.reader(line, delimiter=',')
    new_line = list(info)
    return new_line[1]
#print(getUtt(dataset[1]))

def getClass(line):
    info = csv.reader(line, delimiter=',')
    new_line = list(info)
    new_line_item = ' '.join(new_line[1])
    return new_line_item

def appendClass(dataset):
    for i in range(len(dataset)):
        dataset[i].append(getClass(classds[i])) #Can assume this since they're all in the same position
    return dataset
dataset_with_classes = appendClass(dataset_without_classes)
print(dataset_with_classes[20])

#Separate by class
#Assuming last index has the class nb
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (int(vector[-1]) not in separated):
            separated[int(vector[-1])] = []
        separated[int(vector[-1])].append(vector)
    return separated

def createBigrams(ds, n, c):
    text = getUtt(ds[n])
    vector = ds[n]
    class1 = vector[-1]
    textNeeded = ''.join(text)
    textNeeded = textNeeded.replace(" ","")
    n = 2
    if c == 1:
        bigrams = [[(textNeeded[i:i+n]),class1] for i in range(0,len(textNeeded), 1)]
    else:
        bigrams = [(textNeeded[i:i + n]) for i in range(0, len(textNeeded), 1)]
    return bigrams

all_ngrams = []
for i in range(len(dataset_with_classes)):
    all_ngrams.extend(createBigrams(dataset_with_classes,i,1))
print("All ngrams list generated")

all_ngrams_unique_without_class = set()
for i in range(len(dataset_with_classes)):
    bigrams = createBigrams(dataset_with_classes,i,0)
    for j in range(len(bigrams)):
        all_ngrams_unique_without_class.add(bigrams[j])
print("All ngrams list generated 2")


separated = separateByClass(all_ngrams)

def calculatePrior(classDataset, language):
    print("Calculating prior")
    sum = len(classDataset)
    occurence = 0
    for i in range(sum):
        if(getClass(classDataset[i]) == language):
            occurence = occurence + 1
    prior = occurence/float(sum)
    return prior
#nb = calculatePrior(classds, '4')
#print(nb)

#Added laplace smoothing
def calculate_likelihood(all_seq, data, class_wanted):
    # print("Calculating likelihood")
    arr = separated[class_wanted]
    occurence = 0
    for i in range(len(separated[class_wanted])):
        if data in arr[i]:
            occurence += 1
    h = (occurence + 1) / (float(len(separated[class_wanted])) + len(all_ngrams_unique_without_class))
    return h

    # nb = calculateLikelihood(all_ngrams, "er", 1)
    # print(nb)

#This isn't getting used
def classify_a_bigram(bigram):
    arr_of_possibilities = [0,0,0,0,0]
    #Hardcoded the prior calculated above because my computer slow AF
    prior_arr = [0.0512337396977,0.510937844689,0.253054965879,0.133857954484,0.0509154952498]
    arr_of_langs = ["Slovak","French","Spanish","German","Polish"]
    for i in range(len(arr_of_possibilities)):
        #print("Calculation of stats for " + arr_of_langs[i])
        prior = prior_arr[i]
        likelihood = calculate_likelihood(all_ngrams, bigram, i)
        arr_of_possibilities[i] = prior*likelihood
    if arr_of_possibilities == [0,0,0,0,0]:
        lang = 1
    else:
        lang = numpy.argmax(arr_of_possibilities)
    return lang
#print(classify_a_bigram("vp"))

def load_testset_csv(filename):
    info = open(filename, "rb")
    has_header = csv.Sniffer().has_header(info.read(1024))
    info.seek(0)
    incsv = csv.reader(info)
    if has_header:
        next(incsv)  # Skip header
    dataset = list(incsv)
    return dataset


filename = 'test_set_x.csv'
dataset_test_set = load_testset_csv(filename)
print('Loaded data file {0} with {1} rows').format(filename, len(dataset_test_set))
length = len(dataset_test_set)
# Hardcoded the prior calculated above because my computer slow AF
prior_arr = [0.0512337396977, 0.510937844689, 0.253054965879, 0.133857954484, 0.0509154952498]
likelihood_localizer = {0: {},
                        1: {},
                        2: {},
                        3: {},
                        4: {}
                        }
with open('output.csv', "wb") as csvfile:
    writer = csv.writer(csvfile)
    header = "Id"
    category = "Category"
    writer.writerow([header] + [category])
    for i in range(length):
        ngrams = createBigrams(dataset_test_set, i, 0)
        #print("We are at the sentence " + str(i) + " on 118508")
        likelihood = {0: [],
                      1: [],
                      2: [],
                      3: [],
                      4: []
                       }
        for j in range(len(ngrams)):
            m = 0
            while m < 5:
                if likelihood_localizer[m].has_key(ngrams[j]) is False:
                    #print("we here")
                    likelihood_calculated = calculate_likelihood(all_ngrams, ngrams[j], m)
                    likelihood_localizer[m][ngrams[j]] = likelihood_calculated
                    likelihood[m].append(likelihood_calculated)
                else:
                    likelihood_calculated = likelihood_localizer[m].get(ngrams[j])
                    likelihood[m].append(likelihood_calculated)
                m = m + 1
        # array_of_languages.append(classify_a_bigram(ngrams[j]))
        k = 0
        array_of_languages = []
        while k < 5:
            prod = numpy.prod(likelihood[k])
            # Bayesian for each language
            array_of_languages.append(prod * prior_arr[k])
            k = k + 1
        # Winning language
        language = str(numpy.argmax(array_of_languages))
        #print(array_of_languages)
        print("sentence " + str(i) + " is in " + language)
        writer.writerow(getId(dataset_test_set[i]) + [language])
