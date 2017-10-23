import unicodecsv
import numpy
import csv

#Load the dataset without labels
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

#Load the labels dataset
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

#
#Getters
#
#get Id from list object
def get_id(line):
    info = unicodecsv.reader(line, delimiter=',')
    new_line = list(info)
    return new_line[0]

#get Utterance from list object
def get_utt(line):
    info = unicodecsv.reader(line, delimiter=',')
    new_line = list(info)
    return new_line[1]

#get class from
def get_class(line):
    info = csv.reader(line, delimiter=',')
    new_line = list(info)
    new_line_item = ' '.join(new_line[1])
    return new_line_item

#Creating array of data + class labels
def append_class(dataset):
    for i in range(len(dataset)):
        dataset[i].append(get_class(classds[i])) #Can assume this since they're all in the same position
    return dataset
dataset_with_classes = append_class(dataset_without_classes)

#Separating by class
#Assuming last index has the class number
def separate_by_class(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (int(vector[-1]) not in separated):
            separated[int(vector[-1])] = []
        separated[int(vector[-1])].append(vector)
    return separated

#Creating bigrams out of utterances
def create_bigrams(ds, n, c):
    text = get_utt(ds[n])
    vector = ds[n]
    class1 = vector[-1]
    textNeeded = ''.join(text)
    textNeeded = textNeeded.replace(" ","")
    n = 5
    if c == 1:
        bigrams = [[(textNeeded[i:i+n]),class1] for i in range(0,len(textNeeded), 1)]
    else:
        bigrams = [(textNeeded[i:i + n]) for i in range(0, len(textNeeded), 1)]
    return bigrams

all_ngrams = []
for i in range(len(dataset_with_classes)):
    all_ngrams.extend(create_bigrams(dataset_with_classes, i, 1))
print("All ngrams list generated")

#Used for laplace smoothing
all_ngrams_unique_without_class = set()
for i in range(len(dataset_with_classes)):
    bigrams = create_bigrams(dataset_with_classes, i, 0)
    for j in range(len(bigrams)):
        all_ngrams_unique_without_class.add(bigrams[j])
print("All ngrams list generated 2")

#Separated by class dataset for likelihood
separated = separate_by_class(all_ngrams)

#This method does not run, we ran it once and stored the data
#for rapidity purposes
def calculate_prior(classDataset, language):
    print("Calculating prior")
    sum = len(classDataset)
    occurence = 0
    for i in range(sum):
        if(get_class(classDataset[i]) == language):
            occurence = occurence + 1
    prior = occurence/float(sum)
    return prior

#Likelihood + laplace smoothing
def calculate_likelihood(all_seq, data, class_wanted):
    arr = separated[class_wanted]
    occurence = 0
    for i in range(len(separated[class_wanted])):
        if data in arr[i]:
            occurence += 1
    h = (occurence + 1) / (float(len(separated[class_wanted])) + len(all_ngrams_unique_without_class))
    return h

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
# Hardcoded the prior calculated above
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
        ngrams = create_bigrams(dataset_test_set, i, 0)
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
                    likelihood_calculated = calculate_likelihood(all_ngrams, ngrams[j], m)
                    likelihood_localizer[m][ngrams[j]] = likelihood_calculated
                    likelihood[m].append(likelihood_calculated)
                else:
                    likelihood_calculated = likelihood_localizer[m].get(ngrams[j])
                    likelihood[m].append(likelihood_calculated)
                m = m + 1
        k = 0
        array_of_languages = []
        while k < 5:
            prod = numpy.prod(likelihood[k])
            # Bayesian for each language
            array_of_languages.append(prod * prior_arr[k])
            k = k + 1
        # Winning language
        language = str(numpy.argmax(array_of_languages))
        print("sentence " + str(i) + " is in " + language)
        writer.writerow(get_id(dataset_test_set[i]) + [language])
