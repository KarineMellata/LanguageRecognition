# LanguageRecognition
Simple language recognition classifier using Naive Bayes approach

All in Jupyter Notebook files
## Approach
First approach will be using word n grams with n=2

### Some useful notes on theory
Bayes is based on conditional probability :

P(class|data) = P(data|class)*P(class) / P(data)

Which is equivalent to

Posterior = likelihood * priori / evidence

Strong NB assumptions that make this work:
All features are independant (which is rarely true irl but still yields good predictions in practice)

## Information on dataset
0: Slovak
1: French
2: Spanish
3: German
4: Polish

## For report referencing
http://www.diva-portal.org/smash/get/diva2:839705/FULLTEXT01.pdf