# Data Camp Projet: Tweets Classification


_Authors: Abdellahi Ebnou Oumar, Yao Pacome, Nicolas Tollot, Sonali Patekar, Moustapha Mohamed Mahmoud, Mohamed Vadhel Ebnou Oumar_

## Introduction

Twitter sentiment analysis is a Natural Language Processing (NLP) method that involves determining the sentiment behind a tweet, whether it is positive, neutral, or negative. The goal is to extract subjective information from a document using machine learning and NLP techniques and classify it according to its polarity.

In real life, sentiment analysis is crucial for business analytics, where data is mined to identify patterns that help better understand customers and improve sales and marketing. Sentiment analysis is a multidisciplinary field that encompasses natural language processing, data mining, and text mining. As organizations strive to integrate computational intelligence methods into their operations, sentiment analysis is becoming increasingly important in shedding light on and improving their products and services.

This project involves analyzing the sentiments behind tweets related to four companies: Apple, Google, Twitter, and Microsoft. The sentiments are categorized as Positive, Neutral, Negative, or Irrelevant, indicating people's feelings towards these companies.

### Dataset
Our dataset contains 5112 tweets labeled as irrelevant, negative, neutral, or positive, representing people's feelings towards the four companies mentioned above. We split the data into training and testing sets.

Types	| Occurrences
--- | ---
irrelevant |	1689
--- | ---
negative	| 572
--- | ---
neutral	| 2333
--- | ---
positive	| 519
--- | ---
Total général |	5113


#### Set up

Open a terminal and

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install ramp-workflow
  ```
  
2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [dedicated notebook](tweets_classification_starting_kit.ipynb).

To test the starting-kit, run


```
ramp-test --quick-test
```


#### Help
Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](https://ramp.studio) ecosystem.



