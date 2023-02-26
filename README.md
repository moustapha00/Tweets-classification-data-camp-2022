# Data Camp Projet: Tweets Classification


_Authors: Yao Pacome, Nicolas Ntollot, Sonali, Moustapha, Vadhel

##II Introduction

Twitter sentiment analysis is the method of Natural Language Processing (NLP). 
Sentiment Analysis is the process of determining the sentiment behind the tweet whether a piece of written text(tweet) is positive, neutral or negative.

The goal is using machine learning techniques and natural language processing to extract the subjective information of a document and try to classify it according to its polarity such as positive, neutral or negative. In real life, Sentiment analysis is very important, for example, in business analytics data is mined for patterns that would help better understand customers, and improve sales and marketing. Also, sentiment analysis brings together various research areas such as natural language
processing, data mining and text mining, and is fast becoming of major importance to organizations as they strive to integrate computational intelligence methods into their operations, and attempts to shed more light on, and improve, their products and services.

 In this project, we analyze the sentiments behind the twitter’s tweet. Those refers to the people felling  about four compagnies: Apple, Google, Tweeter, Microsoft.  We have four  type of sentiment: Positive, Neutral, Negative and Irrelevant

### Dataset
Our dataset contains 5112 tweets and those tweet are labelling w.r.t  the type above. We split the data to the training data and testing.

Types	Occurrences
irrelevant	1689
negative	572
neutral	2333
positive	519
Total général	5113


#### Set up

Open a terminal and

1. install the `ramp-workflow` library (if not already done)
  ```
  $ pip install ramp-workflow
  ```
  
2. Follow the ramp-kits instructions from the [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki/Getting-started-with-a-ramp-kit)

#### Local notebook

Get started on this RAMP with the [dedicated notebook](solar_wind_starting_kit.ipynb).

To test the starting-kit, run


```
ramp-test --quick-test
```


#### Help
Go to the `ramp-workflow` [wiki](https://github.com/paris-saclay-cds/ramp-workflow/wiki) for more help on the [RAMP](https://ramp.studio) ecosystem.



