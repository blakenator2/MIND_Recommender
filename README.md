# MIND Recommender
## **Introduction**

This is a full news recommendation system using the small version of the ***Microsoft News Dataset***. I did this project to learn more about the entire process of creating a recommendation system. My model is a neural recommender that recommends certain news articles given a user's impression history. Everything from getting the data to the actual recommendations are included in this folder. This project uses Pytorch for the model and MatPlotLib for all graphs and visualizations.
___

## **Dataset description**
The MIND dataset for news recommendation was collected from anonymized behavior logs of Microsoft News website. For this project, I used the MIND-small dataset, which was created by randomly sampling 50,000 users and their behavior logs. The MIND-small dataset has a preset training and validation split, which should make the results I got much more repeatable.

The MIND dataset training and validation contain 4 files. behaviors.tsv file contains impression logs of users. news.tsv includes information about each news article. entity_embedding.vec contains embeddings of entities in the news article. relation_embedding.vec contains the relationships between news articles learned from the entity embeddings.

For an in depth description of each of the files and their data included in the training and validation sets, you can go to the [MSNews Github page linked here.](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md)

___

## **Methodology**
What model architecture you used
What the pipeline looks like
How data flows through the system
What “negative samples” even means



___

## **Training details**
Every model was evaluated based on loss, AUC, MRR, ndcg@5, and nDCG@10. There are 4 models, one base model and 3 with hyperparameter modifications. Here are the specifics for each model:

**All models share these hyperparameters**: Batch size = 64, Learning Rate = 1e-4, Epochs = 5

|Model Name| Negative Samples | Max History | Max Title Length| Number of Attention Heads | Attention Head Dimension|
|----------|------------------|-------------|-----------------|---------------------------|-------------------------|
|**Base** | 4| 50|   30| 16| 16|
|**Short**| 4| 25|   15| 16| 16|
|**Negative**| 8| 50|   30| 16| 16|
|**Less Attention**| 4| 50|   30| 8| 8|

___

## **Results**

|Model|Loss|AUC|MRR|nDCG@5|nDCG@10|
|-----|----|---|---|------|-------|
|Base|1.2711|0.4987|0.4596|0.5917|0.5917|
|Short| 1.2775| 0.5081|0.4660|0.5967|0.5967|
|Negative|1.2995|0.7582|0.4657|0.5968|0.5968|
|Less Attention| | | | |

___

## **Discussion and conclusion**


___