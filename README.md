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
This project uses an NRMS model, or neural news recommendation with multi-head self-attention. The NRMS uses a news encoder and a user encoder. The news encoder uses multi-head self-attention to learn news embeddings from news titles. The user encoder learns embeddings of users from their browsed news and its relation to the news in the same way the news encoder does.

The data starts in the data_load function in the data_loader.py file, which is called by the train function in train.py. The data_load function calls preprocess in the same file. First, the preprocess function tokenizes the news titles of both the train and validation sets. Then, it loads the GloVe file into an array. Next, the titles are encoded, which simply means their length is padded or truncated. Finally, we call parse_behavior which create an array of the history, label, and adds negative samples, or samples that were not clicked on. These samples are added to help the model generalize its learning. Finally, the data is put into a PyTorch DataLoader which will load the data in increments equal to the batch size.

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
|Less Attention| 1.3271|0.5093 | 0.4610 | 0.5932 | 0.5932 |

___

## **Discussion and conclusion**

From the gathered results above, we can see that the Negative model performed the best in every stat but loss. We also see that the base model performed the worst out of all the other models. These conclusions, for the most part, do make sense. 
My negative model had to work harder to learn more generalized traits of news recommendation. By nature of having more negative samples, it also had the most data to work off of. This seemingly proved to be very beneficial in this small set of epochs. 
The short model performed the next best. I believe this is because it had to predict less and deal with less noise in the data. So even though it had less data to work with, it had a smaller range of values it was guessing and a smaller range of values that could interfere with the output, which I believe would lead to stronger guesses overall. 
My less attention model proved to be slightly better than the base model in most stats, which confuses me. My best guess is that with more epochs, the base model will heavily out perform the less attention model. It just so happened that in this small subset, the less attention model was able to keep up and out perform the base model.

These results have shown me the power of negative sampling, even in such a small dataset. It also leads me to believe that keeping track of training and testing loss curves is super important. Having these methods of tracking loss will help me determine how much the model needs to be trained. It should also help me avoid fluke results like the less attention model here.

___