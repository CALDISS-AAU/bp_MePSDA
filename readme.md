# MePSDA

## Project description
This project focuses on training a BERTopic model based on collected data from news articles and transcribed data from YouTube. The data consists of interviews, recorded videos and radio interviews from the Danish film director Jonas Risvig. The primary sources are YouTube and the Danish news article database Infomedia. The main focus was to analyze and interpret, via computational methods, Jonas Risvigs attitudes towards social issues of gender, youth, casting and filmmaking targeting a young audience.

The repository consists of two pipelines each with several scripts: one for creating word-embeddings of the collected data and comparing it with a sample of wildcard words proposed by the researcher. Second pipeline utilizes said word-embeddings to build and train a BERTopic model to analyze and interpret the generated topics of the full dataset.

All scripts are written in python.

#### Summarizing the pipeline

1. Collecting Youtube videos
2. Creating data structure
3. Calculating word-embeddings using sentence-transformers
4. Matching embedding space with keywords divided into 3 themes
5. Training BERTopic model
6. Validating model output and creating visualizations based on results


#### Python packages:
pandas: Structuring and formatting data.
numba: Dependency of BERTopic
bertopic: Package used to build and train a BERT-based topic model
spacy: Creating the stopwords list for better handling in creating topics
tqdm: Progressbar for training and keyword search.
pysbd: chunking, segmentation and cleaning of text data.
scikit-learn: Calculating cosine similarity of word-embeddings
sentence-transformers: For using language model pipelines and utilizing pretrained models from the 🤗-Hub.
numpy: handling numpers and data arrays such as the embeddings.
umap: Uniform Manifold Approximation and Projection for Dimension Reduction. For dimensionality reductions of created topics.
hdbscan: Cluster data using hierarchical density-based clustering.
transformers: Utilizing pretrained sentence-transformers models for word-embeddings
openpyxl: For Filehandling
topic-wizard: For creating plots of model.

## BERTopic details
BERTopic is a hiearchical topic model leveraing BERT and TF-IDF to create meaningful topics of documents (Grootendoorst, 2020).
Several steps are utilized in BERTopic to create these topics:
1. Embed documents 

    - Converting text to numerical representations through sentence-transformers. Using a sentence-transformer helps preserve semantic meaning of the text which enhances the performance of the clustering.

2. Dimensionality reduction.
    - The numerical representations of the documents are then reduced. The default is UMAP, which is also utilized in this project as this works great as a technique to preserve local and global structure of the dimensions of data.

3. Clusering of documents
    - After the reduction the data is then clustered through a density-based clustering method, HDBSCAN. This methods is hiearchical which means makes it ideal in finding outliers and clustering together documents and thus not forcing documents into cluster where they do not belong.

4. Bag-of-Words
    - Before the creation of topic representations, all texts in each cluster are represented as bag-of-words in order to perform word counts.

5. Topic representation
    - From the bag-of-words representation the topic reprsentation is then calculated through the application of c-TF-IDF which compares the importance of words between documents.

## Scripts details

#### 1. Get YouTube (py-scripts/01_get_yt.py).

The script uses the pytube-fix package to download the desired youtube videos in .mp4 format for later transcription.

#### 1.5. Transcription of youtube-videos

After downloading the YouTube videos, we used a transcriber based on the Whisper architecture to automatically transcribe all videos into csv files. The transcription process was performed on an internally available cloud platform, so no dedicated script is provided for this step. 

The overall goal was to create a clean and consistent dataset, enabling the seamless integration and merging of the different source materials.

#### 2. Datastructure (py-scripts/02_datastructure.py)

The second script manages the data structure of the transcribed files and news articles provided as .txt files. It creates a separate DataFrame for each data source and then concatenates them into a single DataFrame. The YouTube videos are sourced from three different playlists: interviews, JR’s master class, and miscellaneous videos. Each source is processed into its own DataFrame, which is then merged and combined with the news articles to form a comprehensive DataFrame containing all the material.

Additionally, the script segments each text column into chunks of 200 characters, ensuring the splits align with natural pauses for improved coherence.
The function can be found at the available `mepsda_funs.py` script in the modules folder.

#### 3. Calculating word embeddings (py-scripts/03_calc_embeddings.py)

The third script calculates embeddings of the full dataset using the sentence transformer [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) and saves a numpy array for further analysis. For a general introduction to word embeddings please read this [introductory post of NLP methods and word embeddings using transformers](https://medium.com/@RobinVetsch/nlp-from-word-embedding-to-transformers-76ae124e6281).

#### 4. Keyword search (py-scripts/04_keyword_search.py)

Fourth script utilizes the calculated embeddings and compares them with the mean calculated embeddings of the keywords. The keywords are a collection of thematic verbatim and wildcard keywords defined by the collaborating researcher based on assumptions of occurring subjects or topics such as gender, casting, locations, filmmaking terms etc. Keywords are formatted in a nested dictionary with keywords spread across each theme. The mean cosine similarity resulted in overall succesfull results with varying degrees of likeness with some themes being more represented than others, specifically theme 2 was most represented when accounting for the score.

##### The themes are defined as such:
```
keywords = {
    'theme1': [
        'pige',
        'dreng*',
        'tøj*',
        'ven*',
        'diversitet',
        'ung*',
        'seksu*',
        'sex',
        'stereotyp*',
        'mand*'
        'kvinde*',
        'maskulin*',
        'feminin*'
        ],
    'theme2': [
        'improvis*',
        'cast*',
        'skuespil*',
        'skabe*',
        'målgruppe',
        'kritik*',
        'drøm*',
        'film*',
        'serie*',
        'drama*',
        'leg*',
        'værktøj',
        'håndværk',
        'inspir*',
        'lær*'
    ],
    'theme3': [
        'location',
        'Silkeborg',
        'Jylland',
        'Nordsjælland',
        'verden',
        'univers',
        'by',
        'provins*',
        'gymnasie*',
        'miljø',
        'virkelig*',
        'autenti*',
        'filter'
    ]
}
```

The sentences are scored based on the count of keywords in the sentence. Both datasets are then compared using cosine similarity from the scikit-learn package to compared the two embeddings spaces likeness.
A final file where the top 5 sentences from the data for each theme is created and saved.

#### 5. Train model (py-scripts/05_train_topicmodel.py)
The next script defines the model parameters and trains the BERTopic model on our data. SpaCy is utilized to define custom stopwords that are not captured by the pre-trained language model. These stopwords include a small collection of onomatopoeic expressions from the transcribed YouTube videos and article-specific terms from the Infomedia articles.

As described in the introduction to BERTopic, the model primarily operates at the document level, unlike classical LDA. This approach was not suitable for our use case due to the varying lengths of documents in the dataset. To address this, we implemented the chunking method that splits each document into 200-character-long segments, ensuring the model could identify appropriately sized topics and clusters.

For embeddings, we used the [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) sentence-transformer model, which demonstrated good overall performance on Danish data and high computational efficiency.

The HDBSCAN, UMAP, CountVectorizer, and BERTopic parameters were carefully fine-tuned through multiple iterations to achieve optimal results. The final model successfully identified 14 topics of varying sizes and token representations. Additionally, the model uncovered several underlying subtopics and provided further insights into thematic patterns based on the predefined keyword sets.

#### 6. Validating the model (py-scripts/06_model_validate.py)
The final script in the pipeline validates the model results by generating various plots to visualize and assess the topics and output. For this purpose, plotly.graph_objects, topicwizard, and BERTopic’s built-in visualization tools were utilized. These plots specifically visualize the topic clusters, providing an overview of the data's structure and similarity.

In this context, the clustering makes sense since most of the content revolves around Jonas Risvig and his work as a Danish film director. Despite the sources being closely related in terms of content, the BERTopic model successfully distinguishes between individual paragraphs and identifies several distinct subjects within the material.

## Concluding remarks
This repository documents the usecase of a more advanced topic modelling called BERTopic which utilizes modern transformer architechture and clustering methods in order to find topics in text data. The overall findings of the repository was a reasonble amount of topics in overall not so diverse dataset from different sources. It further explores the use case of a simple computational pipeline from digital transcription, data management and topic analysis in a field thats been typically dominated by qualitative methods.

## Future directions
As topic modelling and data analysis pipelines in the computational humanities continue to evolve, there is a growing need for more research and open repositories. Developing better workflows will help streamline and automate the analysis of text corpora, enabling faster and more accurate insights in the data. Further enhancing and supplementing the general methodology of textual analysis.


## References

Explosion AI. (n.d.). spaCy language model for Danish (da_core_news_lg) [Computer software]. https://spacy.io/models/da

Grootendorst, M. P. (n.d.). BERTopic. https://maartengr.github.io/BERTopic/index.html

Robin. (2022, January 22). NLP — from word embedding to transformers | by Robin | medium. Medium. https://medium.com/@RobinVetsch/nlp-from-word-embedding-to-transformers-76ae124e6281
