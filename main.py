import pandas as pd
from src.preprocess import clean_and_lemmatize
from src.embeddings import train_word2vec
from src.clustering import run_kmeans,clusterization,cluster_vectorization
from src.scorer import similarity_score


def main(job_description):
    print("--- 1. LOADING DATASET ---")
    df = pd.read_csv("UpdatedResumeDataSet.csv")
    print(f"Successfully loaded {len(df)} resumes.")

    print("\n--- 2. PREPROCESSING DATA ---")
    print("Standardizing language and removing noise (this may take a moment)...")
    df['tokenized_resume'] = df['Resume'].apply(clean_and_lemmatize)

    print("\n--- 3. TRAINING LANGUAGE ARCHITECTURE ---")
    corpus = df['tokenized_resume'].tolist()
    w2v_model = train_word2vec(corpus, vector_size=100, window=5)

    print("\n--- 4. EXECUTING DOMAIN CLUSTERING ---")
    num_clusters = 26
    word_to_clusters = run_kmeans(w2v_model, num_clusters=num_clusters)

    print("\n--- 5. CONFIGURING TARGET JOB DESCRIPTION ---")
    job_description = clean_and_lemmatize(job_description)
    jd_clusters = clusterization(job_description,num_clusters, word_to_clusters)
    job_cluster_vectors = cluster_vectorization(jd_clusters, w2v_model)

    print("\n--- 6. EVALUATING CANDIDATES ---")
    print("Computing localized context similarity vectors across the DataFrame...")
    df['tokenized_resume'] = df['tokenized_resume'].apply(lambda x:clusterization(x,num_clusters,word_to_clusters))
    df['tokenized_resume'] = df['tokenized_resume'].apply(lambda x:cluster_vectorization(x,w2v_model))
    df['similarity_score'] = df['tokenized_resume'].apply(lambda x: similarity_score(job_cluster_vectors,x))
    df=df.drop(columns=['tokenized_resume'])
    pd.set_option('display.max_columns', None)
    print(df)

job_description = """
We rely on insightful data to power our production systems and engineering solutions. 
We are seeking an experienced data scientist to deliver business insights on a daily basis. 
The ideal candidate will have mathematical and statistical expertise, along with natural 
curiosity and a creative mind. While mining, interpreting, and cleaning our data, this 
person will be relied on to ask questions, connect the dots, and uncover hidden opportunities 
for realizing the data full potential. As part of a team of specialists, the data scientist 
will slice and dice data using various methods and create new visions for the future.

Objectives of this role:
Collaborate with product design and engineering teams to develop an understanding of needs.
Research and devise innovative statistical models for data analysis.
Communicate findings to all stakeholders.
Enable smarter business processes by using analytics for meaningful insights.
Keep current with technical and industry developments.

Responsibilities:
Serve as lead data strategist to identify and integrate new datasets that can be leveraged 
through our product capabilities, and work closely with the engineering team in the 
development of data products.
Execute analytical experiments to help solve problems across various domains and industries.
Identify relevant data sources and sets to mine for client business needs, and collect 
large structured and unstructured datasets and variables.
Devise and utilize algorithms and models to mine big data stores perform data and error 
analysis to improve models clean and validate data for uniformity and accuracy.
Analyze data for trends and patterns, and interpret data with clear objectives in mind.
Implement analytical models in production by collaborating with software developers and 
machine learning engineers.

Required skills and qualifications:
Seven or more years of experience in data science.
Proficiency with data mining, mathematics, and statistical analysis.
Advanced experience in pattern recognition and predictive modeling.
Experience with Excel, PowerPoint, Tableau, SQL, and programming languages like Java, 
Python, and SAS.
Ability to work effectively in a dynamic, research oriented group that has several 
concurrent projects.
"""

main(job_description)