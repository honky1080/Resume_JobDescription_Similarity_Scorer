An unsupervised Natural Language Processing (NLP) architecture designed to parse, categorize, and evaluate candidate resumes against complex job descriptions. By mapping a high-density vocabulary space using a Skip-gram Word2Vec configuration and partitioning technical capabilities using K-Means Clustering, this engine computes localized segment similarities. This approach establishes a robust defensive filter against "keyword-stuffing" techniques that typically exploit primitive automation systems.


[ Messy Raw Resumes ] ---> Regex Cleaning & Morphological Lemmatization ---> Skip-gram Word2Vec Neural Embedding (100-D Space) ---> K-Means Mathematical Domain Partitioning (K=26) ---> Localized Document Vector Extraction & Filtered Cosine Similarity ---> [ Rank-Ordered Shortlist (0 - 10 Scale) ]


1. Skip-gram Word2Vec over TF-IDF
Traditional keyword matchers or frequency-based models (like CountVectorizer or TF-IDF) only look for precise textual overlaps. If a job description requests React, they fail to award value to a profile heavily emphasizing Angular or Vue. This engine utilizes a Word2Vec Skip-gram model (sg=1) to predict sliding context windows. It maps words based on structural neighborhood contexts so that complementary technologies automatically cluster closely in geometric distance. Skip-gram was selected over CBOW because it extracts richer signals from small text corpora and preserves niche, rare technical competencies.

2. Localized Document Filters over Global Averages
A naive vector space implementation averages an entire document's words together into one coordinate vector. This introduces a major vulnerability: candidates can inflate their scores by keyword-stuffing unrelated terms at the bottom of their resumes.
This system solves that by calculating separate vectors for individual K-Means clusters. When scoring against a specific job description, the engine isolates only the active target clusters demanded by the role and evaluates the candidate's vector alignment strictly within those functional boundaries, ignoring background noise.