from sklearn.metrics.pairwise import cosine_similarity


def similarity_score(job_cluster_vectors,resume_cluster_vectors):
  similarity_scores=0
  for cluster_id,job_vector in job_cluster_vectors.items():
    if cluster_id in resume_cluster_vectors:
      resume_vector=resume_cluster_vectors[cluster_id]
      similarity_scores+=cosine_similarity(job_vector.reshape(1, -1),resume_vector.reshape(1, -1))[0][0]
  return (similarity_scores/len(job_cluster_vectors))*10
