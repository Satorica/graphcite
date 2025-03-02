from sentence_transformers import SentenceTransformer, models

# Load SciBERT using Hugging Face's Transformers
transformer = models.Transformer("./scibert_scivocab_uncased")
pooler = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")

# Create a SentenceTransformer model
model = SentenceTransformer(modules=[transformer, pooler])

# Save the model in sentence-transformers format
model.save("./scibert_scivocab_uncased_sentence-transformers")