import numpy as np
import tensorflow as tf

def preprocess_captions(raw_captions, vocab_size=5000, max_length=None):
    # raw_captions: list of strings, contoh ["<start> a dog running <end>", ...]
    # Mengonversi teks menjadi sequence integer dan otomatis melakukan padding [cite: 163, 165]
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=max_length,
        standardize='lower_and_strip_punctuation'
    )
    
    vectorizer.adapt(raw_captions)
    
    vocab = vectorizer.get_vocabulary()
    word_to_index = {word: index for index, word in enumerate(vocab)}
    index_to_word = {index: word for index, word in enumerate(vocab)}
    
    return vectorizer, word_to_index, index_to_word

# Cara pakai
# vectorized_captions = vectorizer(raw_captions).numpy()