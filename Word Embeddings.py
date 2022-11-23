class UnigramFrequencies:
  def __init__(self, list_words=None):
    self.unigram_counts = Counter()
    for index, word_list in enumerate(list_words):
      for word  in word_list:
        words = word.split(" ")
        for token in words:
          self.unigram_counts[token]+=1
      
    self.word_to_index = {}
    for index, word in enumerate(self.unigram_counts.keys()):
      self.word_to_index[word] = index
      
    self.index_to_word = {}
    for index, word in enumerate(self.word_to_index.items()):
      self.index_to_word[index] = word
      
class SkipgramFrequencies:
  def __init__(self, word_lists, backward_window_size=2, forward_window_size=2):
    self.backward_window_size = backward_window_size
    self.forward_window_size = forward_window_size
    self.skipgram_counts = Counter()
    self.unigrams = UnigramFrequencies(word_lists)
    for index_token, word_list in enumerate(word_lists):
        for index, word in enumerate(word_list):
            context_window_start = max(0, index - self.backward_window_size)
            context_window_end = min(len(word_list) - 1, index + self.forward_window_size) + 1
            context = [context_idx for context_idx in range(context_window_start,context_window_end) if context_idx != index]
            for context_idx in context:
                skipgram = (word_list[index], word_list[context_idx])
                self.skipgram_counts[skipgram] += 1
    @property               
    def index_to_word(self):
    return self.unigrams.index_to_word
    @property
    def word_to_index(self):
    return self.unigrams.word_to_index
    
    
  def calculate_pairwise_frequency_matrix(skipgrams):
  row_vals = []
  col_vals = []
  matrix_values = []
  for (index_1, index_2), skipgram_count in skipgrams.skipgram_counts.items():
      row_vals.append(index_1)
      col_vals.append(index_2)
      matrix_values.append(skipgram_count)
  sparse_m = csr_matrix((matrix_values, (row_vals, col_vals)))
  return sparse_m
  
  # This code is from https://dustinstansbury.github.io/theclevermachine/info-theory-word-embeddings
  
  def calculate_pmi_matrix(skipgrams):
    frequency_matrix = calculate_pairwise_frequency_matrix(skipgrams)
    n_skipgrams = frequency_matrix.sum()
    word_sums = np.array(frequency_matrix.sum(axis=0)).flatten()
    context_sums = np.array(frequency_matrix.sum(axis=1)).flatten()
    row_idxs = []
    col_idxs = []
    matrix_values = []
 
    for (skipgram_word_idx, skipgram_context_idx), skipgram_count in skipgrams.skipgram_counts.items():
        join_probability = skipgram_count / n_skipgrams
        n_word = context_sums[skipgram_word_idx]
        p_word = n_word / n_skipgrams
        n_context = word_sums[skipgram_context_idx]
        p_context = n_context / n_skipgrams `
        pmi = np.log(join_probability / (p_word * p_context))
        
        row_idxs.append(skipgram_word_idx)
        col_idxs.append(skipgram_context_idx)
        matrix_values.append(pmi)
 
    return csr_matrix((matrix_values, (row_idxs, col_idxs)))

def calculate_word_vectors(matrix, dim=300):
    pmi_matrix = matrix
    svd = TruncatedSVD(n_components=dim, n_iter=50)
    left_vectors = svd.fit_transform(pmi_matrix)
    return left_vectors * np.sqrt(svd.singular_values_)


skipgram_frequencies = SkipgramFrequencies(tokenized_dataset)
pmi_matrix = calculate_pmi_matrix(skipgram_frequencies)
embeddings_matrix = calculate_word_vectors(pmi_matrix, dim=300)
print(embeddings_matrix.shape)


plt.figure(figsize=(12,5))
plt.spy(pmi_matrix[10000:15000, 10000:15000], markersize=2, color="green")
plt.title("A plot of a subset of a matrix showing the association between words(PMI Matrix)")


