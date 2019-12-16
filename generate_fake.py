import numpy as np 
import pickle

# data = torch.randn(5, 10, 3, 10, 64, 64)
chunked_data = np.random.normal(size=(5, 10, 3, 10, 64, 64))
data = np.random.normal(size=(5, 100, 3, 64, 64))
labels = np.ones((5,))

pickle.dump(chunked_data, open("./_data/chunked_data.p", "wb"))
pickle.dump(data, open("./_data/data.p", "wb"))
pickle.dump(labels, open("./_data/labels.p", "wb"))
