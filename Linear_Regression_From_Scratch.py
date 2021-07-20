#%%
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

X, y = datasets.load_boston(return_X_y=True)

def test_train_split(X, y, test_size=0.2):
    idx = 0
    length_of_X = len(X)
    y_test = []
    X_test = []
    
    while  idx < length_of_X*test_size:
        random_number_gen = np.random.randint(low=0, high=len(X))
        y_test.append(y[random_number_gen])
        X_test.append(X[random_number_gen])
        X = np.delete(X, random_number_gen, axis=0)
        y = np.delete(y, random_number_gen, axis=0)
        idx += 1
    
    return X, np.array(X_test), y, np.array(y_test)

X_train, X_test, y_train, y_test = test_train_split(X, y)
X_train, X_val, y_train, y_val = test_train_split(X_train, y_train)

#normalise dataset
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train = (X_train - X_mean) / X_std
X_val = (X_val - X_mean) / X_std
X_test = (X_test - X_mean) / X_std


class DataLoader:
    """
    Class to create mini-batch dataset for a model
    """
    def __init__(self, X, y, return_arg=True):
        self.batch(X, y)

    def batch(self, X, y, batch_size=16):
        self.batches = []
        idx = 0
        while idx < len(X):
            batch = (X[idx:idx+batch_size], y[idx:idx+batch_size])
            self.batches.append(batch)
            idx += batch_size
    
    def __getitem__(self, idx):
        return self.batches[idx]

    def __len__(self):
        return len(self.batches)
        



class LinearRegression:
    """
    Class implementing a linear regression
    """
    def __init__(self, n_features):
        # initialise random weight and bias
        self.W = np.random.randn(n_features)
        self.b = np.random.rand()
        
    def fit(self, X, y, epochs=15):
        # setup a learning rate
        lr = 0.01
        loss_per_epoch = []
        dataloader = DataLoader(X, y)
        for epoch in range(epochs):
            loss = []

            for X_batch, y_batch in dataloader:
                y_hat = self.predict(X_batch)
                loss_per_batch = self._get_mean_squuared_error_loss(y_hat, y_batch)
                loss.append(loss_per_batch)
                grad_W, grad_b = self._compute_grads(X_batch, y_batch)
                self.W -= lr * grad_W
                self.b -= lr * grad_b           
            #print out the loss for each epoch
            #print(f"epoch {epoch+1} loss: {np.mean(loss)}")
            loss_per_epoch.append(np.mean(loss))
        plt.plot(loss_per_epoch)
        plt.show()
        #draw the loss for each epoch



   


    def predict(self, X):
        return np.matmul(X, self.W) + self.b


    def _get_mean_squuared_error_loss(self, y_hat, y):
        return np.mean((y_hat - y)**2)
        

    def _compute_grads(self, X_batch, y_batch):
        y_hat = self.predict(X_batch)
        grad_b = 2 * np.mean(y_hat - y_batch)

        gradient_individuals = []
        for i in range(len(X_batch)):
            grad_i = 2 * (y_hat[i] - y_batch[i]) * X_batch[i]
            gradient_individuals.append(grad_i)        
        grad_W = np.mean(gradient_individuals, axis=0)

        return grad_W, grad_b
        


linear_model = LinearRegression(n_features=X.shape[1])
linear_model.fit(X_train, y_train)





# %%



