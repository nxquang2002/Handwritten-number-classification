import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import scipy.io as sci
from google.colab import drive

drive.mount('/content/drive')

#LOAD DATA FROM MNIST DATASET
mnist = sci.loadmat('/content/drive/My Drive/Machine learning project/MNIST Handwritten Number Classification/mnist-original.mat')
mnist_data = mnist["data"].T
mnist_label = mnist["label"].T

print(mnist_data.shape)
print(mnist_label.shape)
img_side_len = int(np.sqrt(mnist_data.shape[1]))

#FUNCTION TO SHOW EXAMPLE
def img_rand_example(data, label):
    idx = np.random.randint(data.shape[0])
    edx = data[idx].reshape(img_side_len, img_side_len)
    plt.imshow(edx)
    print("Accurate label: ", int(label[idx]))
    return idx

#=================================================================================================================
#FEATURE EXTRACTION PHASE
#Normalize data
def standardization(data):
    mean = np.mean(data, axis = 1, keepdims = True)
    devi = np.std(data, axis = 1, keepdims = True)
    std = (data - mean)/devi
    return std

mnist_data = standardization(mnist_data)

X_train, X_test, y_train, y_test = train_test_split(mnist_data, mnist_label, train_size = 0.8, random_state = 42)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
print("The shape of X_train is: ", X_train.shape)
print("The shape of X_test is: ", X_test.shape)
print("The shape of y_train is: ", y_train.shape)
print("The shape of y_test is: ", y_test.shape)

y_train_0 = (y_train == 0).astype(int)
y_train_1 = (y_train == 1).astype(int)
y_train_2 = (y_train == 2).astype(int)
y_train_3 = (y_train == 3).astype(int)
y_train_4 = (y_train == 4).astype(int)
y_train_5 = (y_train == 5).astype(int)
y_train_6 = (y_train == 6).astype(int)
y_train_7 = (y_train == 7).astype(int)
y_train_8 = (y_train == 8).astype(int)
y_train_9 = (y_train == 9).astype(int)

y_test_0 = (y_test == 0).astype(int)
y_test_1 = (y_test == 1).astype(int)
y_test_2 = (y_test == 2).astype(int)
y_test_3 = (y_test == 3).astype(int)
y_test_4 = (y_test == 4).astype(int)
y_test_5 = (y_test == 5).astype(int)
y_test_6 = (y_test == 6).astype(int)
y_test_7 = (y_test == 7).astype(int)
y_test_8 = (y_test == 8).astype(int)
y_test_9 = (y_test == 9).astype(int)

#====================================================================================================================
#TRAINNING MODEL
def initialize(num_feature):
    W = np.zeros((num_feature,1))
    B = 0
    return W, B

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def forward_backward_prop(X, Y, W, B):
    m = X.shape[0];
    dW = np.zeros((W.shape[0],1))
    dB = 0
    
    z = np.dot(X, W) + B
    Yhat = sigmoid(z)
    J = -(1/m)*(np.dot(Y.T, np.log(Yhat)) + np.dot((1-Y).T, np.log(1-Yhat))).reshape(1,1)
    dW = (1/m)*np.dot(X.T, Yhat - Y)
    dB = (1/m)*np.sum(Yhat - Y)
    return J, dW, dB

def predict(X, W, B):
    Yhat_prob = sigmoid(np.dot(X, W) + B)
    Yhat = np.round_(Yhat_prob).astype(int)
    return Yhat, Yhat_prob

def gradient_descent(X, Y, W, B, epochs = 1000, learning_rate = 0.01, verbose = 50):
    cost_hist = []
    
    for epoch in range(epochs):
        J,dW, dB = forward_backward_prop(X,Y,W,B)
        W = W - learning_rate*dW
        B = B - learning_rate*dB
        cost_hist.append(J)
        
        Yhat, _ = predict(X,W,B)
        if(np.sqrt(np.mean(Yhat - Y)**2) < 10e-4):
            break
        if(epoch % verbose == 0):
            print("Epoch {} loss {}".format(epoch, J))
    
    return cost_hist, W, B

#LOGISTIC REGRESSION MODEL
def LogReg_model(X_train, Y_train, X_test, Y_test):
    num_feature = X_train.shape[1]
    W,B = initialize(num_feature)
    
    epochs = 1000
    learning_rate = 0.01
    verbose = 50
    cost_hist, W, B = gradient_descent(X_train, Y_train, W, B, epochs, learning_rate, verbose)
    
    Y_hat_train, _ = predict(X_train, W, B)
    Y_hat_test, _ = predict(X_test, W, B)
    
    train_accuracy = accuracy_score(Y_train, Y_hat_train)
    test_accuracy = accuracy_score(Y_test, Y_hat_test)
    cfs_matrix = confusion_matrix(Y_test, Y_hat_test)
    
    model = {
        "weights": W,
        "bias" : B,
        "train_accuracy" : train_accuracy,
        "test_accuracy" : test_accuracy,
        "confusion_matrix: ": cfs_matrix,
        "cost function" : cost_hist
    }
    
    return model 


#model_1 = LogReg_model(X_train, y_train_1, X_test, y_test_1)
#print("Training complete!")

#cost = np.concatenate(model_1['cost function']).ravel().tolist()
#plt.plot(list(range(len(cost))),cost)
#plt.title('Evolution of the cost by iteration')
#plt.xlabel('Iteration')
#plt.ylabel('Cost');
#plt.show()

def random_check(model, datum, label):
    W = model["weights"]
    B = model["bias"]
    
    Yhat, _ = predict(datum, W, B)
    if Yhat == 1:
        pred_label = label
    else:
        pred_label = 'not' + label
    return pred_label


Y_train_list = [y_train_0, y_train_1, y_train_2, y_train_3, y_train_4, y_train_5, y_train_6, y_train_7, y_train_8, y_train_9]
Y_test_list = [y_test_0,y_test_1,y_test_2,y_test_3, y_test_4, y_test_5, y_test_6, y_test_7, y_test_8, y_test_9]
model_list = []
    
for i in range(0,10):
    print("Trainning model {} for digit {}:".format(i,i))
    model = LogReg_model(X_train, Y_train_list[i], X_test, Y_test_list[i])
    print("Trainning complete!")
    print('Test accuracy: ',  model["test_accuracy"])
    print('-' * 50)
    model_list.append(model)

#ONE-VS-ALL REGRESSION
def one_hot_coding(model, X): 
    classProbabilities = np.zeros((X.shape[0], 10))

    for i in range (10):
      W = model[i]['weights']
      B = model[i]['bias'] 
      _, Yhat_prob = predict(X, W, B)
      classProbabilities[:,i] = Yhat_prob.T
    max_prob = np.amax(classProbabilities, axis = 1, keepdims=True)
    final_prob = (classProbabilities == max_prob).astype(int)
    label = []
    for i in range (X.shape[0]) :
      lbl = np.where(final_prob[i, :] == 1)
      label.append(lbl)
    return label

#=======================================================================================================================
#SAVING MODEL LIST
import pickle
pkl_filename = "/content/drive/My Drive/Machine learning project/MNIST Handwritten Number Classification/Model_list.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model_list, file)

#ONE HOT CODING, PREDICT THE TEST SET
label = one_hot_coding(model_list, X_test)  

#RANDOM CHECK
idx = img_rand_example(X_test, y_test)
print("Predicted label: ", label[idx])