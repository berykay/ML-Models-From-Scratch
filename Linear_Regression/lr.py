from typing import List

class LinearRegression:
    def __init__(self, learning_rate = 0.000005, epochs= 1000):
        # BIM = m1 * Height + m2 * Weight + b
        # z = m1 * x1 + m2 * x2 + b
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.m1 = 1
        self.m2 = 2
        self.b = 0
        self.error_history = []
        self.test_error_history = []
        self.accuracy_history = []

    def fit(self, x_train, y_train, z_train):
        num_samples = len(x_train)
        for i in range(self.epochs):
            m1_grad = 0
            m2_grad = 0
            b_grad = 0
            total_error = 0
            for j in range(num_samples):
                x1 = x_train[j]
                x2 = y_train[j]
                z_real = z_train[j]

                prediction = self.m1 * x1 + self.m2 * x2 + self.b
                error = (prediction - z_real)

                m1_grad += error * x1
                m2_grad += error * x2
                b_grad += error
                total_error += error**2

            m1_grad /= num_samples/2
            m2_grad /= num_samples/2
            b_grad /= num_samples/2

            self.m1 = self.m1 - self.learning_rate * m1_grad
            self.m2 = self.m2 - self.learning_rate * m2_grad
            self.b = self.b - self.learning_rate * b_grad

            epoch_error = total_error / num_samples
            self.error_history.append(epoch_error)
    
    def predict(self, x_test, y_test):
        num_samples = len(x_test)
        z_pred = []
        z_pred_round = []
        for i in range(num_samples):
            x1 = x_test[i]
            x2 = y_test[i]
            z = self.m1 * x1 + self.m2 * x2 + self.b          
            z_pred.append(z)
            z_pred_round.append(round(z))
            self.test_error_history.append((z - z_pred_round[i])**2)
        return z_pred, z_pred_round
    
    def accuracy(self, z_test, z_pred_round):
        self.accuracy_history = []
        num_samples = len(z_test)
        correct = 0
        for i in range(num_samples):
            if z_test[i] == z_pred_round[i]:
                correct += 1
            self.accuracy_history.append(correct / num_samples)
        return correct / num_samples
    
    def rsquared(self, z_test, z_pred):
        num_samples = len(z_test)
        z_mean = sum(z_test) / num_samples
        ss_tot = 0
        ss_res = 0
        for i in range(num_samples):
            ss_tot += (z_test[i] - z_mean)**2
            ss_res += (z_test[i] - z_pred[i])**2
        return 1 - (ss_res/ss_tot)


# if __name__ == '__main__':  
    # X, y = ...
    # X_train, X_test, y_train, y_test = ...

    # clf = DecisionTreeClassifier(max_depth=5)
    # clf.fit(X_train, y_train)
    # yhat = clf.predict(X_test)    
    