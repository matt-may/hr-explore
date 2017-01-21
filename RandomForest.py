import csv
import numpy as np
import math

class RandomForest(object):
    class __DecisionTree(object):
        def __init__(self, m = math.sqrt, max_depth = 10, min_for_split = 2):
            self.m = m
            self.max_depth = max_depth
            self.min_for_split = min_for_split

        def learn(self, X, y):
            self.X = X
            self.y = y

            # Compute the number of sub-features to split on.
            num_features = X.shape[1]
            num_sub_features = int(self.m(num_features))

            # Find the indices of the sub-features.
            indices = np.random.choice(num_features, size=num_sub_features,
                                       replace=False)

            # Build the decision tree and assign it to self.tree.
            self.tree = self.build_tree(X, y, indices)

        def classify(self, test_instance):
            """
            Returns predicted label for a single instance.

            """

            node = self.tree

            # While the node is a Node instance (not an integer classification),
            # traverse the tree, checking whether the test instance's value for
            # the feature is less than or equal to the threshold.
            # If yes, select the left branch.
            # If no, select the right branch.
            while isinstance(node, Node):
                if test_instance[node.fi] <= node.thresh:
                    node = node.b_1
                else:
                    node = node.b_2

            # The leaf node will contain the classification.
            klass = node

            # Return the class.
            return klass

        def build_tree(self, X, y, indices, depth = 0):
            # If any of our stopping conditions are met,
            if self.gini(y) == 0.0 or \
               self.max_depth == depth or \
               len(y) < self.min_for_split:
                # Return the most common value.
                return self.mode(y)

            # Retrieve the feature index and threshold for the best split.
            fi, thresh = self.best_split(X, y, indices)

            # Perform the split, splitting samples and labels into two groups.
            X_1, y_1, X_2, y_2 = self.split(X, y, fi, thresh)

            # If we're empty on either side, return the most common value in y.
            if y_1.shape[0] == 0 or y_2.shape[0] == 0:
                return self.mode(y)

            # Create branches.
            new_depth = depth + 1
            b_1 = self.build_tree(X_1, y_1, indices, new_depth)
            b_2 = self.build_tree(X_2, y_2, indices, new_depth)

            return Node(fi, thresh, b_1, b_2)

        def best_split(self, X, y, indices):
            """
            Finds the best split for the given feature indices. Returns the
            feature index and threshold.

            """

            # Initialize our gain, feature index, and threshold values.
            gain, fi, thresh = 0, 0, 0

            # For each feature index,
            for i in indices:
                # Sort the values for the feature.
                vals = np.sort(np.unique(X[:, i]))

                # For every row,
                for v in xrange(len(vals) - 1):
                    # Compute a threshold.
                    new_thresh = (vals[v] + vals[v+1]) / 2

                    # Perform a split base on the threshold.
                    X_1, y_1, X_2, y_2 = self.split(X, y, i, new_thresh)

                    # Compute the new gain value.
                    new_gain = self.gini_gain(y, y_1, y_2)

                    # If the new gain exceeds the current gain, update the
                    # gain, feature index, and threshold.
                    if new_gain > gain:
                        gain, fi, thresh = new_gain, i, new_thresh

            return fi, thresh

        def split(self, X, y, fi, thresh):
            """
            Splits both features and labels into two groups each, based on a
            defined threshold.

            """

            # Retrieve all rows for the feature that are less than or equal to
            # the threshold.
            less_than = np.where(X[:,fi] <= thresh)

            # Retrieve all rows for the feature that are greater than the
            # threshold.
            greater_than = np.where(X[:,fi] > thresh)

            # Features and labels less than or equal to the threshold.
            X_1 = X[less_than]
            y_1 = y[less_than]

            # Features and labels greater than the threshold.
            X_2 = X[greater_than]
            y_2 = y[greater_than]

            return X_1, y_1, X_2, y_2

        def mode(self, y):
            """
            Returns the most common value in a set of labels y.

            """

            return np.bincount(y).argmax()

        def gini_gain(self, y, left, right):
            """
            Computes the Gini gain of splitting a set of labels into two sets,
                left and right.

            """

            def weighted_gini(part, num_labs):
                return (len(part) / num_labs) * self.gini(part)

            num_labs = float(len(y))
            weighted_sum = weighted_gini(left, num_labs) + \
                           weighted_gini(right, num_labs)

            return self.gini(y) - weighted_sum

        def gini(self, y):
            """
            Computes the Gini index for a set of labels y.

            """

            num_y = float(len(y))
            counts = np.bincount(y)
            return 1 - np.sum([(count / num_y)**2 for count in counts])

    def __init__(self, num_trees = 10, m = math.sqrt, max_depth = 10,
                 min_for_split = 2, bootstrap = 0.9):
        """
        Creates a new random forest.

        Args:
            num_trees: Integer. Number of trees to grow in constructing the
                forest.
            m: Function. Expression returning the number of features to test
                for each split. The function is passed the total number of
                features in the data.
            max_depth: Integer. Maximum depth that each tree will be grown.
            min_for_split: Integer. Minimum number of samples required for a
                new split to happen.
            bootstrap: Float. Percentage of data to select as a subset for
                growing each tree.

        """

        print("Initializing %d trees." % num_trees)

        self.decision_trees = []

        for t in xrange(num_trees):
            self.decision_trees.append(self.__DecisionTree(m=m, max_depth=max_depth,
                                                           min_for_split=min_for_split))

        self.bootstrap = bootstrap

    def fit(self, X, y):
        # Subset for bootstrapping.
        subset = int(self.bootstrap * len(y))

        # For every decision tree,
        for tree in self.decision_trees:
            # Shuffle the data.
            X, y = RandomForest.shuffle(X, y)

            # Retrieve the subset of the data to train the tree on.
            subset_X, subset_y = X[:subset], y[:subset]

            # Train the tree.
            tree.learn(subset_X, subset_y)

    def predict(self, X):
        y = np.array([], dtype = int)

        for instance in X:
            votes = np.array([decision_tree.classify(instance)
                              for decision_tree in self.decision_trees])

            counts = np.bincount(votes)

            y = np.append(y, np.argmax(counts))

        return y

    @staticmethod
    def shuffle(list_a, list_b):
        """
        Shuffles two lists, maintaining index relationships between them. The
        two lists should be the same length.

        Returns the shuffled lists.

        """

        assert len(list_a) == len(list_b)
        perm = np.random.permutation(len(list_a))
        return list_a[perm], list_b[perm]

class Node(object):
    """ Node abstraction for the decision tree. """

    def __init__(self, fi, thresh, b_1, b_2):
        """
        Creates a new node in the decision tree.

        Args:
          fi     : Integer. Feature index.
          thresh : Float. Threshold for the node.
          b_1    : Node. Left branch.
          b_2    : Node. Right branch.

        """

        self.fi = fi
        self.thresh = thresh
        self.b_1 = b_1
        self.b_2 = b_2

def main():
    X, y = [], []

    # Load dataset.
    with open('hr_dataset.csv') as f:
        first_line = f.readline()
        fieldnames = first_line.split(',')

        data_fields = [
          'DistanceFromHome',
          'Education',
          'EnvironmentSatisfaction',
          'HourlyRate',
          'JobInvolvement',
          'JobLevel',
          'JobSatisfaction',
          'NumCompaniesWorked',
          'PercentSalaryHike',
          'RelationshipSatisfaction',
          'StockOptionLevel',
          'TotalWorkingYears',
          'TrainingTimesLastYear',
          'WorkLifeBalance',
          'YearsAtCompany',
          'YearsInCurrentRole',
          'YearsSinceLastPromotion'
        ]

        for line in csv.DictReader(f, fieldnames=fieldnames):
            X.append([line[k] for k in data_fields])
            y.append(line['PerformanceRating'])

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)

    # Shuffle the data.
    X, y = RandomForest.shuffle(X, y)

    # Number of folds for cross-validation.
    K = 10

    # Initialize a left bound, and right bound, that will advance for cross-
    # validation.
    lbound = 0
    bound_size = X.shape[0] / K
    rbound = lbound + bound_size

    # Create a container for the accuracies across cross-validation.
    accuracies = []

    # Perform K-fold cross validation.
    for i in xrange(K):
        # Prepare a training set.
        X_train = np.concatenate((X[:lbound,:], X[rbound:,:]))
        y_train = np.concatenate((y[:lbound], y[rbound:]))

        # Prepare a test set.
        X_test = X[lbound:rbound,:]
        y_test = y[lbound:rbound]

        # Initialize according to your implementation
        randomForest = RandomForest(10)

        # Fit the classifier.
        randomForest.fit(X_train, y_train)

        # Predict results of the test set.
        y_predicted = randomForest.predict(X_test)

        # Determine our successes, and failures.
        results = [prediction == truth for prediction, truth in zip(y_predicted, y_test)]

        # Compute accuracy.
        accuracy = float(results.count(True)) / float(len(results))
        accuracies.append(accuracy)

        print "Accuracy: %.4f" % accuracy

        # Increment the boundaries.
        lbound += bound_size
        rbound += bound_size

    print("Final accuracy from %d-fold cross-validation: %.4f" % (K, np.average(accuracies)))