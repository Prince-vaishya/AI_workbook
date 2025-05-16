from approvedimports import *

def make_xor_reliability_plot(train_x, train_y):
    """ Insert code below to complete this cell according to the instructions in the activity descriptor.
    Finally it should return the fig and axs objects of the plots created.

    Parameters:
    -----------
    train_x: numpy.ndarray
        feature values

    train_y: numpy array
        labels

    Returns:
    --------
    fig: matplotlib.figure.Figure
        figure object

    ax: matplotlib.axes.Axes
        axis
    """

    # ====> insert your code below here
    # Define a list of hidden layer sizes to test, ranging from 1 to 10 neurons
    hidden_layer_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Initialize a list to store the number of successful runs (100% accuracy) for each layer size
    success_counts = [0] * 10
    # Create a 2D list to store the number of epochs for each trial (10 trials per layer size)
    epoch_list = [[0 for _ in range(10)] for _ in range(10)]

    # Iterate through each hidden layer size to evaluate model performance
    for size_index in range(len(hidden_layer_sizes)):
        # Get the current number of neurons for this iteration
        current_size = hidden_layer_sizes[size_index]

        # Run 10 trials for each layer size to account for randomness
        for trial in range(10):
            # Initialize an MLPClassifier with specified parameters:
            # - Single hidden layer with 'current_size' neurons
            # - Maximum 1000 iterations for training
            # - Regularization parameter alpha set to 0.0001
            # - Stochastic gradient descent solver
            # - Learning rate of 0.1
            # - Random state set to trial number for reproducibility
            mlp_model = MLPClassifier(
                hidden_layer_sizes=(current_size,),
                max_iter=1000,
                alpha=0.0001,
                solver='sgd',
                learning_rate_init=0.1,
                random_state=trial
            )

            # Train the model on the provided training data
            mlp_model.fit(train_x, train_y)

            # Calculate the accuracy on the training data as a percentage
            accuracy = mlp_model.score(train_x, train_y) * 100

            # If the model achieves 100% accuracy, record the success
            if accuracy == 100:
                # Increment the success count for this layer size
                success_counts[size_index] = success_counts[size_index] + 1
                # Store the number of epochs taken to converge
                epoch_list[size_index][trial] = mlp_model.n_iter_

    # Initialize a list to store the average epochs for successful runs
    average_epochs = [0] * 10
    # Calculate the average number of epochs for each layer size
    for size_index in range(10):
        # Track total epochs and number of successful runs
        total_epochs = 0
        successful_runs = 0

        # Sum epochs for trials where the model succeeded (non-zero epochs)
        for trial in range(10):
            if epoch_list[size_index][trial] > 0:
                total_epochs = total_epochs + epoch_list[size_index][trial]
                successful_runs = successful_runs + 1

        # Compute the average epochs for successful runs
        if successful_runs > 0:
            average_epochs[size_index] = total_epochs / successful_runs
        else:
            # If no successful runs, assign a high value (1000) to indicate failure
            average_epochs[size_index] = 1000

    # Create a figure with two subplots for visualizing reliability and efficiency
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot success counts (reliability) vs. hidden layer sizes
    ax[0].plot(hidden_layer_sizes, success_counts, marker='o')
    ax[0].set_title("Reliability")
    ax[0].set_xlabel("Hidden Layer Width")
    ax[0].set_ylabel("Success Rate")
    ax[0].set_xticks(hidden_layer_sizes)

    # Plot average epochs (efficiency) vs. hidden layer sizes
    ax[1].plot(hidden_layer_sizes, average_epochs, marker='o')
    ax[1].set_title("Efficiency")
    ax[1].set_xlabel("Hidden Layer Width")
    ax[1].set_ylabel("Mean Epochs")
    ax[1].set_xticks(hidden_layer_sizes)

    # Adjust layout to prevent overlap of subplots
    plt.tight_layout()
    # raise NotImplementedError("Complete the function")
    # <==== insert your code above here

    return fig, ax

# Ensure required libraries are available for machine learning tasks
from approvedimports import *

# Define a class to compare different supervised learning algorithms
class MLComparisonWorkflow:
    """ Class to implement a comparison of supervised learning algorithms on a dataset """ 

    def __init__(self, datafilename: str, labelfilename: str):
        """ Initialize the workflow by loading feature and label data from specified files
        and setting up storage for models and their performance metrics.

        Assumes features are continuous and labels are categorical (encoded as integers).
        Both input files should have matching row counts, each row representing
        feature values and the corresponding label for a training example.
        """
        # Initialize dictionaries to store models and track best performers
        self.stored_models: dict = {"KNN": [], "DecisionTree": [], "MLP": []}
        self.best_model_index: dict = {"KNN": 0, "DecisionTree": 0, "MLP": 0}
        self.best_accuracy: dict = {"KNN": 0, "DecisionTree": 0, "MLP": 0}

        # Load feature and label data from files
        # ====> insert your code below here
        # Read feature data from the specified file into a NumPy array, assuming comma-separated values
        self.data_x = np.loadtxt(datafilename, delimiter=",")
        # Read label data from the specified file into a NumPy array, assuming comma-separated values
        self.data_y = np.loadtxt(labelfilename, delimiter=",")
        # <==== insert your code above here

    def preprocess(self):
        """ Preprocess the dataset by:
           - Splitting it into training and testing sets (70:30 ratio)
           - Normalizing feature values to a common scale
           - Creating one-hot encoded labels for MLP if there are more than two classes

           Uses a fixed random state (12345) for reproducibility in train-test splitting.
        """
        # ====> insert your code below here
        # Split data into training (70%) and testing (30%) sets, ensuring class distribution is preserved
        self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(
            self.data_x, self.data_y, test_size=0.3, random_state=12345, stratify=self.data_y
        )

        # Normalize features to scale them between 0 and 1 based on min/max values
        min_vals = []
        max_vals = []

        # Calculate minimum and maximum values for each feature across the entire dataset
        for i in range(self.data_x.shape[1]):
            feature_values = self.data_x[:, i]
            min_vals.append(min(feature_values))
            max_vals.append(max(feature_values))

        # Normalize training data by scaling each feature value
        train_norm = []
        for row in self.train_x:
            norm_row = []
            for i in range(len(row)):
                # Avoid division by zero; if min equals max, set normalized value to 0
                if max_vals[i] == min_vals[i]:
                    norm_row.append(0)
                else:
                    # Apply min-max normalization formula: (x - min) / (max - min)
                    norm_row.append((row[i] - min_vals[i]) / (max_vals[i] - min_vals[i]))
            train_norm.append(norm_row)

        # Normalize test data using the same min/max values for consistency
        test_norm = []
        for row in self.test_x:
            norm_row = []
            for i in range(len(row)):
                # Apply same normalization as training data
                if max_vals[i] == min_vals[i]:
                    norm_row.append(0)
                else:
                    norm_row.append((row[i] - min_vals[i]) / (max_vals[i] - min_vals[i]))
            test_norm.append(norm_row)

        # Convert normalized data to NumPy arrays for model compatibility
        self.train_x = np.array(train_norm)
        self.test_x = np.array(test_norm)

        # Handle label encoding for MLP, which may require one-hot encoding for multi-class problems
        unique_classes = list(set(self.data_y))
        num_classes = len(unique_classes)

        # Determine if one-hot encoding is needed (more than two classes)
        if num_classes > 2:
            # Create one-hot encoded labels for training data
            train_onehot = []
            for label in self.train_y:
                # Initialize a zero vector of length equal to number of classes
                onehot = [0] * num_classes
                # Set the index corresponding to the class to 1
                class_index = unique_classes.index(label)
                onehot[class_index] = 1
                train_onehot.append(onehot)

            # Create one-hot encoded labels for test data
            test_onehot = []
            for label in self.test_y:
                onehot = [0] * num_classes
                class_index = unique_classes.index(label)
                onehot[class_index] = 1
                test_onehot.append(onehot)

            # Store one-hot encoded labels as NumPy arrays
            self.train_y_onehot = np.array(train_onehot)
            self.test_y_onehot = np.array(test_onehot)
        else:
            # For binary classification, use original labels directly
            self.train_y_onehot = self.train_y
            self.test_y_onehot = self.test_y
        # <==== insert your code above here

    def run_comparison(self):
        """ Compare three supervised learning algorithms (KNN, DecisionTree, MLP) by:
        - Tuning hyperparameters for each algorithm
        - Training and evaluating models for each hyperparameter combination
        - Storing models and tracking the best performer for each algorithm

        Stores models in self.stored_models and tracks best accuracy/index for each algorithm.
        """
        # ====> insert your code below here
        # Tune K-Nearest Neighbors by testing different numbers of neighbors
        k_values = [1, 3, 5, 7, 9]
        for i, k in enumerate(k_values):
            # Initialize and train a KNN model with the current number of neighbors
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(self.train_x, self.train_y)
            # Store the trained model
            self.stored_models["KNN"].append(knn)

            # Evaluate model accuracy on the test set
            predictions = knn.predict(self.test_x)
            correct = 0
            for j in range(len(self.test_y)):
                if predictions[j] == self.test_y[j]:
                    correct += 1
            # Calculate accuracy as a percentage
            accuracy = (correct / len(self.test_y)) * 100

            # Update best KNN model if current accuracy is higher
            if accuracy > self.best_accuracy["KNN"]:
                self.best_accuracy["KNN"] = accuracy
                self.best_model_index["KNN"] = i

        # Tune Decision Tree by testing combinations of depth, split, and leaf parameters
        depths = [1, 3, 5]
        splits = [2, 5, 10]
        leafs = [1, 5, 10]

        dt_index = 0
        for depth in depths:
            for split in splits:
                for leaf in leafs:
                    # Initialize and train a Decision Tree with current hyperparameters
                    dt = DecisionTreeClassifier(
                        max_depth=depth,
                        min_samples_split=split,
                        min_samples_leaf=leaf,
                        random_state=12345
                    )
                    dt.fit(self.train_x, self.train_y)
                    # Store the trained model
                    self.stored_models["DecisionTree"].append(dt)

                    # Evaluate model accuracy on the test set
                    predictions = dt.predict(self.test_x)
                    correct = 0
                    for j in range(len(self.test_y)):
                        if predictions[j] == self.test_y[j]:
                            correct += 1
                    # Calculate accuracy as a percentage
                    accuracy = (correct / len(self.test_y)) * 100

                    # Update best Decision Tree model if current accuracy is higher
                    if accuracy > self.best_accuracy["DecisionTree"]:
                        self.best_accuracy["DecisionTree"] = accuracy
                        self.best_model_index["DecisionTree"] = dt_index

                    # Increment index to track model position in stored_models
                    dt_index += 1

        # Tune Multi-Layer Perceptron by testing layer sizes and activation functions
        first_layers = [2, 5, 10]
        second_layers = [0, 2, 5]
        activations = ["logistic", "relu"]

        mlp_index = 0
        for first in first_layers:
            for second in second_layers:
                for activation in activations:
                    # Configure layer sizes: single layer if second=0, else two layers
                    if second == 0:
                        layers = (first,)
                    else:
                        layers = (first, second)

                    # Initialize and train an MLP with current hyperparameters
                    mlp = MLPClassifier(
                        hidden_layer_sizes=layers,
                        activation=activation,
                        max_iter=1000,
                        random_state=12345
                    )
                    mlp.fit(self.train_x, self.train_y_onehot)
                    # Store the trained model
                    self.stored_models["MLP"].append(mlp)

                    # Evaluate model accuracy on the test set
                    predictions = mlp.predict(self.test_x)
                    correct = 0

                    # Handle multi-class classification with one-hot encoded labels
                    if len(set(self.data_y)) > 2:
                        for j in range(len(self.test_y)):
                            # Determine predicted class by finding index of maximum probability
                            pred_class = 0
                            max_val = predictions[j][0]
                            for k in range(1, len(predictions[j])):
                                if predictions[j][k] > max_val:
                                    max_val = predictions[j][k]
                                    pred_class = k

                            # Determine true class from one-hot encoded test labels
                            true_class = 0
                            max_val = self.test_y_onehot[j][0]
                            for k in range(1, len(self.test_y_onehot[j])):
                                if self.test_y_onehot[j][k] > max_val:
                                    max_val = self.test_y_onehot[j][k]
                                    true_class = k

                            # Increment correct count if prediction matches true class
                            if pred_class == true_class:
                                correct += 1
                    else:
                        # Handle binary classification directly
                        for j in range(len(self.test_y)):
                            if predictions[j] == self.test_y[j]:
                                correct += 1

                    # Calculate accuracy as a percentage
                    accuracy = (correct / len(self.test_y)) * 100

                    # Update best MLP model if current accuracy is higher
                    if accuracy > self.best_accuracy["MLP"]:
                        self.best_accuracy["MLP"] = accuracy
                        self.best_model_index["MLP"] = mlp_index

                    # Increment index to track model position in stored_models
                    mlp_index += 1
        # <==== insert your code above here

    def report_best(self):
        """ Identify and return the best-performing model across all algorithms.

        Returns
        -------
        accuracy: float
            Accuracy of the best-performing model
        algorithm: str
            Name of the algorithm ("KNN", "DecisionTree", or "MLP")
        model: fitted model
            The best-performing fitted model
        """
        # ====> insert your code below here
        # Initialize variables to track the best algorithm and its accuracy
        best_algo = ""
        best_acc = 0

        # Compare accuracy of best models from each algorithm
        algos = ["KNN", "DecisionTree", "MLP"]
        for algo in algos:
            if self.best_accuracy[algo] > best_acc:
                best_acc = self.best_accuracy[algo]
                best_algo = algo

        # Retrieve the best model using its stored index
        best_model = self.stored_models[best_algo][self.best_model_index[best_algo]]

        # Return the best accuracy, algorithm name, and model
        return best_acc, best_algo, best_model
        # <==== insert your code above here
