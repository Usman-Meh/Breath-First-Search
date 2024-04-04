# Mental Disorder Classifier

This project implements a classifier for predicting mental disorders based on a provided dataset. The classifier employs four different machine learning algorithms: K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Naive Bayes, and Decision Tree. Additionally, it includes visualization of the accuracy of each algorithm.

## Dataset
The dataset used for this project is included in the repository. It contains information about various mental disorders along with associated features. Each instance in the dataset represents a patient and includes attributes such as age, gender, family history, symptoms, and diagnosis.

## Algorithms
1. **K-Nearest Neighbors (KNN):** This algorithm classifies a data point based on the majority class of its nearest neighbors.
2. **Support Vector Machine (SVM):** SVM works by finding the hyperplane that best separates the classes in a high-dimensional space.
3. **Naive Bayes:** Naive Bayes is a probabilistic classifier that calculates the probability of each class given the input features.
4. **Decision Tree:** Decision trees recursively split the data into subsets based on features, aiming to maximize information gain at each split.

## Usage
To run the classifier, follow these steps:
1. Ensure Python and necessary libraries are installed (e.g., scikit-learn, matplotlib).
2. Clone this repository.
3. Execute the script, which will automatically load the dataset from the `data` directory.

## Visualization
The accuracy of each algorithm is visualized using a plot chart. This visualization helps in comparing the performance of different algorithms on the provided dataset.

## Note
- It's important to preprocess the dataset appropriately before training the classifier to handle missing values, normalize features, and encode categorical variables.
- Experiment with different algorithms and hyperparameters to find the best model for your dataset.
- Consider cross-validation for a more robust evaluation of the classifier's performance.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
