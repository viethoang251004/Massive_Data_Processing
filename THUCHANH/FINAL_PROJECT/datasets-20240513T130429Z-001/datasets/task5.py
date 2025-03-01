# from pyspark.sql import SparkSession
# from pyspark.ml.feature import VectorAssembler
# from pyspark.ml.classification import MultilayerPerceptronClassifier, RandomForestClassifier, LinearSVC
# from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# import matplotlib.pyplot as plt

# class DataLoader:
#     def __init__(self, file_path):
#         self.spark = SparkSession.builder.appName("MNISTClassifier").getOrCreate()
#         self.data = self.spark.read.csv(file_path, header=True, inferSchema=True)
    
#     def preprocess(self):
#         feature_columns = self.data.columns[:-1]
#         assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
#         self.data = assembler.transform(self.data).select("features", "label")
#         self.train_data, self.test_data = self.data.randomSplit([0.8, 0.2], seed=1234)
#         return self.train_data, self.test_data

# class Classifier:
#     def __init__(self, model):
#         self.model = model
#         self.evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    
#     def train(self, train_data):
#         self.model = self.model.fit(train_data)
    
#     def evaluate(self, data):
#         predictions = self.model.transform(data)
#         accuracy = self.evaluator.evaluate(predictions)
#         return accuracy

# class MultiLayerPerceptron(Classifier):
#     def __init__(self):
#         layers = [784, 128, 64, 10]  # Cấu trúc mạng MLP
#         mlp = MultilayerPerceptronClassifier(layers=layers, seed=1234)
#         super().__init__(mlp)

# class RandomForest(Classifier):
#     def __init__(self):
#         rf = RandomForestClassifier(numTrees=100, seed=1234)
#         super().__init__(rf)

# class LinearSVM(Classifier):
#     def __init__(self):
#         lsvc = LinearSVC(maxIter=10, regParam=0.1)
#         super().__init__(lsvc)

# def plot_accuracies(train_accuracies, test_accuracies, labels):
#     x = range(len(labels))
#     fig, ax = plt.subplots()
#     ax.bar(x, train_accuracies, width=0.4, label='Train Accuracy', align='center')
#     ax.bar(x, test_accuracies, width=0.4, label='Test Accuracy', align='edge')
#     ax.set_xlabel('Model')
#     ax.set_ylabel('Accuracy')
#     ax.set_title('Comparison of Model Accuracies')
#     ax.set_xticks(x)
#     ax.set_xticklabels(labels)
#     ax.legend()
#     plt.show()

# if __name__ == "__main__":
#     data_loader = DataLoader("mnist_mini.csv")
#     train_data, test_data = data_loader.preprocess()
    
#     classifiers = [MultiLayerPerceptron(), RandomForest(), LinearSVM()]
#     labels = ["MLP", "Random Forest", "Linear SVM"]
    
#     train_accuracies = []
#     test_accuracies = []
    
#     for classifier in classifiers:
#         classifier.train(train_data)
#         train_accuracy = classifier.evaluate(train_data)
#         test_accuracy = classifier.evaluate(test_data)
#         train_accuracies.append(train_accuracy)
#         test_accuracies.append(test_accuracy)
    
#     plot_accuracies(train_accuracies, test_accuracies, labels)



from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier, RandomForestClassifier, LinearSVC, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import matplotlib.pyplot as plt
import numpy as np

# Data Loader Class
class DataLoader:
    def __init__(self, file_path):
        conf = SparkConf() \
            .setAppName("MNISTClassifier") \
            .set("spark.python.worker.reuse", "true") \
            .set("spark.worker.timeout", "600") \
            .set("spark.network.timeout", "600s") \
            .set("spark.executor.heartbeatInterval", "60s") \
            .set("spark.executor.memory", "2g") \
            .set("spark.executor.cores", "2") \
            .set("spark.driver.memory", "2g")

        self.spark = SparkSession.builder.config(conf=conf).getOrCreate()
        self.data = self.spark.read.csv(file_path, header=True, inferSchema=True)

    def preprocess(self):
        feature_columns = self.data.columns[:-1]
        label_column = self.data.columns[-1]
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        self.data = assembler.transform(self.data).withColumnRenamed(label_column, "label").select("features", "label")
        self.train_data, self.test_data = self.data.randomSplit([0.8, 0.2], seed=1234)
        return self.train_data, self.test_data

# Base Classifier Class
class Classifier:
    def __init__(self, model):
        self.model = model
        self.evaluator = MulticlassClassificationEvaluator(metricName="accuracy")

    def train(self, train_data):
        self.model = self.model.fit(train_data)

    def evaluate(self, data):
        predictions = self.model.transform(data)
        accuracy = self.evaluator.evaluate(predictions)
        return accuracy

# Multi-layer Perceptron Classifier
class MultiLayerPerceptron(Classifier):
    def __init__(self):
        layers = [784, 128, 64, 10]
        mlp = MultilayerPerceptronClassifier(layers=layers, seed=1234)
        super().__init__(mlp)

# Random Forest Classifier
class RandomForest(Classifier):
    def __init__(self):
        rf = RandomForestClassifier(numTrees=100, seed=1234)
        super().__init__(rf)

# Linear SVM Classifier (handled via OneVsRest)
class LinearSVM(Classifier):
    def __init__(self):
        lsvc = LinearSVC(maxIter=10, regParam=0.1)
        ovr = OneVsRest(classifier=lsvc)
        super().__init__(ovr)

# Workflow to train, evaluate, and plot results
class SparkClassificationWorkflow:
    def __init__(self):
        self.model_names = ["MLP", "Random Forest", "Linear SVM"]
        self.train_accuracies = []
        self.test_accuracies = []

    def load_data(self, input_path):
        data_loader = DataLoader(input_path)
        return data_loader.preprocess()

    def train_and_evaluate(self, classifier, train_data, test_data):
        classifier.train(train_data)
        train_accuracy = classifier.evaluate(train_data)
        test_accuracy = classifier.evaluate(test_data)
        return train_accuracy, test_accuracy

    def run_workflow(self, train_data, test_data):
        classifiers = [MultiLayerPerceptron(), RandomForest(), LinearSVM()]
        for classifier in classifiers:
            train_acc, test_acc = self.train_and_evaluate(classifier, train_data, test_data)
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
        self.plot_accuracies()

    def plot_accuracies(self):
        x = np.arange(len(self.model_names))
        fig, ax = plt.subplots()
        ax.bar(x - 0.2, self.train_accuracies, width=0.4, label='Train Accuracy')
        ax.bar(x + 0.2, self.test_accuracies, width=0.4, label='Test Accuracy')
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy')
        ax.set_title('Comparison of Model Accuracies')
        ax.set_xticks(x)
        ax.set_xticklabels(self.model_names)
        ax.legend()
        plt.show()

# Main Execution
if __name__ == "__main__":
    workflow = SparkClassificationWorkflow()
    train_data, test_data = workflow.load_data("mnist_mini.csv")
    workflow.run_workflow(train_data, test_data)
