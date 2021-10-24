import pandas as pd
import math
import numpy as np

class Leaf:
    def __init__(self, label):
        self.label = label

class Node:
    def __init__(self):
        self.left = None
        self.right = None
        self.split_point = None
        self.feature = None

    def print_node(self):
        print( "Feature", self.feature, "Split Point", self.split_point)

# Entropy calculation
def calc_entropy(data_labels):
    classes = set(data_labels)
    label_counts = data_labels.value_counts()
    tot_size = len(data_labels)

    entropy = 0
    for label in label_counts.keys():
        p_count = label_counts[label]/tot_size
        entropy += -(p_count * math.log(p_count, 2))
    return entropy
    # Gini impurity calculation
    '''
    impurity = 1
    for label in label_counts.keys():
        p_count = label_counts[label] / tot_size
        impurity -= p_count ** 2
    return impurity
    '''

# Partitioning data into left and right edges after splitting.
def partition(df, feature_col, threshold, data_labels):
    left_part = []
    right_part = []
    left_labels = []
    right_labels = []
    for i in range(df.shape[0]):
        if df[feature_col].iloc[i] < threshold:
            left_labels.append(data_labels.iloc[i])
            left_part.append(df.iloc[i])
        else:
            right_labels.append(data_labels.iloc[i])
            right_part.append(df.iloc[i])
    return pd.DataFrame(left_part), pd.DataFrame(right_part)

# Getting best split point based on information gain.
def best_split( train_df ):
    num_features = train_df.shape[1] - 1
    entropy_before_split = calc_entropy(train_df[4])

    best_gain = 0
    best_split_point = {}
    for feature_col in range(num_features):
        # Sorting feature-col
        df = train_df.sort_values(by=feature_col)
        data_labels_col = df[4]

        for i in range(len(data_labels_col)-1):
            if data_labels_col.iloc[i] == data_labels_col.iloc[i+1]:
                continue

            split_point = (df[feature_col].iloc[i] + df[feature_col].iloc[i+1])/2
            left_part, right_part = partition(df, feature_col, split_point, data_labels_col)
            left_part_len = len(left_part)
            right_part_len = len(right_part)

            left_entropy = right_entropy = 0
            if left_part_len != 0:
                left_entropy = calc_entropy(left_part[4])

            if right_part_len != 0:
                right_entropy = calc_entropy(right_part[4])
            # Information gain
            inf_gain = entropy_before_split - (left_part_len * left_entropy + right_part_len * right_entropy)/(left_part_len + right_part_len)

            if inf_gain > best_gain:
                best_gain = inf_gain
                best_split_point['split_point'] = split_point
                best_split_point['feature_col'] = feature_col
                best_split_point['left'] = left_part
                best_split_point['right'] = right_part

    return best_split_point

# Returning predicted label.
def check(row, node):
    if node is None:
        return

    if isinstance(node, Leaf):
        return node.label

    feature = node.feature
    split_point = node.split_point

    if row[feature] < split_point:
        return check(row, node.left)
    else:
        return check(row, node.right)

# Evaluating Accuracy
def evaluate(df, root):
    rows = len(df)
    sum = 0.0

    for row in range(rows):
        pred = check(df.iloc[row], root)
        if pred == (df.iloc[row])[4]:
            sum += 1
    return (sum*100)/rows

# Building full tree
def build_tree( train_df ):
    if train_df.empty:
        return None

    label_counts = train_df[4].value_counts()
    if len(label_counts) == 1:
        return Leaf(label_counts.keys()[0])

    split_point = best_split( train_df )

    node = Node()
    node.feature = split_point['feature_col']
    node.split_point = split_point['split_point']
    #print("\nBuilding left tree")
    node.left = build_tree(split_point['left'])
    node.right = build_tree(split_point['right'])

    return node

# Function to print tree. Solution tree is drawn by hand.
def print_tree(node):
    if node is None:
        return
    if isinstance(node, Leaf):
        print("Leaf Node with label\n", node.label)
        return
    print (node.feature, node.split_point)
    print("\n")
    print_tree(node.left)
    print_tree(node.right)

def main():
    train_df = pd.read_csv('set_a.csv', header=None)
    root = build_tree(train_df)
    #print_tree(root)
    acc_train = evaluate(train_df, root)
    print("Train accuracy", acc_train)

if __name__ == '__main__':
    main()
