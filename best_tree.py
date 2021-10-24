import pandas as pd
import q1_full_tree
import matplotlib.pyplot as plt

# Builds tree for a given depth.
def build_tree( train_df, level, depth ):
    if train_df.empty:
        return None

    label_counts = train_df[4].value_counts()

    if level == depth:
        max_count = -1
        res_label = 0
        # Returns label with maximum count.
        for label in label_counts.keys():
            count = label_counts[label]
            if count > max_count:
                max_count = count
                res_label = label
        return q1_full_tree.Leaf(res_label)

    if len(label_counts) == 1:
        return q1_full_tree.Leaf(label_counts.keys()[0])

    split_point = q1_full_tree.best_split(train_df)

    node = q1_full_tree.Node()
    node.feature = split_point['feature_col']
    node.split_point = split_point['split_point']
    node.left = build_tree(split_point['left'], level+1, depth)
    node.right = build_tree(split_point['right'], level+1, depth)

    return node

def main():
    k = 10
    rows = 100
    set_size = int(rows / k)
    train_df = pd.read_csv('set_a.csv', header=None)

    best_acc_val = 0.0
    best_depth = 0

    depth_train_acc = {}
    depth_val_acc = {}

    # Choosing depth of tree
    for depth in range(0,12):
        print("\nBuilding Trees for depth", depth)
        i = 0
        tot_acc_val = 0.0
        tot_acc_train = 0.0

        # 10-cross validation
        for sets in range(k):
            if sets == 0:
                training_set = train_df.iloc[set_size:rows]
                validation_set = train_df.iloc[0:set_size]
            else:
                training_set = train_df.iloc[0:i]
                training_set = training_set.append(train_df.iloc[i+set_size:rows])
                validation_set = train_df.iloc[i:i+set_size]

            root = build_tree(training_set, 0, depth)
            acc_train = q1_full_tree.evaluate(training_set, root)
            tot_acc_train += acc_train
            #print(acc_train)

            acc_val = q1_full_tree.evaluate(validation_set, root)
            tot_acc_val += acc_val
            #print(acc_val)
            i += set_size

        if (tot_acc_val/10) > best_acc_val:
            best_acc_val = tot_acc_val/10
            best_depth = depth

        #print("Avg training acc for depth", depth, tot_acc_train/10)
        #print("Avg validation acc for depth", depth, tot_acc_val/10)
        #print('')
        depth_train_acc[depth] = tot_acc_train/10
        depth_val_acc[depth] = tot_acc_val/10

    # Graph plots for accuracies vs/depth. Also see them in directory graph_plots.
    lists = sorted( depth_train_acc.items())
    x, y = zip(*lists)
    plt.figure()
    plt.xlabel('Depth of Tree')
    plt.ylabel('Avg Training Accuracy')
    plt.title('Avg Training Accuracy vs Depth of Tree')
    plt.style.use('seaborn-whitegrid')
    plt.plot(x, y, '-o')

    lists = sorted( depth_val_acc.items())
    x, y = zip(*lists)
    plt.figure()
    plt.xlabel('Depth of Tree')
    plt.ylabel('Avg Validation Accuracy')
    plt.title('Avg Validation Accuracy vs Depth of Tree')
    plt.style.use('seaborn-whitegrid')
    plt.plot(x, y, '-o')

    plt.show()

    print("#############")
    print("Best Depth", best_depth)
    print("#############")

    # Building tree with best depth.
    root = build_tree(train_df, 0, best_depth)

    #q1_full_tree.print_tree(root)

    # Accuray on Set A
    acc_train = q1_full_tree.evaluate(train_df, root)
    print("Training Accuracy", acc_train)


if __name__ == '__main__':
    main()
