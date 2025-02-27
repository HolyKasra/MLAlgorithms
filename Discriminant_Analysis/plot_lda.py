import matplotlib.pyplot as plt

def plot_linear_discriminants(X_lda, y, title, figsize=(4,4), ax=None):
    if ax is None:
        plt.figure(figsize=figsize)
        ax  = plt.subplot(111)
    label_dict = {0: 'Setosa', 1: 'Versicolor', 2:'Virginica'}

    for label, marker, color in zip(range(3),('^', 's', 'o'),('blue', 'red', 'green')):
        plt.scatter(-X_lda[y==label, 0], X_lda[y==label, 1], marker=marker, color = color, label = label_dict[label])

    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title(title)
        # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",  
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)    

    plt.grid()
    plt.tight_layout()
    plt.show()