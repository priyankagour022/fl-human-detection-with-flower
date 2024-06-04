import matplotlib.pyplot as plt
from pathlib import Path

def plot_learning_graph(history, save_path):

    if hasattr(history, 'metrics_centralized'):
        metrics = history.metrics_centralized
    else:
        print("Error: 'metrics_centralized' not found in history")
        return

    # Extracting accuracy values from metrics
    accuracy_values = [value for _, value in metrics['accuracy']]
    rounds = range(len(accuracy_values))

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, accuracy_values, label='Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per Round')
    plt.legend()
    plt.tight_layout()
    
    # Save the plot as an image file
    graph_path = Path(save_path) / 'learning_graph.png'
    plt.savefig(graph_path)
    print(f"Learning graph saved to {graph_path}")
    plt.show()
