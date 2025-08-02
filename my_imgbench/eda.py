import matplotlib.pyplot as plt
import numpy as np

def plot_class_distribution(dataset, out_path="class_distribution.png"):
    classes, counts = np.unique(dataset.targets, return_counts=True)
    plt.bar(range(len(classes)), counts)
    plt.xticks(range(len(classes)), dataset.classes)
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Class Distribution')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def show_sample_images(dataset, n_per_class=5, out_path="sample_images.png"):
    fig, axs = plt.subplots(len(dataset.classes), n_per_class, figsize=(12, 2*len(dataset.classes)))
    for i, class_idx in enumerate(range(len(dataset.classes))):
        idxs = [j for j, y in enumerate(dataset.targets) if y == class_idx][:n_per_class]
        for j, idx in enumerate(idxs):
            img, _ = dataset[idx]
            axs[i, j].imshow(np.transpose(img.numpy(), (1,2,0)))
            axs[i, j].axis('off')
            if j == 0:
                axs[i, j].set_ylabel(dataset.classes[class_idx])
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

