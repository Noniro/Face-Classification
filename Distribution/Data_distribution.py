import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import seaborn as sns

def analyze_image_distribution(dataset_path):
    """
    Analyzes and visualizes the distribution of images across different people folders.

    Parameters:
    - dataset_path: str, path to the dataset directory.

    Returns:
    - counts: Counter object with counts per person
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Path not found: {dataset_path}")

    person_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    counts = {}
    for person in person_folders:
        person_path = os.path.join(dataset_path, person)
        image_files = [f for f in os.listdir(person_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        counts[person] = len(image_files)

    counts = Counter(counts)

    df = pd.DataFrame({
        'Person': list(counts.keys()),
        'Image Count': list(counts.values())
    })

    df = df.sort_values('Image Count', ascending=False)

    ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 150), (150, 200), (200, float('inf'))]
    range_labels = ['0-20', '20-40', '40-60', '60-80', '80-100', "100-150", "150-200", '200+']

    range_counts = []
    for i, (low, high) in enumerate(ranges):
        count = len(df[(df['Image Count'] >= low) & (df['Image Count'] <= high)])
        range_counts.append((range_labels[i], count))
        print(f"Number of people with {low}-{high} images: {count}")

    range_df = pd.DataFrame(range_counts, columns=['Range', 'Number of People'])

    plt.figure(figsize=(12, 6))
    sns.barplot(x='Range', y='Number of People', data=range_df)
    plt.title('Number of People per Image Count Range')
    plt.xlabel('Image Count Range')
    plt.ylabel('Number of People')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('people_range_distribution.png')
    plt.show()

    total_images = sum(counts.values())
    total_people = len(counts)
    print(f"\nTotal images: {total_images}")
    print(f"Total people: {total_people}")
    print(f"Average images per person: {total_images / total_people:.2f}")
    print(f"Median images per person: {np.median(list(counts.values()))}")
    print(f"Min images: {min(counts.values())}")
    print(f"Max images: {max(counts.values())}")

    return counts


if __name__ == "__main__":
    path = "../Data-peopleFaces"
    counts = analyze_image_distribution(path)
