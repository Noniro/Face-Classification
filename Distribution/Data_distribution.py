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
    # Get all person folders
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Path not found: {dataset_path}")

    person_folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

    # Count images per person
    counts = {}
    for person in person_folders:
        person_path = os.path.join(dataset_path, person)
        image_files = [f for f in os.listdir(person_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
        counts[person] = len(image_files)

    # Convert to Counter for easier manipulation
    counts = Counter(counts)

    # Create dataframe for easier plotting
    df = pd.DataFrame({
        'Person': list(counts.keys()),
        'Image Count': list(counts.values())
    })

    # Sort by count
    df = df.sort_values('Image Count', ascending=False)

    # Define the specific ranges you requested
    ranges = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100), (100, 150), (150, 200), (200, float('inf'))]
    range_labels = ['0-20', '20-40', '40-60', '60-80', '80-100', "100-150", "150-200", '200+']

    # Count people in each range
    range_counts = []
    for i, (low, high) in enumerate(ranges):
        count = len(df[(df['Image Count'] >= low) & (df['Image Count'] <= high)])
        range_counts.append((range_labels[i], count))
        print(f"Number of people with {low}-{high} images: {count}")

    # Count total images in each range
    range_image_totals = []
    for i, (low, high) in enumerate(ranges):
        # Sum images for people in this range
        total_images = df[(df['Image Count'] >= low) & (df['Image Count'] <= high)]['Image Count'].sum()
        range_image_totals.append((range_labels[i], total_images))
        print(f"Total images in range {low}-{high}: {total_images}")

    # Plot range distribution (number of people)
    plt.figure(figsize=(12, 6))
    range_df = pd.DataFrame(range_counts, columns=['Range', 'Number of People'])
    sns.barplot(x='Range', y='Number of People', data=range_df)
    plt.title('Distribution of People by Number of Images')
    plt.xlabel('Image Count Range')
    plt.ylabel('Number of People')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('people_range_distribution.png')
    plt.show()

    # Plot range distribution (total images)
    plt.figure(figsize=(12, 6))
    range_images_df = pd.DataFrame(range_image_totals, columns=['Range', 'Total Images'])
    sns.barplot(x='Range', y='Total Images', data=range_images_df)
    plt.title('Distribution of Total Images by Range')
    plt.xlabel('Image Count Range')
    plt.ylabel('Total Images')
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('images_range_distribution.png')
    plt.show()

    # Print some statistics
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
    # Try several path options until one works
    possible_paths = ["../Data-peopleFaces"]



    found_path = False
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found valid path: {path}")
            dataset_path = path
            found_path = True
            break

    if not found_path:
        print("No valid path found. Please enter the full path:")
        dataset_path = input()
        if not os.path.exists(dataset_path):
            print("Path still invalid. Program will exit.")
            exit(1)

    counts = analyze_image_distribution(dataset_path)
