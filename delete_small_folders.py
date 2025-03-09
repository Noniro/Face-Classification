import os
import shutil
import random


def manage_image_folders(root_path, min_photos=9, max_photos=200, extensions=('.jpg', '.jpeg', '.png')):
    """
    Manages image folders:
    1. Deletes folders that contain fewer than min_photos image files.
    2. Reduces folders with more than max_photos to exactly max_photos.

    Parameters:
    root_path (str): Path to the main directory containing person folders
    min_photos (int): Minimum number of photos required to keep a folder
    max_photos (int): Maximum number of photos to keep in a folder
    extensions (tuple): File extensions to consider as photos

    Returns:
    dict: Statistics about the operation
    """
    if not os.path.exists(root_path):
        print(f"Error: The path {root_path} does not exist.")
        return

    stats = {
        "total_folders": 0,
        "deleted_folders": 0,
        "preserved_folders": 0,
        "reduced_folders": 0,
        "total_photos_deleted": 0
    }

    # Lists for operation tracking
    folders_to_delete = []
    folders_to_reduce = []

    # First pass: identify folders to delete or reduce
    for person_folder in os.listdir(root_path):
        person_path = os.path.join(root_path, person_folder)

        # Skip if not a directory
        if not os.path.isdir(person_path):
            continue

        stats["total_folders"] += 1

        # Get all image files in the folder
        image_files = [
            file for file in os.listdir(person_path)
            if os.path.isfile(os.path.join(person_path, file))
               and file.lower().endswith(extensions)
        ]
        photo_count = len(image_files)

        if photo_count < min_photos:
            folders_to_delete.append({
                "folder": person_folder,
                "path": person_path,
                "photo_count": photo_count
            })
        elif photo_count > max_photos:
            folders_to_reduce.append({
                "folder": person_folder,
                "path": person_path,
                "photo_count": photo_count,
                "image_files": image_files
            })

    # Show confirmation for deletion
    if folders_to_delete:
        print(f"Found {len(folders_to_delete)} folders with less than {min_photos} photos:")
        for i, folder_info in enumerate(folders_to_delete, 1):
            print(f"{i}. {folder_info['folder']} - {folder_info['photo_count']} photos")

        confirm_delete = input("\nAre you sure you want to delete these folders? (yes/no): ")
        if confirm_delete.lower() not in ["yes", "y"]:
            print("Deletion operation cancelled.")
        else:
            # Perform deletion
            for folder_info in folders_to_delete:
                try:
                    shutil.rmtree(folder_info["path"])
                    stats["deleted_folders"] += 1
                    stats["total_photos_deleted"] += folder_info["photo_count"]
                    print(f"Deleted: {folder_info['folder']}")
                except Exception as e:
                    print(f"Error deleting {folder_info['folder']}: {e}")
    else:
        print(f"No folders with less than {min_photos} photos were found.")

    # Show confirmation for reduction
    if folders_to_reduce:
        print(f"\nFound {len(folders_to_reduce)} folders with more than {max_photos} photos:")
        for i, folder_info in enumerate(folders_to_reduce, 1):
            print(f"{i}. {folder_info['folder']} - {folder_info['photo_count']} photos")

        confirm_reduce = input(f"\nAre you sure you want to reduce these folders to {max_photos} photos? (yes/no): ")
        if confirm_reduce.lower() not in ["yes", "y"]:
            print("Reduction operation cancelled.")
        else:
            # Perform reduction
            for folder_info in folders_to_reduce:
                try:
                    folder_path = folder_info["path"]
                    image_files = folder_info["image_files"]
                    files_to_remove = len(image_files) - max_photos

                    # Randomly select files to remove
                    files_to_delete = random.sample(image_files, files_to_remove)

                    # Delete the selected files
                    for file_name in files_to_delete:
                        file_path = os.path.join(folder_path, file_name)
                        os.remove(file_path)
                        stats["total_photos_deleted"] += 1

                    stats["reduced_folders"] += 1
                    print(f"Reduced: {folder_info['folder']} (removed {files_to_remove} photos)")
                except Exception as e:
                    print(f"Error reducing {folder_info['folder']}: {e}")
    else:
        print(f"No folders with more than {max_photos} photos were found.")

    stats["preserved_folders"] = stats["total_folders"] - stats["deleted_folders"]

    # Print summary
    print("\nOperation completed:")
    print(f"- Total folders processed: {stats['total_folders']}")
    print(f"- Folders deleted (< {min_photos} photos): {stats['deleted_folders']}")
    print(f"- Folders reduced (> {max_photos} photos): {stats['reduced_folders']}")
    print(f"- Folders preserved without changes: {stats['preserved_folders'] - stats['reduced_folders']}")
    print(f"- Total photos deleted: {stats['total_photos_deleted']}")

    return stats


if __name__ == "__main__":
    # Replace this with your actual path
    dataset_path = r"C:\Users\97253\PycharmProjects\Face-Classification\Data-peopleFaces"

    # Call the function
    manage_image_folders(dataset_path, min_photos=9, max_photos=40)
    print("\nDone.")
