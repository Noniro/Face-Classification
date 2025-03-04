import os
import shutil


def delete_small_folders(root_path, min_photos=9, extensions=('.jpg', '.jpeg', '.png')):
    """
    Delete folders that contain fewer than min_photos image files.

    Parameters:
    root_path (str): Path to the main directory containing person folders
    min_photos (int): Minimum number of photos required to keep a folder
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
        "total_photos_deleted": 0
    }

    # List of folders to delete (for confirmation)
    folders_to_delete = []

    # First pass: identify folders to delete
    for person_folder in os.listdir(root_path):
        person_path = os.path.join(root_path, person_folder)

        # Skip if not a directory
        if not os.path.isdir(person_path):
            continue

        stats["total_folders"] += 1

        # Count photos in the folder
        photo_count = sum(1 for file in os.listdir(person_path)
                          if os.path.isfile(os.path.join(person_path, file))
                          and file.lower().endswith(extensions))

        if photo_count < min_photos:
            folders_to_delete.append({
                "folder": person_folder,
                "path": person_path,
                "photo_count": photo_count
            })

    # Show confirmation
    if folders_to_delete:
        print(f"Found {len(folders_to_delete)} folders with less than {min_photos} photos:")
        for i, folder_info in enumerate(folders_to_delete, 1):
            print(f"{i}. {folder_info['folder']} - {folder_info['photo_count']} photos")

        confirm = input("\nAre you sure you want to delete these folders? (yes/no): ")
        if confirm.lower() not in ["yes", "y"]:
            print("Operation cancelled.")
            return stats

        # Second pass: delete confirmed folders
        for folder_info in folders_to_delete:
            try:
                shutil.rmtree(folder_info["path"])
                stats["deleted_folders"] += 1
                stats["total_photos_deleted"] += folder_info["photo_count"]
                print(f"Deleted: {folder_info['folder']}")
            except Exception as e:
                print(f"Error deleting {folder_info['folder']}: {e}")

        stats["preserved_folders"] = stats["total_folders"] - stats["deleted_folders"]

        # Print summary
        print("\nOperation completed:")
        print(f"- Total folders processed: {stats['total_folders']}")
        print(f"- Folders deleted: {stats['deleted_folders']}")
        print(f"- Folders preserved: {stats['preserved_folders']}")
        print(f"- Total photos deleted: {stats['total_photos_deleted']}")
    else:
        print(f"No folders with less than {min_photos} photos were found.")

    return stats


if __name__ == "__main__":
    # Replace this with your actual path
    dataset_path = r"C:\Users\HP\Downloads\archive\lfw-funneled\lfw_funneled"

    # Call the function
    delete_small_folders(dataset_path, min_photos=9)
    print ("galgay")