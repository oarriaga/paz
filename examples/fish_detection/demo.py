import os
from Deepfish_data_processing import (
    extract_compresed_file,
    process_class_data,
    get_file,
)
from paz.backend.from_yolo_dataloader import get_data_PAZ_formate


if __name__ == "__main__":
    downloaded_file_local_path = get_file(
        file_url="https://drive.google.com/file/d/10Pr4lLeSGTfkjA40ReGSC8H3a9onfMZ0/view",
        output_filename="fish_dataset_.zip",
    )
    extract_compresed_file(
        downloaded_file_local_path,
    )

    process_class_data(
        raw_data_path=os.path.join(
            os.path.dirname(downloaded_file_local_path), "Deepfish"
        ),
        output_path=r"./Data/processed/yolo_dataset",
    )

    images_directory_full_path = r"./Data/processed/yolo_dataset/images/train"
    labels_directory_full_path = r"./Data/processed/yolo_dataset/labels/train"

    data = get_data_PAZ_formate(
        images_directory_full_path, labels_directory_full_path, normalize=False
    )
    print(f"Number of data entries: {len(data)}")
    print(f"First data entry: {data[0]}")

    print("First data entry image path: ", data[0]["image"])
    print("First data entry box data: ", data[0]["boxes"])
