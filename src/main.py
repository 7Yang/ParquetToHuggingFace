from create_parquet import ParquetFileCreator
from upload_to_huggingface import HuggingFaceUploader
from download_from_huggingface import HuggingFaceDownloader

config_path = "./config/config.yaml"

def main():
    # Create an instance of the ParquetFileCreator class to create Parquet files
    parquet_creator = ParquetFileCreator(config_path)
    parquet_creator.create_parquet_files()

    # After creating Parquet files, upload them to Hugging Face
    hf_uploader     = HuggingFaceUploader(config_path)
    if hf_uploader.create_repo():
        hf_uploader.upload_files()

    # Test download from huggingface
    hf_downloader   = HuggingFaceDownloader(config_path)
    datasets        = hf_downloader.download()
    print(datasets)

if __name__ == "__main__":
    main()
