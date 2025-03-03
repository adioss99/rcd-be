import os
from dotenv import load_dotenv
from imagekitio import ImageKit
from imagekitio.models.UploadFileRequestOptions import UploadFileRequestOptions
from imagekitio.models.DeleteFolderRequestOptions import DeleteFolderRequestOptions

load_dotenv()

# Get the values from environment variables
private_key = os.getenv('PRIVATE_KEY')
public_key = os.getenv('PUBLIC_KEY')
url_endpoint = os.getenv('URL_ENDPOINT')

# Initialize ImageKit with the environment variables
imagekit = ImageKit(
    private_key=private_key,
    public_key=public_key,
    url_endpoint=url_endpoint
)

def upload(file_path, filename):
    options = UploadFileRequestOptions(
        folder='/cnnxai/',
        response_fields=["is_private_file", "tags", 'embedded_metadata', 'custom_metadata'],
        tags=["CNN", "XAI"]
    )
    
    with open(file_path, 'rb') as file:
        upload = imagekit.upload_file(
            file=file,
            file_name=filename,
            options=options,
        )
    return upload.response_metadata.raw

def delete_image(id):
    file_id = str(id)
    result = imagekit.delete_file(file_id=file_id)
    return result.response_metadata.raw

def delete_folder():
    delete_folder = imagekit.delete_folder(options=DeleteFolderRequestOptions(folder_path="/cnnxai"))
    return delete_folder