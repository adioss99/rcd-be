from fastapi.responses import JSONResponse
from fastapi import UploadFile

import os
import uuid
from PIL import Image
from io import BytesIO

from core.lib.imagekit import delete_folder, upload, delete_image
from core.model.model import get_heatmap, save_and_display_gradcam

# Code to delete the image with the same path
def delete_image_file(image_path):
    try:
        os.remove(image_path)
        return {"message": "Image deleted successfully"}
    except FileNotFoundError:
        return {"error": "File not found"}
    except Exception as e:
        return {"error": str(e)}

def compress_image(arg, quality=60):
  original_image = Image.open(BytesIO(arg))
  
  # Resize image
  max_size = (500, 500)
  original_image.thumbnail(max_size, Image.Resampling.LANCZOS)
  
  with BytesIO() as buffer:
    original_image.save(buffer, format="JPEG", quality=quality)  # Compress (quality 0-100)
    compressed_contents = buffer.getvalue()  # Get compressed image bytes

  return compressed_contents

async def predict(image: UploadFile): 
  try:
    if not image.content_type.startswith("image/"):
      return {
        "success": False,
        "message": "File format not supported. Please upload an image file."
        }
    
    image.filename = f"{uuid.uuid4().hex[:6]}.jpg"
    contents = await image.read()
    
    image_path = f"./core/image/{image.filename}"
    # compress image
    compressed_contents = compress_image(contents)
    
    # Write compressed image to disk
    with open(image_path, "wb") as f:
        f.write(compressed_contents)
  
    result = get_heatmap(image_path)
    visualized_path = save_and_display_gradcam(image_path, result[0], cam_path=f'./core/image/viz_{image.filename}.jpg')
    
    upload_cnn = upload(image_path, 'cnn')
    upload_xai = upload(visualized_path, 'xai')
  
    delete_image_file(image_path)
    delete_image_file(visualized_path)
    
    return JSONResponse({
      "success" : True,
      "message": "Predict success", 
      'response': {
        "class": result[1][0], 
        "confidence": result[1][1],
        "cnn_img": { 
          "img_id":upload_cnn['fileId'],
          "name":upload_cnn["name"],
          "url": upload_cnn['url'],
          },
        "xai_img": { 
          "img_id":upload_xai['fileId'],
          "name":upload_cnn["name"],
          "url": upload_xai['url'],
          },
        }
      
      })
  except Exception as e:
    return JSONResponse({
      'success': False,
      'message': "Empty file. Please upload an image file."
      })
  
def delete(request):
  try:
    param = str(request.path_params['img_id'])
    res = delete_image(param)
    return JSONResponse({'message': 'delete success','response' : res})
  except Exception as e:
    print(e)
    return JSONResponse({"message": "data not found","e":str(e)})
  
def remove_folder():
  try:
    delete_folder()
    print('message : cron executed')
  except Exception as e:
    print({"message": "data not found","e":str(e)})
