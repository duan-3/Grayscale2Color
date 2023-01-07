from fastapi import FastAPI, File, HTTPException
from PIL import Image

app = FastAPI()

@app.post("/resize")
async def resize_image(image: bytes = File(...)):
    try:
        # Open the image file and resize it
        image = Image.open(image)
        image = image.resize((256, 256))

        # Save the resized image to a temporary file
        temp_file = 'temp.jpg'
        image.save(temp_file)

        # Return the resized image file in the response
        return open(temp_file, 'rb')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))