import asyncio
import websockets
import os
import json
import time
import hashlib

IMAGE_DIRECTORY = './images_t'
REPEAT_TIMES = 10
KEY = "12345"

async def send_images(websocket, path):
    if path != "/wsprocess":
        await websocket.close(reason="Invalid path")
        return  # Close connection if the path is not "/weprocess"

    try:
        start = time.time()
        received_key = await websocket.recv()
        print(f"Received key: {received_key}")
        if received_key != KEY:
            print("Invalid key received, closing connection.")
            await websocket.close(reason="Invalid key")
            return  # Exit the function if the key is incorrect

        for filename in os.listdir(IMAGE_DIRECTORY):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(IMAGE_DIRECTORY, filename)
                with open(image_path, 'rb') as image_file:
                    image_data = image_file.read()

                md5_hash = hashlib.md5(image_data).hexdigest()
                message_id = md5_hash.encode('utf-8')
                message = message_id + image_data

                for i in range(REPEAT_TIMES):
                    await websocket.send(message)

                    response = await websocket.recv()
                    response_data = json.loads(response)
                    print(response_data['id'])
                    print(response_data['ocr_result'])

                print(f"Received response for image: {filename}")
        end = time.time()
        print(f"Time taken: {end - start} seconds")
    except Exception as e:
        print(f"Error: {e}")

async def main():
    async with websockets.serve(send_images, "0.0.0.0", 8765):
        print("Server started on ws://0.0.0.0:8765")
        await asyncio.Future()  

if __name__ == "__main__":
    asyncio.run(main())
