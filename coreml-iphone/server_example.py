import asyncio
import websockets
import os
import json
import time


IMAGE_DIRECTORY = './images'
repeat_times = 90
KEY = "1234576"

async def send_images(websocket, path):
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
				for i in range(repeat_times):
					with open(image_path, 'rb') as image_file:
						image_data = image_file.read()
					await websocket.send(image_data)
					# print(f"Sent image: {filename}")
					# Wait for the response embedding
					response = await websocket.recv()
					response_data = json.loads(response)
					print(response_data)
					
				print(f"Received response for image: {filename}")
		end = time.time()
		print(f"Time taken: {end - start} seconds")
	except Exception as e:
		print(f"Error: {e}")

async def main():
    async with websockets.serve(send_images, "0.0.0.0", 8765):
        print("Server started on ws://10.16.50.133:8765")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(main())
