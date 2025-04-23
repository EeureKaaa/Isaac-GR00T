from PIL import Image
import torch
import logging
import asyncio
import json
import numpy as np
import websockets
from openpi_client import msgpack_numpy
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import Gr00tPolicy

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint directory.",
        default="/home/wangxianhao/data/project/reasoning/Isaac-GR00T/checkpoints/first_try/checkpoint-10000",
    )
    parser.add_argument(
        "--embodiment_tag",
        type=str,
        help="The embodiment tag for the model.",
        default="new_embodiment",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        help="The name of the data config to use.",
        choices=list(DATA_CONFIG_MAP.keys()),
        default="custom_coin",
    )
    parser.add_argument(
        "--denoising_steps",
        type=int,
        help="Number of denoising steps.",
        default=10,
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Port number for the server.",
        default=8000,
    )
    parser.add_argument(
        "--host",
        type=str,
        help="Host address for the server.",
        default="0.0.0.0",
    )
    return parser.parse_args()


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load Gr00t inference model


def load_model(args):
    logger.info("Loading Gr00t inference model...")

    # Check if model path exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path {model_path} does not exist")

    # load data config
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()

    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    logger.info("GR00T model loaded successfully")
    return policy

# Process observation and prompt to generate actions


async def process_observation(policy, observation, prompt):
    '''
    For IsaacGR00T model, the image must expend to video(1 frame)
    and the state must match with config
    '''

    # Extract image from observation
    if 'video.base_view' in observation and 'video.wrist_view' in observation and 'state.joint_state' in observation:
        # Convert numpy array to PIL Image if needed
        base_image_array = observation['video.base_view']
        left_image_array = observation['video.left_view']
        wrist_image_array = observation['video.wrist_view']
        joint_state = observation['state.joint_state']

        # Ensure images are numpy arrays with correct shape
        if not isinstance(base_image_array, np.ndarray):
            raise ValueError(
                f"Ego image must be a numpy array, got {type(base_image_array)}")

        if not isinstance(wrist_image_array, np.ndarray):
            raise ValueError(
                f"Wrist image must be a numpy array, got {type(wrist_image_array)}")

        if not isinstance(joint_state, np.ndarray):
            if isinstance(joint_state, list):
                joint_state = np.array(joint_state, dtype=np.float32)
            else:
                raise ValueError(
                    f"Joint state must be a numpy array or list, got {type(joint_state)}")

        if base_image_array.shape[-3:-1] != (256, 256):
            logger.info("Resizing ego image to 256x256")
            # Convert to PIL Image for resizing
            if len(base_image_array.shape) == 4:  # (1, H, W, C)
                resized_images = []
                for img in base_image_array:
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    resized_img = pil_img.resize((256, 256), Image.BILINEAR)
                    resized_images.append(np.array(resized_img))
                base_image_array = np.stack(resized_images)
            else:  # (H, W, C)
                pil_img = Image.fromarray(base_image_array.astype(np.uint8))
                resized_img = pil_img.resize((256, 256), Image.BILINEAR)
                base_image_array = np.array(resized_img)
        if left_image_array.shape[-3:-1] != (256, 256):
            logger.info("Resizing left image to 256x256")
            # Convert to PIL Image for resizing
            if len(left_image_array.shape) == 4:  # (1, H, W, C)
                resized_images = []
                for img in left_image_array:
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    resized_img = pil_img.resize((256, 256), Image.BILINEAR)
                    resized_images.append(np.array(resized_img))
                left_image_array = np.stack(resized_images)
            else:  # (H, W, C)
                pil_img = Image.fromarray(left_image_array.astype(np.uint8))
                resized_img = pil_img.resize((256, 256), Image.BILINEAR)
                left_image_array = np.array(resized_img)

        if wrist_image_array.shape[-3:-1] != (256, 256):
            logger.info("Resizing wrist image to 256x256")
            # Convert to PIL Image for resizing
            if len(wrist_image_array.shape) == 4:  # (1, H, W, C)
                resized_images = []
                for img in wrist_image_array:
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    resized_img = pil_img.resize((256, 256), Image.BILINEAR)
                    resized_images.append(np.array(resized_img))
                wrist_image_array = np.stack(resized_images)
            else:  # (H, W, C)
                pil_img = Image.fromarray(wrist_image_array.astype(np.uint8))
                resized_img = pil_img.resize((256, 256), Image.BILINEAR)
                wrist_image_array = np.array(resized_img)

        # Set the task description for the model
        policy.task_description = prompt

        # Add batch dimension if needed
        if len(base_image_array.shape) == 3:  # (H, W, C)
            base_image_array = np.expand_dims(
                base_image_array, axis=0)  # (1, H, W, C)
        if len(left_image_array.shape) == 3:  # (H, W, C)
            left_image_array = np.expand_dims(
                left_image_array, axis=0)  # (1, H, W, C)
        if len(wrist_image_array.shape) == 3:  # (H, W, C)
            wrist_image_array = np.expand_dims(
                wrist_image_array, axis=0)  # (1, H, W, C)
        if len(joint_state.shape) == 1:  # (D,)
            joint_state = np.expand_dims(joint_state, axis=0)  # (1, D)

        # map the obs to gr00t-compatibility format
        obs = {
            "video.base_view": base_image_array,
            "video.left_view": left_image_array,
            "video.wrist_view": wrist_image_array,
            "state.joint_state": joint_state,
            "annotation.human.action.task_description": [prompt],
        }
        logger.info(
            f"Processing observation with shapes: base_view={base_image_array.shape}, left_view={left_image_array.shape}, wrist_view={wrist_image_array.shape}, joint_state={joint_state.shape}")
        # Step the model with the observation to get actions
        action_chunk = policy.get_action(obs)['action.ee_action']

        logger.info(f"Generated action chunk shape: {action_chunk.shape}")
        return {
            'actions': action_chunk,
            'status': 'success'
        }
    else:
        missing_keys = []
        if 'video.base_view' not in observation:
            missing_keys.append('video.base_view')
        if 'video.wrist_view' not in observation:
            missing_keys.append('video.wrist_view')
        if 'state.joint_state' not in observation:
            missing_keys.append('state.joint_state')
        if 'annotation.human.action.task_description' not in observation:
            missing_keys.append('annotation.human.action.task_description')
        error_msg = f"Observation missing required keys: {missing_keys}"
        logger.error(error_msg)
        return {'error': error_msg, 'status': 'error'}


# WebSocket server handler
async def websocket_handler(websocket, policy):
    # Send metadata to client
    metadata = {
        'server_name': 'Gr00t Action Server',
        'version': '1.0',
        'capabilities': ['action_generation']
    }
    await websocket.send(msgpack_numpy.packb(metadata))

    logger.info("Client connected. Metadata sent.")

    try:
        async for message in websocket:
            try:
                # Unpack the message
                data = msgpack_numpy.unpackb(message)
                print('---------------------------------------------')
                logger.info(f"Received data with keys: {data.keys()}")

                # Extract image and prompt directly from the data
                # breakpoint()
                base_image = data.get('base_image')
                left_image = data.get('left_image')
                wrist_image = data.get('wrist_image')
                joint_state = data.get('joint_state')
                inspect_list = [base_image, left_image,
                                wrist_image, joint_state]
                prompt = data.get('prompt', '')

                logger.info(
                    f"Processing with prompt: '{prompt}' and image shape: {base_image.shape if base_image is not None else 'None'}")
                if max([x is None for x in inspect_list]):
                    raise ValueError('the value of inspect_list is wrong')
                # Create observation with the image for process_observation
                observation = {
                    'video.base_view': base_image,
                    'video.left_view': left_image,
                    'video.wrist_view': wrist_image,
                    'state.joint_state': joint_state
                }

                # Process the observation and generate actions
                result = await process_observation(policy, observation, prompt)

                # Send back the result
                await websocket.send(msgpack_numpy.packb(result))
                logger.info("Sent action response to client")

            except Exception as e:
                error_msg = f"Error processing message: {str(e)}"
                logger.error(error_msg)
                await websocket.send(msgpack_numpy.packb({'error': error_msg, 'status': 'error'}))
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")

# Start the WebSocket server


async def start_server(policy, host='0.0.0.0', port=8000):
    server = await websockets.serve(
        lambda ws: websocket_handler(ws, policy),
        host,
        port,
        max_size=None,  # No limit on message size
        compression=None  # Disable compression for performance
    )

    logger.info(f"WebSocket server started at ws://{host}:{port}")

    return server

# Main function


async def main():
    global policy

    args = parse_args()
    # Load the Gr00t model
    policy = load_model(args)

    # Start the WebSocket server
    server = await start_server(policy, args.host, args.port)

    # Keep the server running
    await asyncio.Future()

# Run the server
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
