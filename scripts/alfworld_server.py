import numpy as np, yaml, os, cv2, asyncio, requests
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path
from IPython import embed
import uuid, json
from pickle import loads, dumps
from base64 import b64decode, b64encode


__this_folder__ = Path(__file__).parent.absolute()
app = FastAPI()
worker_id = str(uuid.uuid4())[:8]
semaphore = None
visual_config = __this_folder__.parent / "configs/base_config_visual.yaml"
text_config = __this_folder__.parent / "configs/base_config.yaml"
configs = {"visual": visual_config, "text": text_config}
env = None


def release_semaphore():
    global semaphore
    semaphore.release()


def acquire_worker_semaphore():
    global semaphore
    if semaphore is None:
        semaphore = asyncio.Semaphore(1)
    return semaphore.acquire()


@app.post("/set_environment")
async def set_environment(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    assert params["env_type"] in ["visual"]
    
    print(params)
    config_filename = configs[params["env_type"]]
    print(config_filename)
    with open(config_filename) as reader:
        config = yaml.safe_load(reader)
    env_type = config['env']['type']  # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'
    print("Environment: {}".format(env_type))
    
    global env
    env = getattr(environment, env_type)(config, train_eval='eval_out_of_distribution')
    env = env.init_env(batch_size=1)
    release_semaphore()


@app.post("/reset")
async def reset(request: Request):
    global env
    params = await request.json()
    await acquire_worker_semaphore()
    obs, infos = env.reset(**params)
    image = env.get_image()[0]
    release_semaphore()
    ret = ([obs, image], image)
    ret_str = b64encode(dumps(ret))
    return ret_str


@app.post("/step")
async def step(request: Request):
    global env
    params = await request.json()
    await acquire_worker_semaphore()
    obs, scores, dones, infos = env.step(**params)
    image = env.get_image()[0]
    release_semaphore()
    
    ret = ([obs, image], scores, dones, infos)
    ret_str = b64encode(dumps(ret))
    return ret_str


if __name__ == "__main__":
 
    
    
    # print(__this_folder__)
    # app.run()
    # print(config_file)
    # print(env_type)
    # exit(0)
    # env.step(dict(action='Initialize', gridSize=0.5, fieldOfView='110', visibilityDistance='4'))
    """
    admissible_commands = infos['admissible_commands'][0]
    print(obs[0])
    print("Action", admissible_commands)
    cnt = 0
    while True:
        frame = env.get_image()[0][..., ::-1]
        cnt += 1
        print(frame.shape)
        # cv2.imshow('image', frame)
        cv2.imwrite(f"images/{cnt}.png", frame)
        
        random_actions = [np.random.choice(admissible_commands)]

        # step
        obs, scores, dones, infos = env.step(random_actions)
        
        # action = input("Action: ").strip()
        # obs, scores, dones, infos = env.step(action)
        print(obs)
        if cnt >= 10:
            exit(0)
        
    print(frame[0].shape)
    print(obs)
    embed()
    exit(0)
    
    data_path = __this_folder__.parent / "data/json_2.1.1/valid_unseen"
    print(data_path)
    print(len(get_json_files(data_path)))
    exit(0)
    
    
    env.reset()
    embed()
    exit(0)
    
    # setup environment
    env = getattr(environment, env_type)(config, train_eval='eval_out_of_distribution')
    env = env.init_env(batch_size=1)

    # interact
    obs, info = env.reset()
    print(obs)
    while True:
        # get random actions from admissible 'valid' commands (not available for AlfredThorEnv)
        admissible_commands = list(info['admissible_commands']) # note: BUTLER generates commands word-by-word without using admissible_commands
        random_actions = [np.random.choice(admissible_commands[0])]

        # step
        obs, scores, dones, infos = env.step(random_actions)
        print("Action: {}, Obs: {}".format(random_actions[0], obs[0]))

    """
