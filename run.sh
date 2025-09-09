#!/bin/bash

docker run -it \
--gpus all \
-e DISPLAY=$DISPLAY \
-e SDL_VIDEODRIVER=x11 \
--net=host \
robomimic
