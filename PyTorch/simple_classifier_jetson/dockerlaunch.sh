#!/bin/bash
docker run -it --rm --runtime nvidia --network host -v $(pwd):/workspace testimage