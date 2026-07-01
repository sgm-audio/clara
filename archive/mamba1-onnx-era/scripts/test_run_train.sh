#!/usr/bin/env bash
# Test script to run inside distrobox
cd "/home/sgm/Github Repo's/clara"
HSA_OVERRIDE_GFX_VERSION=11.0.0 ./scripts/run_train.sh ~/datasets/musdb18hq/train --help 2>&1 | head -50
