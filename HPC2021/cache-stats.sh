#!/bin/sh
perf stat -B -e task-clock,cycles,cache-references,cache-misses,L1-dcache-loads,L1-dcache-load-misses,L1-dcache-stores,L1-dcache-store-misses "$@"
