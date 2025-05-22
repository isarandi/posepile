#!/bin/bash

# File containing the UUIDs
FILE="$DATA_ROOT/egoexo4d/takes.txt"

COUNTER=0
UUID_BATCH=()

while IFS= read -r line; do
  let COUNTER=COUNTER+1
  if (( COUNTER % 5 == 0 )); then
    UUID=$(echo "$line" | awk '{print $1}')
    UUID_BATCH+=("$UUID")
  fi
done < "$FILE"

if [ ${#UUID_BATCH[@]} -ne 0 ]; then
  UUIDs=$(IFS=' '; echo "${UUID_BATCH[*]}")
  egoexo -o $DATA_ROOT/egoexo4d/ --parts takes --uids $UUIDs --views exo -y
fi