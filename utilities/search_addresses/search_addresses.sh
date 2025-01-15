#!/bin/bash

search_dir="ray_results/"
input_file="input.txt"
output_file="address_results.txt"


while IFS= read -r line
do
    value=$(echo "$line" | cut -d'=' -f2)
    value_name=$(echo "$line" | cut -d'=' -f1)
    echo "Searching for $value"
    result=$(find "$search_dir" -type d -name "*$value*")

    if [ ! -z "$result" ]; then
        echo "$value_name=$result" >> "$output_file"
    else
        echo "No results for $value" >> "$output_file"
    fi
done < "$input_file"

echo "Search completed. Results are in $output_file"