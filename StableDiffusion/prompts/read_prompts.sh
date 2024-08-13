!/bin/bash
# exec < processed_prompts.csv
# read header
# while IFS="," read -r line_nb prompt
# do
#    echo "Record is : $prompt"
# done 

# arr_record1=( $(tail -n +2 processed_prompts.csv | cut -d ',' -f1) )

# arr_record1=( $(cut -d "," -f1,3 processed_prompts.csv | tail -n +2))
# il faut faire line nb + 2 pour enlever le header
for PROMPT_LINE in 400 600 800 
do
    line=$(($PROMPT_LINE + 2))
    arr_record1=( $(head -n $line processed_prompts.csv | tail -n 1 | cut -d ',' -f2) )
    string="${arr_record1[@]}"
    char='"'
    output=$(echo "$string" | tr -d "$char")
    echo "Prompt: ${output}"
done