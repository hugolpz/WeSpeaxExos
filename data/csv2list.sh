# Converts csv with `word,occurences,otherparameters` into `# word` and `word	òccurences` 
# Usage: $ bash script <filenames>

mkdir -p ./frequency-sorted-hash ./frequency-sorted-count
for filename in "$@";
do
    [ -e "$filename" ] || continue
    echo "Processing: $(basename "$filename" .csv).csv "
    cat ./input/$(basename "$filename" .csv).csv | tail -n +2 | sed -E 's/("([^"]*)")?,/\2\t/g' | sort -k 2,2 -n -r | sed '$ {/^$/d;}' | cut -d$'\t' -f1 | sed -E 's/^/# /g' > ./frequency-sorted-hash/$(basename "$filename" .csv).txt
    cat ./input/$(basename "$filename" .csv).csv | tail -n +2 | sed -E 's/("([^"]*)")?,/\2\t/g' | sort -k 2,2 -n -r | sed '$ {/^$/d;}' | cut -d$'\t' -f1,2  > ./frequency-sorted-count/$(basename "$filename" .csv).csv
done
