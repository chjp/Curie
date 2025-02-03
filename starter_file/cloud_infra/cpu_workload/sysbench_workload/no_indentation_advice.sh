# The following code block is CORRECT. Do this:
for i in {1..3}; do
    cat <<EOT > file$i.txt
This is file $i.
EOT
done

## The following code block is INCORRECT. Do not do this:
for i in {1..3}; do
    cat <<EOT > file$i.txt
    This is file $i.
    EOT  # This whitespace before `EOT` will cause an issue
done


## The correct version has no indentation for the heredoc block and the closing EOT. The incorrect block will fail because of indentation.