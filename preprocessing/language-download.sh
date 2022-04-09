mkdir -p vendor
mkdir -p build
cd vendor
for language in python java javascript c-sharp
do
	git clone https://github.com/tree-sitter/tree-sitter-${language}
done