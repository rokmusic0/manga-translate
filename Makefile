clean:
	fd --glob 'main.*' --exclude '*.tex' --exclude '*.pdf' report -X rm -v {}
