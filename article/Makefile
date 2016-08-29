pdf:
	pandoc --variable mainfont="Palatino" --variable sansfont="Century Gothic" --variable monofont="Consolas" --variable fontsize=12pt article.md --bibliography article.bib -o article.pdf
clean:
	rm article.pdf
rescience:
	pandoc --variable mainfont="Palatino" --standalone --filter ~/.cabal/bin/pandoc-crossref --template=rescience-template.tex --latex-engine=xelatex --biblatex --bibliography=article.bib -M "crossrefYaml=crossref.yaml" --output article.tex article.md
	lualatex article
	biber article
	lualatex article
	lualatex article
