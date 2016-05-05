pdf:

	pandoc --variable mainfont="Palatino" --variable sansfont="Century Gothic" --variable monofont="Consolas" --variable fontsize=12pt article.md --bibliography article.bib -o article.pdf
clean:
	rm article.pdf
