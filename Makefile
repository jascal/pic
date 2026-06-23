# PIC — build the formalism paper.
#   make paper   -> paper/pic_calculus.pdf   (needs a TeX distribution: latexmk + pdflatex + stmaryrd)
#   make clean   -> remove latexmk build artifacts

.PHONY: paper clean

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode pic_calculus.tex

clean:
	cd paper && latexmk -C
