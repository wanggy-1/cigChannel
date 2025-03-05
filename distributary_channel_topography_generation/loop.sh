for i in {1006..3000};
do
	./main
	mv ./discharge256/result.tiff ./discharge256/$i.tiff
done
