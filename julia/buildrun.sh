docker build --tag docker.io/rvogt2/julopt .
docker run -it --entrypoint /bin/bash --rm -v $(pwd):/scratch  docker.io/rvogt2/julopt
