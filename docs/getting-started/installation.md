# Installation

## From PyPI
Flicker is available on [PyPI](https://pypi.org/project/flicker/). 
You can install it using `pip`.

```bash
pip install flicker
```

Use the `--user` if you're installing outside a virtual environment. 
Flicker is intended for Python 3.


## Within a Docker container
If you just want to try out Flicker (or even PySpark) without having to
install or setup anything, consider using the `flicker-playground`
available at 
[DockerHub](https://hub.docker.com/r/ankurio/flicker-playground). It comes 
with Java 8, PySpark, Flicker, Jupyter, IPython, and other python packages 
commonly needed when using Spark.

```bash
# From DockerHub (https://hub.docker.com/r/ankurio/flicker-playground) 
docker pull ankurio/flicker-playground
```

The corresponding Dockerfile is available inside the repository 
[here](https://github.com/ankur-gupta/flicker/blob/master/docker/Dockerfile), 
if you choose to build it yourself. The instructions to build the 
Docker image are available 
[here](https://github.com/ankur-gupta/flicker/blob/master/docker/Dockerfile).

After pulling (or building) the Docker image, you can mount data and 
run a Jupyter notebook or IPython from it.   
   
### Jupyter Notebook
```bash
# From $REPO_ROOT/docker in the host machine
docker run -it \
-p 8888:8888 \
-p 8080:8080 \
-p 4040:4040 \
--volume $PWD/notebooks:/home/neo/notebooks \
ankurio/flicker-playground
```

### IPython
```bash
# From $REPO_ROOT/docker in the host machine
docker run -it \
-p 8080:8080 \
-p 4040:4040 \
--volume $PWD/notebooks:/home/neo/notebooks \
ankurio/flicker-playground /bin/bash
# Run ipython on the shell prompt
# neo@95657e2ed2c6:~$ ipython 
``` 

The ports `8080` and `4040` 
below are for Spark's Web UI. In the following commands, we bind the Docker 
container's ports to host machine's ports. We will still need to setup the 
Spark session inside the container to use these ports. Typically, the 
Spark's Web UI can be accessed at http://localhost:8080 for 
Spark Master Web UI (which may not always be available) and at 
http://localhost:4040 (for a particular Spark app).


## Building from source
This may be needed if you want to develop and contribute code. 

### Clone the Repository
```bash
git clone git@github.com:ankur-gupta/flicker.git
```

### Install the requirements
It is recommended that you use a virtual environment for the rest of the steps.
```bash
# Within $REPO_ROOT
pip install -r requirements.txt
```

### Run tests 
```bash
# Within $REPO_ROOT
# Within a virtual environment with requirements installed 
python -m pytest flicker/tests
```

### Build the package
A script is available to build the package. This helps standardize what 
build artifacts are generated. 
```bash
# Within $REPO_ROOT
# Within a virtual environment with requirements installed 
./scripts/build-package
``` 
